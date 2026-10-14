"""Synthesize short samples or UTF-8 text files with selected fine-tuned weights."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/task.json')
    parser.add_argument('--character')
    parser.add_argument('--task')
    parser.add_argument('--gpt-epoch', type=int)
    parser.add_argument('--sovits-epoch', type=int)
    parser.add_argument('--japanese-only', action='store_true')
    parser.add_argument('--text-files', nargs='+', help='UTF-8 plain text files to narrate in Chinese')
    args = parser.parse_args()
    if args.task and not args.character:
        parser.error('--task requires --character')
    from character_manager import CharacterStore
    store=CharacterStore()
    profile=None
    if args.character:
        _,task_id,profile,_=store.task(args.character,args.task)
        args.config=profile['config']
    if args.text_files and args.japanese_only:
        parser.error('--text-files generates Chinese narration; do not combine with --japanese-only.')
    cfg = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    selected=None
    for character_id,character in store.read()['characters'].items():
        for task_id,entry in character['tasks'].items():
            if store.local(entry['config'])==(ROOT/args.config).resolve():
                profile=entry
                choices=store.models(character_id,task_id)
                selected=store.defaults(character_id,task_id)
                for stage,epoch in [('gpt',args.gpt_epoch),('sovits',args.sovits_epoch)]:
                    if epoch is not None:
                        matches=[p for label,p in choices[stage] if label.startswith(f'{epoch}e · ')]
                        if len(matches)!=1: raise ValueError(f'Select one existing {stage} epoch {epoch}.')
                        selected[stage]=matches[0]
                    else:
                        label=next((label for label,p in choices[stage] if p==selected[stage]),'')
                        token=label.split('e · ',1)[0]
                        setattr(args,stage+'_epoch',int(token) if token.isdigit() else None)
                break
    if not selected:
        args.gpt_epoch = args.gpt_epoch or cfg['training']['gpt_epochs']
        args.sovits_epoch = args.sovits_epoch or cfg['training']['sovits_epochs']
    repo, work, output = (ROOT/cfg[k] for k in ('repository', 'work', 'output'))
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo/'GPT_SoVITS')]
    os.environ['version'] = cfg['version']
    os.environ['PATH'] = str(repo)+os.pathsep+os.environ.get('PATH', '')
    import torch
    import numpy as np
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS, TTS_Config
    weights = ROOT/cfg['weights']
    gpt = weights/'gpt'/f'{cfg["name"]}-e{args.gpt_epoch}.ckpt'
    sovits = sorted((weights/'sovits').glob(f'{cfg["name"]}_e{args.sovits_epoch}_s*.pth'))
    if selected:
        if not selected['gpt'] or not selected['sovits']: raise ValueError('Task has no paired weights.')
        gpt=Path(selected['gpt'])
        sovits=[Path(selected['sovits'])]
    if not gpt.is_file() or len(sovits) != 1:
        raise ValueError('Select one existing fine-tuned checkpoint per phase; no base-model fallback.')
    settings = {'device': 'cuda', 'is_half': True, 'version': cfg['version'],
        't2s_weights_path': str(gpt), 'vits_weights_path': str(sovits[0]),
        'bert_base_path': str(repo/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'),
        'cnhuhbert_base_path': str(repo/'GPT_SoVITS/pretrained_models/chinese-hubert-base')}
    reference=(profile or {}).get('reference')
    rows=[]
    if not reference:
        with (work/'review.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    references = []
    for row in rows:
        path = ROOT/row['audio_path']
        if 3 <= sf.info(path).duration <= 8 and row['reviewed'].lower() == 'true':
            references.append(row)
    if not reference and not references:
        raise ValueError('No reviewed 3–8 second reference.')
    reference = reference or references[0]
    prefix = 'narration' if args.text_files else 'samples'
    suffix=f's{args.sovits_epoch}_g{args.gpt_epoch}'
    if args.sovits_epoch is None or args.gpt_epoch is None:
        suffix=hashlib.sha256((str(sovits[0])+str(gpt)).encode()).hexdigest()[:12]
    target = output/f'{prefix}_{suffix}'
    target.mkdir(exist_ok=True)
    sentences = [('ja_daily', 'all_ja', '今日はいい天気ですね。少し散歩しませんか？')]
    if not args.japanese_only:
        sentences += [('zh_daily', 'all_zh', '今天的天气很好。要不要一起出去走走？'),
                      ('zh_question', 'all_zh', '你真的觉得，这样就可以了吗？')]
    sources = []
    if args.text_files:
        sentences = []
        for value in args.text_files:
            source = (ROOT/value).resolve()
            text = source.read_text(encoding='utf-8-sig').strip()
            if not text:
                raise ValueError(f'Empty narration text: {source}')
            sentences.append((source.stem, 'all_zh', text))
            sources.append({'path': str(source.relative_to(ROOT)),
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        if len({name for name, _, _ in sentences}) != len(sentences):
            raise ValueError('Narration input filenames must have distinct stems.')
    tts = TTS(TTS_Config({'custom': settings}))
    results = []
    for name, language, text in sentences:
        if args.text_files:
            (target/(name+'.txt')).write_text(text+'\n', encoding='utf-8')
        print('Synthesizing:', name, len(text), 'characters', flush=True)
        request = {'text': text, 'text_lang': language,
            'ref_audio_path': str(ROOT/reference['audio_path']), 'prompt_lang': 'all_'+reference.get('language',cfg['language']), 'prompt_text': reference['text'],
            'top_k': 15, 'top_p': 1.0, 'temperature': 1.0, 'text_split_method': 'cut5', 'batch_size': 1,
            'speed_factor': 1.0, 'seed': cfg['seed'], 'parallel_infer': False, 'return_fragment': False}
        chunks = list(tts.run(request))
        if not chunks:
            raise ValueError('No synthesized audio.')
        rate = chunks[0][0]
        if any(sr != rate for sr, _ in chunks):
            raise ValueError('Inconsistent output sample rates.')
        audio = np.concatenate([wave for _, wave in chunks])
        if not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError('Invalid synthesized audio.')
        path = target/(name+'.wav')
        # The upstream generator returns PCM16 arrays.
        sf.write(path, audio, rate, subtype='PCM_16')
        results.append({'name': name, 'text': text, 'language': language,
                        'path': str(path.relative_to(ROOT)), 'seconds': len(audio)/rate,
                        'sample_rate': rate, 'finite_nonzero': True})
        print('Generated:', path, flush=True)
        (target/'manifest.json').write_text(json.dumps({'settings': settings, 'reference': reference,
            'inference_controls': {k: v for k, v in request.items() if k not in ('text', 'text_lang')},
            'epochs': {'sovits': args.sovits_epoch, 'gpt': args.gpt_epoch},
            'sources': sources, 'results': results,
            'purpose': 'long-text narration trial' if args.text_files else 'checkpoint samples for listening comparison'},
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Samples:', target)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    main()
