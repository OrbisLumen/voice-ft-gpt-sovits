"""Compare two epochs of one TTS stage with the other stage and inputs fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import tts_utils as utils

ROOT = utils.ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_pair(first, second, phase='sovits'):
    if phase not in ('sovits', 'gpt'):
        raise ValueError('Comparison phase must be sovits or gpt.')
    if first['reference'] != second['reference']:
        raise ValueError('Reference audio/text differs; do not combine an uncontrolled comparison.')
    changing_weight = 'vits_weights_path' if phase == 'sovits' else 't2s_weights_path'
    controls_a = {k: v for k, v in first['settings'].items() if k != changing_weight}
    controls_b = {k: v for k, v in second['settings'].items() if k != changing_weight}
    if controls_a != controls_b:
        raise ValueError('Fixed-stage/frontend/inference settings differ.')
    if first.get('inference_controls') and second.get('inference_controls'):
        if first['inference_controls'] != second['inference_controls']:
            raise ValueError('Sampling controls differ.')
    rows_a = {r['name']: r for r in first['results']}
    rows_b = {r['name']: r for r in second['results']}
    if not rows_a or set(rows_a) != set(rows_b):
        raise ValueError('Comparison groups differ.')
    for name in rows_a:
        if any(rows_a[name][k] != rows_b[name][k] for k in ('text', 'language')):
            raise ValueError(f'Text/language differs: {name}')
    return rows_a, rows_b


def main():
    import numpy as np
    import soundfile as sf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/task.json')
    parser.add_argument('--first', type=int, default=10)
    parser.add_argument('--second', type=int, default=20)
    parser.add_argument('--gpt', type=int, default=5)
    parser.add_argument('--phase', choices=['sovits', 'gpt'], default='sovits')
    parser.add_argument('--sovits', type=int, default=20, help='Fixed SoVITS epoch when comparing GPT')
    args = parser.parse_args()
    cfg = utils.read_json(ROOT/args.config)
    output = ROOT/cfg['output']
    pairs = [(epoch, args.gpt) if args.phase == 'sovits' else (args.sovits, epoch)
             for epoch in (args.first, args.second)]
    manifests = [utils.read_json(output/f'samples_s{s}_g{g}/manifest.json') for s, g in pairs]
    for manifest, (s, g) in zip(manifests, pairs):
        if manifest.get('epochs') and manifest['epochs'] != {'sovits': s, 'gpt': g}:
            raise ValueError('Sample epoch metadata differs from selected comparison.')
    rows_a, rows_b = validate_pair(*manifests, phase=args.phase)
    folder = (f'comparison_s{args.first}_s{args.second}_g{args.gpt}' if args.phase == 'sovits'
              else f'comparison_s{args.sovits}_g{args.first}_g{args.second}')
    destination = output/folder
    destination.mkdir(parents=True, exist_ok=True)
    normalization = {'repository': cfg['repository'],
                     'final': {'sample_rate': 48000, 'loudness_lufs': -22, 'true_peak_db': -1.5}}
    results = []
    for name in rows_a:
        pieces, parts = [], []
        for (s, g), rows in zip(pairs, (rows_a, rows_b)):
            raw = ROOT/rows[name]['path']
            normalized = destination/'normalized'/f'{name}_s{s}_g{g}.wav'
            utils.finalize(normalization, raw, normalized)
            audio, rate = sf.read(normalized, dtype='float32')
            if rate != 48000 or audio.ndim != 1 or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError(f'Invalid normalized comparison: {normalized}')
            parts.append({'sovits_epoch': s, 'gpt_epoch': g, 'raw': rows[name]['path'],
                          'raw_sha256': sha256(raw), 'normalized': str(normalized.relative_to(ROOT)),
                          'start_seconds': sum(len(piece)/rate for piece in pieces), 'seconds': len(audio)/rate})
            pieces.extend([audio, np.zeros(rate, dtype='float32')])
        combined = destination/f'{name}_{args.first}_then_{args.second}.wav'
        sf.write(combined, np.concatenate(pieces[:-1]), 48000, subtype='PCM_16')
        results.append({'name': name, 'text': rows_a[name]['text'], 'language': rows_a[name]['language'],
                        'combined': str(combined.relative_to(ROOT)), 'parts': parts})
    models = []
    for (s, g), manifest in zip(pairs, manifests):
        model_pair = {'sovits_epoch': s, 'gpt_epoch': g}
        for stage, key in (('sovits', 'vits_weights_path'), ('gpt', 't2s_weights_path')):
            model = Path(manifest['settings'][key])
            model_pair[stage] = {'path': str(model.relative_to(ROOT)), 'sha256': sha256(model)}
        models.append(model_pair)
    record = {'order': [args.first, args.second], 'phase': args.phase, 'gap_seconds': 1,
        'reference': manifests[0]['reference'], 'reference_sha256': sha256(ROOT/manifests[0]['reference']['audio_path']),
        'models': models,
        'normalization': normalization['final'], 'results': results}
    utils.write_json(destination/'manifest.json', record)
    changing = 'SoVITS' if args.phase == 'sovits' else 'GPT'
    fixed = f'GPT {args.gpt}' if args.phase == 'sovits' else f'SoVITS {args.sovits}'
    lines = [f'# {changing} {args.first} / {args.second}轮比较（{fixed}轮固定）', '',
        f'每条顺序：{args.first}轮 → 1秒静音 → {args.second}轮。统一-22 LUFS / 48kHz；参考音频、文本、{fixed}轮相同。', '',
        '关注音色相似度、自然度、中文清晰度和喜欢的日语口音；不以轮数直接判定优劣。', '']
    for row in results:
        lines.append(f'- {row["name"]}：{row["text"]}\n  [连续试听]({(ROOT/row["combined"]).as_posix()})')
    (destination/'试听.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Comparison ready:', destination)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
