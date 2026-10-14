"""Local GPT-SoVITS task: selected audio -> draft ASR -> review -> features -> fine-tune."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import tts_utils as utils

ROOT = utils.ROOT
DEFAULT = 'config/task.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_signature(config):
    return utils.fingerprint({key: config[key] for key in
        ('name', 'source_manifest', 'selected_only', 'dataset', 'language', 'min_seconds', 'max_seconds')})


def read_review(path, manifest):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {r['name']: r for r in manifest['clips'] if r['eligible']}
    if len(rows) != len(expected) or {r['name'] for r in rows} != set(expected):
        raise ValueError('Review CSV must contain every eligible clip exactly once.')
    for row in rows:
        if row['reviewed'].strip().lower() not in ('true', '1', 'yes'):
            raise ValueError(f'Transcript not reviewed: {row["name"]}')
        if not row['text'].strip() or any(x in row['text'] for x in ('|', '\n', '\r')):
            raise ValueError(f'Invalid transcript: {row["name"]}')
        if row['language'] != 'ja':
            raise ValueError('This task is Japanese target speech; do not label it Chinese.')
    return rows


class Task:
    def __init__(self, config):
        self.config_path = ROOT/config
        self.cfg = utils.read_json(ROOT/config)
        self.repo = ROOT/self.cfg['repository']
        self.work = ROOT/self.cfg['work']
        self.dataset = ROOT/self.cfg['dataset']
        self.output = ROOT/self.cfg['output']
        self.weights = ROOT/self.cfg['weights']
        self.backend = self.repo/'.venv/Scripts/python.exe'
        self.pretrained = self.repo/'GPT_SoVITS/pretrained_models'
        self.experiment = self.work/'features'
        self.work.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def exclusive(self):
        import msvcrt
        with (self.work/'active.lock').open('a+b') as stream:
            stream.seek(0)
            stream.write(b'0')
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another TTS task is running; do not start parallel stages.') from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def environment(self, **extra):
        return dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                    MPLBACKEND='Agg', version=self.cfg['version'], hz='25hz',
                    PATH=str(self.repo)+os.pathsep+os.environ.get('PATH', ''),
                    PYTHONPATH=str(self.repo)+os.pathsep+str(self.repo/'GPT_SoVITS'),
                    _CUDA_VISIBLE_DEVICES='0', USE_LIBUV='0', **{k: str(v) for k, v in extra.items()})

    def invoke(self, label, arguments, **env):
        logdir = self.work/'logs'
        logdir.mkdir(exist_ok=True)
        log = logdir/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+label+'.log')
        command = [str(self.backend), *map(str, arguments)]
        print('>', subprocess.list2cmdline(command), '\nLog:', log, flush=True)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=self.repo, env=self.environment(**env),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line.rstrip(), flush=True)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                    process.wait()
                raise
        if code:
            raise RuntimeError(f'{label} failed ({code}); inspect {log}')

    def doctor(self):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        if head != self.cfg['commit']:
            raise ValueError('Upstream revision differs from task configuration.')
        self.invoke('cuda', [ROOT/'scripts/check_cuda.py'])
        self.invoke('imports', ['-c',
            "import torch,torchaudio,transformers,pytorch_lightning,pyopenjtalk,faster_whisper; "
            "from text.japanese import g2p; print('Japanese phones:',g2p('こんにちは。')); "
            "print('Imports OK')"])
        required = ['s1v3.ckpt', 'v2Pro/s2Gv2Pro.pth', 'v2Pro/s2Dv2Pro.pth',
                    'sv/pretrained_eres2netv2w24s4ep4.ckpt', 'chinese-hubert-base/pytorch_model.bin',
                    'chinese-roberta-wwm-ext-large/pytorch_model.bin']
        missing = [p for p in required if not (self.pretrained/p).is_file()]
        utils.write_json(self.work/'doctor.json', {'commit': head, 'missing_models': missing,
            'python': str(self.backend), 'training_vram_not_yet_verified': True})
        print('Missing models:', missing)
        if missing:
            raise ValueError('Run setup_tts.py --models-only before feature extraction.')

    def prepare(self):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        sys.path.insert(0, str(self.repo/'tools'))
        from slicer2 import Slicer
        manifest_path = self.work/'dataset_manifest.json'
        if manifest_path.exists():
            manifest = utils.read_json(manifest_path)
            legacy = 'data_sha256' not in manifest and manifest['config_sha256'] == utils.fingerprint(self.cfg)
            if not legacy and manifest.get('data_sha256') != data_signature(self.cfg):
                raise ValueError('Task data parameters changed; use another task name.')
            for row in manifest['sources']:
                if digest(ROOT/row['path']) != row['sha256']:
                    raise ValueError('Original source changed.')
            for row in manifest['clips']:
                if digest(ROOT/row['path']) != row['sha256']:
                    raise ValueError('Prepared clip changed.')
            if legacy:
                manifest['data_sha256'] = data_signature(self.cfg)
                utils.write_json(manifest_path, manifest)
            print('Reuse verified TTS dataset:', len(manifest['clips']))
            return
        selected = utils.read_json(ROOT/self.cfg['source_manifest'])['files']
        sources = [r for r in selected if r['split'] == 'train' and
                   (r['user_selected'] or not self.cfg['selected_only'])]
        clips, source_rows = [], []
        wavdir = self.dataset/'wavs'
        wavdir.mkdir(parents=True, exist_ok=True)
        if any(wavdir.glob('*.wav')):
            raise ValueError('Existing clips without manifest; inspect before overwriting.')
        slicer = Slicer(sr=32000, threshold=-40, min_length=2000, min_interval=300,
                        hop_size=10, max_sil_kept=300)
        for row in sources:
            source = ROOT/row['destination']
            if digest(source) != row['sha256']:
                raise ValueError(f'Source hash differs: {source}')
            audio, rate = sf.read(source, dtype='float32')
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError('Require finite mono target audio.')
            audio = resample_poly(audio, 32000, rate).astype('float32')
            source_rows.append({'path': str(source.relative_to(ROOT)), 'sha256': row['sha256'],
                                'original': row['source']})
            for number, piece in enumerate(slicer.slice(audio)):
                # Upstream returns a bare waveform only for clips shorter than min_length.
                chunk, start, end = (piece, 0, len(piece)) if isinstance(piece, np.ndarray) else piece
                name = f'{source.stem}_{number:03d}.wav'
                path = wavdir/name
                sf.write(path, chunk, 32000, subtype='PCM_16')
                seconds = len(chunk)/32000
                eligible = self.cfg['min_seconds'] <= seconds <= self.cfg['max_seconds'] and np.any(chunk)
                clips.append({'name': name, 'path': str(path.relative_to(ROOT)), 'source': str(source.relative_to(ROOT)),
                    'start_sample_32k': int(start), 'end_sample_32k': int(end), 'seconds': round(seconds, 4),
                    'eligible': bool(eligible), 'sha256': digest(path)})
        if not any(r['eligible'] for r in clips):
            raise ValueError('No eligible TTS clips.')
        # Retain source-level holdout outside the training list; no random sibling slices leak into training.
        holdout = [r for r in selected if r['split'] == 'holdout']
        manifest = {'config_sha256': utils.fingerprint(self.cfg), 'data_sha256': data_signature(self.cfg), 'sample_rate': 32000,
            'sources': source_rows, 'clips': clips, 'holdout_sources': holdout,
            'eligible_seconds': sum(r['seconds'] for r in clips if r['eligible'])}
        utils.write_json(manifest_path, manifest)
        print('Prepared sources:', len(sources), 'clips:', len(clips),
              'eligible seconds:', round(manifest['eligible_seconds'], 2))

    def transcribe(self, limit=None):
        import torch
        import soundfile as sf
        from scipy.signal import resample_poly
        from faster_whisper import WhisperModel
        modeldir = self.repo/'tools/asr/models/faster-whisper-small'
        if not (modeldir/'model.bin').is_file():
            raise ValueError('Local ASR model missing; run setup_tts.py --models-only.')
        dll = Path(torch.__file__).parent/'lib'
        dll_handle = os.add_dll_directory(str(dll)) if os.name == 'nt' else None
        model = WhisperModel(str(modeldir), device=self.cfg['asr']['device'],
                             compute_type=self.cfg['asr']['compute_type'], local_files_only=True)
        manifest = utils.read_json(self.work/'dataset_manifest.json')
        drafts = self.work/'asr_drafts'
        drafts.mkdir(exist_ok=True)
        eligible = [r for r in manifest['clips'] if r['eligible']]
        for number, row in enumerate(eligible[:limit] if limit else eligible, 1):
            result = drafts/(row['name']+'.json')
            if result.exists():
                continue
            # Pass a 16k waveform directly; newer PyAV removed an argument used by Faster Whisper 1.2.
            waveform, sample_rate = sf.read(ROOT/row['path'], dtype='float32')
            waveform = resample_poly(waveform, 16000, sample_rate).astype('float32')
            segments, info = model.transcribe(waveform, language='ja', beam_size=self.cfg['asr']['beam_size'],
                vad_filter=True, condition_on_previous_text=False)
            pieces = list(segments)
            text = ''.join(p.text for p in pieces).strip()
            utils.write_json(result, {'name': row['name'], 'text': text, 'audio_sha256': row['sha256'],
                'language_probability': info.language_probability,
                'avg_logprob': sum(p.avg_logprob for p in pieces)/len(pieces) if pieces else None})
            print(f'ASR {number}/{len(eligible)} {row["name"]}: {text}', flush=True)
        review = self.work/'review.csv'
        existing = {}
        if review.exists():
            with review.open(encoding='utf-8-sig', newline='') as stream:
                existing = {r['name']: r for r in csv.DictReader(stream)}
        with review.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['name', 'language', 'text', 'reviewed', 'audio_path'])
            writer.writeheader()
            for row in eligible:
                result = drafts/(row['name']+'.json')
                if row['name'] in existing and existing[row['name']]['text'].strip():
                    writer.writerow(existing[row['name']])
                else:
                    writer.writerow({'name': row['name'], 'language': 'ja',
                        'text': utils.read_json(result)['text'] if result.exists() else '', 'reviewed': 'false',
                        'audio_path': row['path']})
        print('Review transcripts:', review)

    def audit(self):
        manifest = utils.read_json(self.work/'dataset_manifest.json')
        rows = read_review(self.work/'review.csv', manifest)
        by_name = {r['name']: r for r in manifest['clips']}
        for row in rows:
            if digest(ROOT/by_name[row['name']]['path']) != by_name[row['name']]['sha256']:
                raise ValueError('Training audio changed since preparation.')
        text = '\n'.join(f'{ROOT/by_name[r["name"]]["path"]}|{self.cfg["name"]}|ja|{r["text"].strip()}' for r in rows)+'\n'
        path = self.work/'train.list'
        if path.exists() and path.read_text(encoding='utf-8') != text and (self.experiment/'2-name2text.txt').exists():
            raise ValueError('Reviewed text changed after feature preparation; use another task name.')
        path.write_text(text, encoding='utf-8')
        utils.write_json(self.work/'transcript_audit.json', {'clips': len(rows), 'review_sha256': digest(self.work/'review.csv'),
            'train_list_sha256': digest(path), 'all_reviewed': True})
        return rows

    def features(self):
        rows = self.audit()
        env = {'inp_text': self.work/'train.list', 'inp_wav_dir': self.dataset/'wavs',
            'exp_name': self.cfg['name'], 'i_part': '0', 'all_parts': '1', 'opt_dir': self.experiment,
            'bert_pretrained_dir': self.pretrained/'chinese-roberta-wwm-ext-large',
            'cnhubert_base_dir': self.pretrained/'chinese-hubert-base', 'is_half': 'True',
            'pretrained_s2G': self.pretrained/'v2Pro/s2Gv2Pro.pth',
            's2config_path': self.repo/'GPT_SoVITS/configs/s2v2Pro.json',
            'sv_path': self.pretrained/'sv/pretrained_eres2netv2w24s4ep4.ckpt'}
        for script in ('1-get-text.py', '2-get-hubert-wav32k.py', '2-get-sv.py', '3-get-semantic.py'):
            self.invoke(script[:-3], [self.repo/'GPT_SoVITS/prepare_datasets'/script], **env)
        textpart = self.experiment/'2-name2text-0.txt'
        semanticpart = self.experiment/'6-name2semantic-0.tsv'
        (self.experiment/'2-name2text.txt').write_bytes(textpart.read_bytes())
        (self.experiment/'6-name2semantic.tsv').write_text('item_name\tsemantic_audio\n'+semanticpart.read_text(encoding='utf-8'), encoding='utf-8')
        expected = {r['name'] for r in rows}
        observed = {line.split('\t')[0] for line in textpart.read_text(encoding='utf-8').splitlines() if line.strip()}
        semantics = {line.split('\t')[0] for line in semanticpart.read_text(encoding='utf-8').splitlines() if line.strip()}
        if observed != expected or semantics != expected:
            raise ValueError('Upstream omitted clips; do not train incomplete features.')
        for name in expected:
            for relative in (f'4-cnhubert/{name}.pt', f'5-wav32k/{name}', f'7-sv_cn/{name}.pt'):
                if not (self.experiment/relative).is_file():
                    raise ValueError(f'Missing feature: {relative}')
        self.validate_features(expected)
        utils.write_json(self.work/'features_ready.json', {'train_list_sha256': digest(self.work/'train.list'), 'clips': len(expected), 'all_finite': True})
        self.training_configs()

    def validate_features(self, expected):
        import torch
        import soundfile as sf
        import numpy as np
        for name in expected:
            hubert = torch.load(self.experiment/f'4-cnhubert/{name}.pt', map_location='cpu', weights_only=True)
            speaker = torch.load(self.experiment/f'7-sv_cn/{name}.pt', map_location='cpu', weights_only=True)
            if hubert.ndim != 3 or hubert.shape[:2] != (1, 768) or not torch.isfinite(hubert).all():
                raise ValueError(f'Invalid HuBERT feature: {name}')
            if speaker.shape != (1, 20480) or not torch.isfinite(speaker).all():
                raise ValueError(f'Invalid speaker embedding: {name}')
            audio, rate = sf.read(self.experiment/f'5-wav32k/{name}')
            if rate != 32000 or audio.ndim != 1 or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError(f'Invalid training audio: {name}')

    def training_configs(self):
        import yaml
        train = self.cfg['training']
        self.output.mkdir(parents=True, exist_ok=True)
        (self.experiment/f'logs_s2_{self.cfg["version"]}').mkdir(parents=True, exist_ok=True)
        for phase in ('sovits', 'gpt'):
            (self.weights/phase).mkdir(parents=True, exist_ok=True)
        s2 = utils.read_json(self.repo/'GPT_SoVITS/configs/s2v2Pro.json')
        s2['train'].update(batch_size=train['sovits_batch_size'], epochs=train['sovits_epochs'],
            fp16_run=True, grad_ckpt=train['gradient_checkpointing'], text_low_lr_rate=0.4,
            pretrained_s2G=str(self.pretrained/'v2Pro/s2Gv2Pro.pth'), pretrained_s2D=str(self.pretrained/'v2Pro/s2Dv2Pro.pth'),
            if_save_latest=True, if_save_every_weights=True, save_every_epoch=train['save_every_epoch'], gpu_numbers='0')
        s2['model']['version'] = self.cfg['version']
        s2['data']['exp_dir'] = s2['s2_ckpt_dir'] = str(self.experiment)
        s2.update(save_weight_dir=str(self.weights/'sovits'), name=self.cfg['name'], version=self.cfg['version'])
        utils.write_json(self.work/'train_sovits.json', s2)
        s1 = yaml.safe_load((self.repo/'GPT_SoVITS/configs/s1longer-v2.yaml').read_text(encoding='utf-8'))
        s1['train'].update(batch_size=train['gpt_batch_size'], epochs=train['gpt_epochs'], precision='16-mixed',
            save_every_n_epoch=train['save_every_epoch'], if_save_every_weights=True, if_save_latest=True, if_dpo=False,
            half_weights_save_dir=str(self.weights/'gpt'), exp_name=self.cfg['name'])
        s1['data'].update(num_workers=train['gpt_num_workers'], max_sec=self.cfg['max_seconds'])
        s1.update(pretrained_s1=str(self.pretrained/'s1v3.ckpt'),
            train_semantic_path=str(self.experiment/'6-name2semantic.tsv'),
            train_phoneme_path=str(self.experiment/'2-name2text.txt'), output_dir=str(self.work/'logs_s1_v2Pro'))
        (self.work/'train_gpt.yaml').write_text(yaml.safe_dump(s1, allow_unicode=True), encoding='utf-8')

    def train(self, phase, epochs=None):
        rows = self.audit()
        ready = utils.read_json(self.work/'features_ready.json')
        if ready['train_list_sha256'] != digest(self.work/'train.list'):
            raise ValueError('Features are stale for current transcript list.')
        self.validate_features({r['name'] for r in rows})
        target_epoch = epochs if epochs is not None else self.cfg['training'][f'{phase}_epochs']
        target_pattern = f'{self.cfg["name"]}_e{target_epoch}_s*.pth' if phase == 'sovits' else f'{self.cfg["name"]}-e{target_epoch}.ckpt'
        existing = list((self.weights/phase).glob(target_pattern))
        if len(existing) == 1:
            print('Reuse completed training target:', existing[0], flush=True)
            return
        self.training_configs()
        if epochs is not None:
            if not 1 <= epochs <= 100:
                raise ValueError('Training epoch override must be between 1 and 100.')
            if phase == 'sovits':
                path = self.work/'train_sovits.json'
                config = utils.read_json(path)
                config['train']['epochs'] = epochs
                utils.write_json(path, config)
            else:
                import yaml
                path = self.work/'train_gpt.yaml'
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                config['train']['epochs'] = epochs
                path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        started = time.time()
        if phase == 'sovits':
            self.invoke('train_sovits', [self.repo/'GPT_SoVITS/s2_train.py', '--config', self.work/'train_sovits.json'])
        else:
            self.invoke('train_gpt', [self.repo/'GPT_SoVITS/s1_train.py', '--config_file', self.work/'train_gpt.yaml'])
        pattern = '*.pth' if phase == 'sovits' else '*.ckpt'
        produced = [p for p in (self.weights/phase).glob(pattern) if p.stat().st_mtime >= started - 2]
        if not produced:
            raise ValueError('Training returned without inference weights.')
        print('Saved inference weights:', *produced, sep='\n')

    def status(self):
        for path in ('doctor.json', 'dataset_manifest.json', 'transcript_audit.json', 'features_ready.json'):
            file = self.work/path
            print(path, 'exists' if file.exists() else 'pending')
        print('Review:', self.work/'review.csv')
        print('Output:', self.output)
        print('Weights:', self.weights)

    def sample(self):
        self.invoke('sample', [ROOT/'scripts/tts_sample.py', '--config', self.config_path,
            '--sovits-epoch', self.cfg['training']['sovits_epochs'],
            '--gpt-epoch', self.cfg['training']['gpt_epochs']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['doctor', 'prepare', 'transcribe', 'audit', 'features', 'train-sovits', 'train-gpt', 'sample', 'status'])
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--limit', type=int, help='Only transcribe this many clips for a smoke check')
    parser.add_argument('--epochs', type=int, help='Total epoch limit override for a short training check')
    args = parser.parse_args()
    task = Task(args.config)
    if Path(sys.executable).resolve() != task.backend.resolve():
        return subprocess.run([str(task.backend), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT).returncode
    with task.exclusive():
        if args.stage.startswith('train-'):
            task.train(args.stage.split('-')[1], args.epochs)
        elif args.stage == 'transcribe':
            task.transcribe(args.limit)
        else:
            getattr(task, args.stage)()
    return 0


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
