"""Install the pinned GPT-SoVITS runtime and public pretrained models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=ROOT):
    print('>', subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=cwd, check=True,
                   env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg'))


def windows_requirements(repo, work):
    lines = (repo/'requirements.txt').read_text(encoding='utf-8').splitlines()
    lines = [x for x in lines if not x.startswith(('--no-binary=', 'pyopenjtalk', 'jieba_fast', 'fastapi'))]
    # PyPI wheels avoid requiring a system-wide MSVC installation on this machine.
    lines += ['pyopenjtalk-plus>=0.4,<1', 'torch==2.11.0', 'torchaudio==2.11.0', 'resampy==0.4.3',
              'fastapi==0.115.6', 'starlette==0.41.3']
    lines += (repo/'extra-req.txt').read_text(encoding='utf-8').splitlines()
    path = work/'requirements-windows.txt'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for relative in ('GPT_SoVITS/text/chinese.py', 'GPT_SoVITS/text/chinese2.py', 'GPT_SoVITS/text/tone_sandhi.py'):
        source = repo/relative
        text = source.read_text(encoding='utf-8')
        if 'jieba_fast' in text:
            backup = work/'upstream_originals'/relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_text(text, encoding='utf-8')
            source.write_text(text.replace('jieba_fast', 'jieba'), encoding='utf-8')
    # torchaudio 2.11 delegates decoding to TorchCodec, absent from this Windows environment.
    # These paths only need ordinary local reference WAVs; SoundFile supplies the same tensor layout.
    helper = repo/'tools/local_audio_io.py'
    helper.write_text('import soundfile as sf\nimport torch\n\ndef load_audio_tensor(path):\n'
                      '    audio, rate = sf.read(path, dtype="float32", always_2d=True)\n'
                      '    return torch.from_numpy(audio.T.copy()), rate\n', encoding='utf-8')
    for relative in ('GPT_SoVITS/prepare_datasets/2-get-sv.py', 'GPT_SoVITS/TTS_infer_pack/TTS.py', 'GPT_SoVITS/inference_webui.py'):
        source = repo/relative
        text = source.read_text(encoding='utf-8')
        if 'torchaudio.load(' in text:
            backup = work/'upstream_originals'/relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_text(text, encoding='utf-8')
            text = text.replace('import torchaudio\n', 'import torchaudio\nfrom tools.local_audio_io import load_audio_tensor\n')
            source.write_text(text.replace('torchaudio.load(', 'load_audio_tensor('), encoding='utf-8')
    source = repo/'GPT_SoVITS/AR/data/data_module.py'
    text = source.read_text(encoding='utf-8')
    if 'persistent_workers=True' in text:
        backup = work/'upstream_originals/GPT_SoVITS/AR/data/data_module.py'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(text, encoding='utf-8')
        text = text.replace('persistent_workers=True', 'persistent_workers=self.num_workers > 0')
        text = text.replace('prefetch_factor=16', 'prefetch_factor=16 if self.num_workers > 0 else None')
        source.write_text(text, encoding='utf-8')
    # One GPU needs no distributed process group on Windows.
    for relative in ('GPT_SoVITS/s1_train.py', 'GPT_SoVITS/AR/data/bucket_sampler.py'):
        source = repo/relative
        text = source.read_text(encoding='utf-8')
        original = text
        if relative.endswith('s1_train.py'):
            text = text.replace('if torch.cuda.is_available()\n        else "auto",',
                                'if torch.cuda.device_count() > 1\n        else "auto",')
            text = text.replace('    trainer.fit(model, data_module, ckpt_path=ckpt_path)\n',
                '    with torch.serialization.safe_globals([type(Path())]):\n'
                '        trainer.fit(model, data_module, ckpt_path=ckpt_path, weights_only=True)\n')
        else:
            text = text.replace('dist.get_world_size() if torch.cuda.is_available() else 1',
                                'dist.get_world_size() if dist.is_initialized() else 1')
            text = text.replace('dist.get_rank() if torch.cuda.is_available() else 0',
                                'dist.get_rank() if dist.is_initialized() else 0')
        if text != original:
            backup = work/'upstream_originals'/relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_text(original, encoding='utf-8')
            source.write_text(text, encoding='utf-8')
    source = repo/'tools/portable_runtime.py'
    text = source.read_text(encoding='utf-8')
    replacement = ('FFPROBE_EXE = RUNTIME_DIR / "ffprobe.exe"\n'
        'if not FFMPEG_EXE.is_file():\n    FFMPEG_EXE = PROJECT_ROOT / "ffmpeg.exe"\n'
        'if not FFPROBE_EXE.is_file():\n    FFPROBE_EXE = PROJECT_ROOT / "ffprobe.exe"')
    if 'if not FFMPEG_EXE.is_file():' not in text:
        backup = work/'upstream_originals/tools/portable_runtime.py'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(text, encoding='utf-8')
        source.write_text(text.replace('FFPROBE_EXE = RUNTIME_DIR / "ffprobe.exe"', replacement), encoding='utf-8')
    # Keep the official inference UI, with task defaults supplied by the local launcher.
    source = repo/'GPT_SoVITS/inference_webui_fast.py'
    text = source.read_text(encoding='utf-8')
    replacements = {
        'type="filepath")': 'type="filepath", value=os.environ.get("local_ref_audio"))',
        'label=i18n("主参考音频的文本"), value="",': 'label=i18n("主参考音频的文本"), value=os.environ.get("local_ref_text", ""),',
        'label=i18n("主参考音频的语种"), choices=list(dict_language.keys()), value=i18n("中文"),':
            'label=i18n("主参考音频的语种"), choices=list(dict_language.keys()), value=i18n("日文") if os.environ.get("local_ref_audio") else i18n("中文"),',
        'value=1 if model_version in V5_VERSIONS else 20': 'value=1',
        'gr.Number(label=i18n("随机种子"), value=-1)': 'gr.Number(label=i18n("随机种子"), value=int(os.environ.get("local_seed", -1)))',
        'gr.Checkbox(label=i18n("保持随机"), value=True,': 'gr.Checkbox(label=i18n("保持随机"), value=False,',
    }
    changed=text
    for before,after in replacements.items():
        changed=changed.replace(before,after)
    if changed!=text:
        backup=work/'upstream_originals/GPT_SoVITS/inference_webui_fast.py'
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists(): backup.write_text(text,encoding='utf-8')
        source.write_text(changed,encoding='utf-8')
    from patch_character_ui import patch_character_ui
    patch_character_ui(repo,work)
    return path


def download_chinese_frontend(repo, work):
    from huggingface_hub import HfApi, hf_hub_download
    identity_path = work/'chinese_frontend.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.exists() else {}
    name = 'XXXXRT/GPT-SoVITS-Pretrained'
    revision = identity.get('revision') or HfApi().model_info(name).sha
    path = hf_hub_download(name, 'G2PWModel.zip', revision=revision, local_dir=work/'downloads')
    destination = (repo/'GPT_SoVITS/text').resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not (destination/info.filename).resolve().is_relative_to(destination):
                raise ValueError('Unsafe frontend archive path.')
        archive.extractall(destination)
    if not (destination/'G2PWModel/g2pW.onnx').is_file():
        raise ValueError('Chinese frontend model missing after extraction.')
    identity.update(repository=name, revision=revision, filename='G2PWModel.zip')
    identity_path.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
    # Current upstream language segmentation also requires the public FastText language model.
    import urllib.request
    cache = repo/'GPT_SoVITS/pretrained_models/fast_langdetect'
    cache.mkdir(parents=True, exist_ok=True)
    model = cache/'lid.176.bin'
    if not model.is_file():
        temporary = cache/'lid.176.bin.partial'
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', temporary)
        temporary.replace(model)


def download_models(repo, work, spec):
    from huggingface_hub import HfApi, snapshot_download
    tasks = [
        ('lj1995/GPT-SoVITS', repo/'GPT_SoVITS/pretrained_models',
         ['s1v3.ckpt', 'v2Pro/s2Gv2Pro.pth', 'v2Pro/s2Dv2Pro.pth', 'sv/pretrained_eres2netv2w24s4ep4.ckpt']),
        ('hfl/chinese-roberta-wwm-ext-large', repo/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
         ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']),
        ('TencentGameMate/chinese-hubert-base', repo/'GPT_SoVITS/pretrained_models/chinese-hubert-base',
         ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']),
        (spec['asr']['repository'], repo/'tools/asr/models/faster-whisper-small',
         ['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt', 'preprocessor_config.json']),
    ]
    record_path = work/'model_downloads.json'
    pinned = ROOT/'config/model_sources.json'
    source = record_path if record_path.exists() else pinned
    record = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
    for name, destination, patterns in tasks:
        revision = record.get(name, {}).get('revision') or HfApi().model_info(name).sha
        print(f'Download {name}@{revision}', flush=True)
        record[name] = {'revision': revision, 'directory': str(destination.relative_to(ROOT)), 'patterns': patterns}
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        snapshot_download(name, revision=revision, local_dir=str(destination), allow_patterns=patterns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/task.json')
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--skip-models', action='store_true')
    parser.add_argument('--chinese-frontend', action='store_true', help='Download the official G2PW frontend for Chinese synthesis')
    args = parser.parse_args()
    spec = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    repo, work = ROOT/spec['repository'], ROOT/spec['work']
    work.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        run(['git', 'clone', spec.get('url', 'https://github.com/RVC-Boss/GPT-SoVITS.git'), repo])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != spec['commit']:
        raise ValueError(f'GPT-SoVITS commit differs: {head}; preserve local changes before switching.')
    backend = repo/'.venv/Scripts/python.exe'
    if not backend.exists():
        if sys.version_info[:2] != (3, 12):
            raise ValueError('Use Python 3.12 on this Windows CUDA machine.')
        run([sys.executable, '-m', 'venv', repo/'.venv'])
    if not args.models_only:
        run([backend, '-m', 'pip', 'install', 'torch==2.11.0', 'torchaudio==2.11.0',
             '--index-url', 'https://download.pytorch.org/whl/cu128'])
        requirements = windows_requirements(repo, work)
        lock = ROOT/'config/requirements-windows.lock.txt'
        run([backend, '-m', 'pip', 'install', '-r', lock if lock.exists() else requirements])
        run([backend, '-m', 'pip', 'check'])
        freeze = subprocess.check_output([str(backend), '-m', 'pip', 'freeze'], text=True)
        (work/'requirements-installed.txt').write_text(freeze, encoding='utf-8')
    for filename in ('ffmpeg.exe', 'ffprobe.exe'):
        installed = shutil.which(filename.removesuffix('.exe'))
        source = Path(installed) if installed else None
        if source is not None and source.is_file() and not (repo/filename).exists():
            shutil.copy2(source, repo/filename)
        if not (repo/filename).is_file():
            raise FileNotFoundError(f'Install FFmpeg or place {filename} in {repo}.')
    if not args.skip_models:
        if Path(sys.executable).resolve() != backend.resolve():
            run([backend, __file__, '--models-only', '--config', args.config,
                 *(['--chinese-frontend'] if args.chinese_frontend else [])])
        elif args.chinese_frontend:
            download_chinese_frontend(repo, work)
        else:
            download_models(repo, work, spec)
    print('Setup complete; run scripts/tts_finetune.py doctor.', flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    main()
