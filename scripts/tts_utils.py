"""Local file, chapter and audio utilities for GPT-SoVITS workflows."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import wave

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audio_info(path):
    with wave.open(str(path)) as wav:
        if not wav.getnframes():
            raise ValueError(f'Empty audio: {path}')
        return {'seconds': round(wav.getnframes()/wav.getframerate(), 3),
                'sample_rate': wav.getframerate(), 'channels': wav.getnchannels()}


def chapters(config):
    source = re.sub(r'<!--.*?-->', '', (ROOT/config['source']).read_text(encoding='utf-8'), flags=re.S)
    parts = re.split(r'^##\s+(.+?)\s*$', source, flags=re.M)
    items = []
    for number, i in enumerate(range(1, len(parts), 2), 1):
        override = config.get('chapter_overrides', {}).get(str(number))
        text = (ROOT/override).read_text(encoding='utf-8') if override else parts[i+1]
        text = re.sub(r'\s*\n\s*', '', text.strip())
        if not text:
            raise ValueError(f'Empty chapter {number}')
        items.append({'chapter': number, 'title': parts[i].strip(), 'text': text})
    if not items:
        raise ValueError('No ## chapters found.')
    return items


def ffmpeg_path(config):
    installed = shutil.which('ffmpeg')
    if installed:
        return installed
    bundled = ROOT/config['repository']/('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    return str(bundled) if bundled.is_file() else None


def finalize(config, src, dst):
    src, dst = Path(src), Path(dst)
    ffmpeg = ffmpeg_path(config)
    if not ffmpeg:
        raise ValueError('FFmpeg missing; place it in the GPT-SoVITS runtime or PATH.')
    if not src.is_file():
        raise FileNotFoundError(src)
    signature = fingerprint({'input': hashlib.sha256(src.read_bytes()).hexdigest(), 'final': config['final']})
    metadata = dst.with_suffix('.json')
    if dst.exists() and metadata.exists() and read_json(metadata).get('signature') == signature:
        audio_info(dst)
        return
    if dst.exists():
        raise ValueError(f'Output exists with different input/settings: {dst}')
    settings = config['final']
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix('.tmp.wav')
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src),
        '-af', f'loudnorm=I={settings["loudness_lufs"]}:TP={settings["true_peak_db"]}:LRA=11',
        '-ar', str(settings['sample_rate']), '-ac', '1', '-c:a', 'pcm_s16le', str(temporary)], check=True)
    info = audio_info(temporary)
    temporary.replace(dst)
    write_json(metadata, {'signature': signature, 'audio': info, 'input': str(src)})
    print('Final audio:', dst)
