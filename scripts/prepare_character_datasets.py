"""Prepare first-pass character datasets and ASR review pages; never train models.

Uses 18 automatically screened sources and 5 source-level holdouts per new voice.
All ASR labels remain unreviewed. Re-running resumes preparation/transcription.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import soundfile as sf

from character_manager import CharacterStore, sha256
from tts_finetune import Task, data_signature
import tts_utils as utils

ROOT = utils.ROOT
ROLES = [('rin', '禀'), ('kei', '圭'), ('ai', '蓝'), ('rina', '里奈'),
         ('shizuku', '雫'), ('yuumi', '优美'), ('makoto', '真琴')]


def migrate():
    """Relocate live labels without changing bytes; repair renamed source paths by hash."""
    store = CharacterStore()
    registry = store.read()
    registry['characters']['misuzu']['raw_directory'] = 'data/source_library/心铃'
    registry['characters']['shizuku']['raw_directory'] = 'data/source_library/雫'
    misuzu = Task('config/task.json')
    index_path = ROOT/misuzu.cfg['source_manifest']
    index = utils.read_json(index_path)
    lookup = {}
    for path in (ROOT/'data/source_library/心铃').rglob('*.wav'):
        lookup.setdefault(path.name, []).append(path)
    fixed = {}
    for row in index['files']:
        name = row['destination'].replace('\\', '/').split('/')[-1]
        matches = [p for p in lookup.get(name, []) if sha256(p) == row['sha256']]
        if not matches:
            raise ValueError(f'Cannot resolve original audio by hash: {name}')
        path = min(matches, key=lambda p: len(str(p))).relative_to(ROOT).as_posix()
        fixed[row['sha256']] = path
        row['source'] = row['destination'] = path
    utils.write_json(index_path, index)
    manifest_path = misuzu.work/'dataset_manifest.json'
    manifest = utils.read_json(manifest_path)
    for row in manifest['sources']:
        row['path'] = row['original'] = fixed[row['sha256']]
    for row in manifest['clips']:
        name = row['source'].replace('\\', '/').split('/')[-1]
        row['source'] = next(r['destination'] for r in index['files']
                             if Path(r['destination']).name == name)
    manifest['holdout_sources'] = [r for r in index['files'] if r['split'] == 'holdout']
    manifest['data_sha256'] = data_signature(misuzu.cfg)
    utils.write_json(manifest_path, manifest)
    for filename in ('review.csv', 'train.list'):
        old, new = misuzu.work/filename, misuzu.dataset/filename
        if old.exists():
            if new.exists() and old.read_bytes() != new.read_bytes():
                raise ValueError(f'Conflicting label files: {old}, {new}')
            new.parent.mkdir(parents=True, exist_ok=True)
            if not new.exists():
                shutil.copy2(old, new)
            if sha256(old) != sha256(new):
                raise ValueError('Label relocation verification failed')
            old.unlink()
    # Rename only the empty, untrained placeholder for Shizuku.
    char = registry['characters']['shizuku']
    old_id, new_id = 'shizuku_v2pro_v1', 'shizuku_gsv_v2pro_v1'
    if old_id in char['tasks']:
        entry = char['tasks'][old_id]
        cfg = utils.read_json(ROOT/entry['config'])
        if utils.read_json(ROOT/cfg['source_manifest'])['files']:
            raise ValueError('Shizuku placeholder has sources; inspect before renaming')
        for key in ('dataset', 'work', 'weights', 'output'):
            if any(p.is_file() for p in (ROOT/cfg[key]).rglob('*')):
                raise ValueError('Shizuku placeholder contains files; inspect before renaming')
        for key in ('source_manifest', 'dataset', 'work', 'weights', 'output'):
            old = (ROOT/cfg[key]).resolve()
            new = Path(str(old).replace(old_id, new_id)).resolve()
            if not old.is_relative_to(ROOT) or not new.is_relative_to(ROOT):
                raise ValueError('Rename outside workspace')
            if new.exists():
                raise ValueError(f'Target already exists: {new}')
            if old.exists():
                old.rename(new)
            cfg[key] = new.relative_to(ROOT).as_posix()
        cfg['name'] = new_id
        utils.write_json(ROOT/entry['config'], cfg)
        char['tasks'][new_id] = char['tasks'].pop(old_id)
        char['active_task'] = new_id
    store.save(registry)


def select_sources(directory, role):
    candidates, identities = [], set()
    # User confirmed szd and kzd are equivalent Shizuku banks; screen them together.
    for path in sorted(directory.rglob('*.wav')):
        info = sf.info(path)
        if not 8 <= info.duration <= 20:
            continue
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            continue
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        clipped = float(np.mean(np.abs(audio) >= .999))
        voiced = float(np.mean(np.abs(audio) > .01))
        if peak < .03 or rms < .008 or clipped > .005 or voiced < .1:
            continue
        identity = sha256(path)
        if identity in identities:
            continue
        identities.add(identity)
        candidates.append({'source': path.relative_to(ROOT).as_posix(),
            'destination': path.relative_to(ROOT).as_posix(), 'sha256': identity,
            'seconds': info.duration, 'rms': rms, 'clipped_fraction': clipped,
            'signal_fraction': voiced, 'rank': abs(info.duration-14) + clipped*100})
    if len(candidates) < 23:
        raise ValueError(f'Need 23 suitable unique source files: {role}, found {len(candidates)}')
    # Prefer continuous ordinary dialogue lengths, spread across filename ranges.
    by_name = sorted(candidates, key=lambda r: r['source'])
    groups = np.array_split(np.arange(len(by_name)), 23)
    chosen = [min((by_name[int(i)] for i in group), key=lambda r: r['rank']) for group in groups]
    holdout_positions = {2, 7, 12, 17, 22}
    for i, row in enumerate(chosen):
        row.pop('rank')
        row.update(split='holdout' if i in holdout_positions else 'train',
                   user_selected=False, selection_method='automatic_acoustic_screen_pending_listening')
        # selected_only selects these sources, but selection is explicitly automatic.
        row['user_selected'] = row['split'] == 'train'
    return {'files': chosen, 'selection': '18 automatically screened train sources + 5 source-level holdouts; human listening pending',
            'candidate_count': len(candidates), 'all_original_audio_retained': True,
            'equivalent_banks': ['szd', 'kzd'] if role == 'shizuku' else None}


def review_page(task, display_name):
    with (task.dataset/'review.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    manifest = utils.read_json(task.work/'dataset_manifest.json')
    acceptance_path = task.work/'training_acceptance.json'
    acceptance = utils.read_json(acceptance_path) if acceptance_path.exists() else None
    accepted_label = '已接受训练' if acceptance else '已确认'
    clips = {r['name']: r for r in manifest['clips']}
    items, flags = [], []
    for row in rows:
        draft_path = task.work/'asr_drafts'/(row['name']+'.json')
        draft = utils.read_json(draft_path) if draft_path.exists() else {}
        warnings = []
        if not row['text'].strip(): warnings.append('空转写，需听写或排除')
        if draft.get('retried_without_vad'): warnings.append('关闭VAD后重试的识别结果，需重点试听')
        if draft.get('avg_logprob') is not None and draft['avg_logprob'] < -.7: warnings.append('识别置信度偏低')
        if len(row['text'])/max(clips[row['name']]['seconds'], 1) > 12: warnings.append('文字密度偏高')
        flags.append({'name': row['name'], 'warnings': warnings, 'avg_logprob': draft.get('avg_logprob')})
        name = html.escape(row['name'])
        items.append(f'<tr><td>{name}<br>{clips[row["name"]]["seconds"]:.2f}s</td>'
                     f'<td><audio controls preload="none" src="wavs/{name}"></audio></td>'
                     f'<td lang="ja">{html.escape(row["text"])}</td><td>{html.escape("；".join(warnings))}</td>'
                     f'<td>{accepted_label if row["reviewed"].lower()=="true" else "待校对"}</td></tr>')
    page = '<!doctype html><meta charset="utf-8"><title>'+html.escape(display_name)+' 标注检查</title>'
    page += '<style>body{font:16px system-ui;margin:32px;color:#182230}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dc;padding:12px;text-align:left}td[lang]{min-width:320px}audio{width:260px}tr:nth-child(even){background:#f4f7fa}</style>'
    page += f'<h1>{html.escape(display_name)} · {task.cfg["name"]}</h1><p>{len(rows)} 条，合计 {manifest["eligible_seconds"]:.2f} 秒。此页为只读快照。</p>'
    page += '<p>请在同目录 <a href="review.csv">review.csv</a> 修改 text，逐条试听确认后把 reviewed 设为 true；name、language、audio_path 保持不变。自动转写尚未人工确认，声线不符、多人重叠、音乐和非语言声音需特别检查。未完成校对时不能提取训练特征或训练。</p>'
    if acceptance:
        page += '<p>本任务已由用户授权采用现有转写训练；“已接受训练”不表示逐句人工听写校对。空转写已排除，原标注及音频保留。训练授权和备份位置见任务 runs 目录的 training_acceptance.json。</p>'
    page += '<table><thead><tr><th>音频</th><th>试听</th><th>标注文本</th><th>检查提示</th><th>状态</th></tr></thead><tbody>'+''.join(items)+'</tbody></table>'
    (task.dataset/'review.html').write_text(page, encoding='utf-8')
    utils.write_json(task.dataset/'review_flags.json', flags)
    (task.dataset/'README.md').write_text(f'# {display_name} · {task.cfg["name"]}\n\n'
        'review.csv 是训练读取的唯一校对文件；review.html 是带音频的只读快照。\n'
        '仅逐条确认后设置 reviewed=true。ASR 是待核对草稿，自动声学筛选不代表人工听音质量认证。\n'
        'wavs/ 为32kHz单声道PCM16切片；train.list 会在全部校对通过后由 audit 生成。\n'
        + ('本任务已由用户授权采用现有转写训练，reviewed=true 表示已接受训练，不代表逐句人工听写校对；授权及备份见任务 runs 目录的 training_acceptance.json。\n' if acceptance else '') +
        f'源级划分及切片范围见 ../../manifests/{Path(task.cfg["source_manifest"]).name} 和 ../../../{task.cfg["work"]}/dataset_manifest.json。\n', encoding='utf-8')
    return {'character': display_name, 'task': task.cfg['name'], 'clips': len(rows),
            'seconds': round(manifest['eligible_seconds'], 2),
            'reviewed': sum(r['reviewed'].lower() == 'true' for r in rows),
            'empty_texts': sum(not r['text'].strip() for r in rows),
            'flags': sum(bool(r['warnings']) for r in flags),
            'review': (task.dataset/'review.csv').relative_to(ROOT).as_posix()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcribe', action='store_true', help='Run local ASR only, never training')
    parser.add_argument('--pages-only', action='store_true', help='Refresh read-only review pages from current CSVs')
    args = parser.parse_args()
    if not args.pages_only:
        migrate()
    store = CharacterStore()
    summaries = [review_page(Task('config/task.json'), '心铃')]
    for role, display in ROLES:
        if role not in store.read()['characters']:
            store.add(role, display, raw_directory=f'data/source_library/{display}')
        _, _, entry, cfg = store.task(role)
        task = Task(entry['config'])
        if not args.pages_only and not utils.read_json(ROOT/cfg['source_manifest'])['files']:
            utils.write_json(ROOT/cfg['source_manifest'], select_sources(ROOT/f'data/source_library/{display}/prepared', role))
        if not args.pages_only:
            with task.exclusive():
                task.prepare()
                task.training_configs()
                if args.transcribe:
                    task.transcribe()
        if (task.dataset/'review.csv').exists():
            summaries.append(review_page(task, display))
        print('Prepared (training not started):', task.cfg['name'], flush=True)
    utils.write_json(ROOT/'data/datasets/preparation_summary.json', summaries)
    items = ''.join(f'<tr><td>{html.escape(r["character"])}</td><td>{r["task"]}</td>'
        f'<td>{r["clips"]}</td><td>{r["seconds"]:.2f}s</td><td>{r["reviewed"]}</td>'
        f'<td>{r["flags"]}</td><td><a href="{r["task"]}/review.html">试听与文本</a> · '
        f'<a href="{r["task"]}/review.csv">校对CSV</a></td></tr>' for r in summaries)
    (ROOT/'data/datasets/index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>角色数据集标注检查</title><style>body{font:16px system-ui;margin:36px;color:#182230}'
        'table{border-collapse:collapse}td,th{padding:14px;border:1px solid #ccd4dc}</style>'
        '<h1>角色数据集标注检查</h1><p>此页列出各任务的数据和标注状态；模型训练状态及可用权重请查看角色管理界面。</p>'
        '<p>在各目录 review.csv 修改文本并逐条确认 reviewed=true；HTML 是只读快照。'
        '修改 CSV 后可运行 scripts/prepare_character_datasets.py --pages-only 刷新页面。</p>'
        '<table><tr><th>角色</th><th>任务</th><th>切片</th><th>时长</th><th>已确认/接受训练</th><th>需重点检查</th><th>检查入口</th></tr>'
        +items+'</table>', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
