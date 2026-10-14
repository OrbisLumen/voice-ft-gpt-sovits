"""Extract selected local manuscript chapters as plain UTF-8 narration inputs."""
import argparse
import hashlib
import sys

from tts_finetune import Task
import tts_utils as utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapters', nargs='+', type=int, default=[1,2])
    args = parser.parse_args()
    cfg = utils.read_json(utils.ROOT/'config/narration.json')
    task = Task('config/task.json')
    destination = task.work/'narration_inputs'
    destination.mkdir(exist_ok=True)
    rows = []
    for row in utils.chapters(cfg):
        if row['chapter'] not in args.chapters:
            continue
        name = 'intro' if row['chapter'] == 1 else f'chapter{row["chapter"]}'
        path = destination/f'{row["chapter"]:02}_{name}.txt'
        if row['chapter'] == 2 and row['title'] == 'PicaPica':
            path = destination/'02_PicaPica.txt'
        path.write_text(row['text']+'\n',encoding='utf-8')
        rows.append({**row,'path':str(path.relative_to(utils.ROOT)),'characters':len(row['text'])})
        print('Prepared:',path)
    if set(args.chapters) != {row['chapter'] for row in rows}:
        raise ValueError('Selected chapter not found.')
    sources=[utils.ROOT/cfg['source'], *[utils.ROOT/v for v in cfg['chapter_overrides'].values()]]
    utils.write_json(destination/'manifest.json',{'sources':[{'path':str(p.relative_to(utils.ROOT)),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'results':rows})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
