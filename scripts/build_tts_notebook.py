"""Build the local fine-tuning notebook without starting any training."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def build(character_id='misuzu', display_name='心铃', config='config/task.json'):
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': (textwrap.dedent(source).strip()+'\n').splitlines(keepends=True)}
        if kind == 'code':
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    add('markdown', '''
    # 心铃本地 TTS：GPT-SoVITS v2Pro 微调

    中文台词 → 本地角色TTS直接生成音声。
    当前采用18段心铃日语；原始音声库全部保留，可继续分割和标注。
    选择Python (voice)内核；计算由独立GPT-SoVITS环境执行。

    本机已完成64条台词校对、特征提取、SoVITS20轮/GPT10轮和长文推理。
    当前总轮数上限20/10，目标权重已存在时复用，不会再增加20/10轮。
    插电、保持Jupyter服务与内核运行并关闭自动睡眠，然后 **Run → Run All Cells**。
    原始数据准备与转写默认跳过，不要同时启动其他GPU训练。
    先试听并校正review.csv里的每条日语台词，将reviewed设为true；脚本会阻止未经校对的标签进入正式训练。
    转写完成不等于校对完成。后续用不同心铃参考片段检验中文发音、音色与语气。

    详细说明：[训练与数据](../docs/training.md)。
    ''')
    add('code', r'''
    from pathlib import Path
    import csv, json, os, subprocess, sys
    from IPython.display import Audio, display, Markdown

    RUN_PREPARE = False  # 本机数据已准备并校对；重建任务时才开启
    RUN_ASR = False      # 使用已校对review.csv
    RUN_FEATURES = False  # 校对review.csv后开启
    RUN_TRAIN = False     # 人工校对并完成特征提取后，手动开启
    RUN_SAMPLE = False    # 训练完成后，手动开启
    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next(p.resolve() for p in candidates if (p/'scripts/tts_finetune.py').is_file())
    CONFIG = ROOT/'config/task.json'
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    BACKEND = ROOT/cfg['repository']/'.venv/Scripts/python.exe'
    WORK = ROOT/cfg['work']
    assert BACKEND.is_file(), '先安装独立TTS环境：python scripts/setup_tts.py'
    ABORTED = False

    def stage(name):
        global ABORTED
        if ABORTED:
            raise RuntimeError('前一阶段失败，先修复问题再重跑参数单元格。')
        command = [str(BACKEND), '-u', str(ROOT/'scripts/tts_finetune.py'), name, '--config', str(CONFIG)]
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line.rstrip(), flush=True)
            code = process.wait()
        except BaseException:
            ABORTED = True
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], capture_output=True)
            process.wait()
            raise
        if code:
            ABORTED = True
            raise RuntimeError(f'{name}失败，详见 {WORK}/logs 或单元格输出。')

    print('任务:', cfg['name'], '版本:', cfg['version'])
    print('正式训练:', RUN_TRAIN, '参数:', cfg['training'])
    ''')
    add('markdown', '## 1. 环境、原素材与转写草稿\n网络下载只在setup阶段进行；这里使用本地模型。')
    add('code', '''
    stage('doctor')
    if RUN_PREPARE:
        stage('prepare')
    if RUN_ASR:
        stage('transcribe')
    stage('status')
    ''')
    add('markdown', '''
    ## 2. 日语台词校对
    打开下方review.csv，逐条听音频并修改text，把确认正确的reviewed改为true。
    不改name和language；language应是ja。保留CSV列名和UTF-8编码。
    背景音乐、多人重叠、长片段需先重新筛选；不要为通过检查直接全选true。
    ''')
    add('code', '''
    review = ROOT/cfg['dataset']/'review.csv'
    print('校对文件:', review)
    if review.exists():
        display(Markdown(f'[打开校对CSV](../{review.relative_to(ROOT).as_posix()})'))
        with review.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        print('共', len(rows), '段；已确认', sum(r['reviewed'].lower() in ('true','1','yes') for r in rows))
        for row in rows[:10]:
            print(row['name'], row['text'])
            display(Audio(filename=str(ROOT/row['audio_path'])))
    else:
        print('先运行prepare与transcribe。')
    ''')
    add('markdown', '## 3. 提取训练特征\n确认转写后才运行。两阶段模型共用同一份校对标签。')
    add('code', '''
    if RUN_FEATURES:
        stage('audit')
        stage('features')
    else:
        print('未提取特征；校对完成后将RUN_FEATURES设为True。')
    ''')
    add('markdown', '''
    ## 4. 本地微调
    batch=2；SoVITS20轮、GPT10轮；半精度、SoVITS梯度检查点、每轮保存。
    两阶段顺序执行，避免8GB显存同时加载多个模型。
    配套推理权重存放在models/下：SoVITS pth和GPT ckpt。
    本机已通过两阶段各1轮实际训练和保存；仍需检查后续质量，不并行重跑。
    ''')
    add('code', '''
    if RUN_TRAIN:
        stage('train-sovits')
        stage('train-gpt')
    else:
        print('正式训练关闭。先完成标签校对与特征提取。')
    stage('status')
    ''')
    add('markdown', '## 5. 合成试听\n使用配置指定的SoVITS/GPT轮数，生成本地中文与日语短句，不调用远端TTS。')
    add('code', '''
    if RUN_SAMPLE:
        stage('sample')
        folder = ROOT/cfg['output']/f"samples_s{cfg['training']['sovits_epochs']}_g{cfg['training']['gpt_epochs']}"
        for path in sorted(folder.glob('*.wav')):
            print(path.name)
            display(Audio(filename=str(path)))
    else:
        print('RUN_SAMPLE=False；训练完成后开启生成试听。')
    ''')
    if character_id != 'misuzu':
        cells[0]['source'] = [f'# {display_name}本地 TTS：GPT-SoVITS v2Pro 微调\n',
            '\n数据、标注接受情况及训练进度以当前任务记录为准；已有目标权重会复用。\n',
            '标注检查入口为 data/datasets/<任务名>/review.html；原始转写及训练接受记录保存在任务 runs 目录。\n',
            '准备、转写、特征、训练、试听开关默认全部关闭；Run All 不会自动训练。\n']
        for cell in cells:
            cell['source'] = [line.replace("CONFIG = ROOT/'config/task.json'", f"CONFIG = ROOT/'{config}'")
                .replace('本机数据已准备并校对；重建任务时才开启', '本机数据已准备；重建任务时才开启')
                .replace('使用已校对review.csv', '使用当前review.csv，重新转写前先备份')
                .replace('本机已通过两阶段各1轮实际训练和保存；仍需检查后续质量，不并行重跑。',
                         '首次训练须先完成标注检查，再顺序提取特征和训练；已有目标权重会复用。') for line in cell['source']]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python (voice)',
        'language': 'python', 'name': 'voice'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
    (ROOT/f'notebooks/finetune_{character_id}_tts.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        registry = json.loads((ROOT/'config/characters.json').read_text(encoding='utf-8'))
        for character_id, character in registry['characters'].items():
            entry = character['tasks'][character['active_task']]
            build(character_id, character['display_name'], entry['config'])
    else:
        build()
