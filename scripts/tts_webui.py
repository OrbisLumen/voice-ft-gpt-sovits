"""Launch the bundled inference UI with explicitly selected local fine-tuned models."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import webbrowser

from tts_finetune import ROOT, Task
from character_manager import CharacterStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/task.json')
    parser.add_argument('--character')
    parser.add_argument('--task', help='Registered training task ID')
    parser.add_argument('--sovits-epoch', type=int)
    parser.add_argument('--gpt-epoch', type=int)
    parser.add_argument('--mode', choices=['main', 'inference'], default='inference')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true', help='Check paths without starting the UI')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    args.port = args.port or (9874 if args.mode == 'main' else 9872)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    store=CharacterStore()
    character_id=args.character or store.read()['default_character']
    _,task_id,entry,_=store.task(character_id,args.task)
    if args.character or args.task: args.config=entry['config']
    task = Task(args.config)
    if task.config_path.resolve()!=store.local(entry['config']):
        raise ValueError('任务配置未匹配角色登记，请使用--character和--task。')
    default=store.defaults(character_id,task_id)
    sovits=Path(default['sovits']) if default['sovits'] else None
    gpt=Path(default['gpt']) if default['gpt'] else None
    if args.sovits_epoch is not None:
        options=list((task.weights/'sovits').glob(f'{task.cfg["name"]}_e{args.sovits_epoch}_s*.pth'))
        if len(options)!=1: raise ValueError('Select one existing fine-tuned SoVITS checkpoint.')
        sovits=options[0]
    if args.gpt_epoch is not None:
        gpt=task.weights/'gpt'/f'{task.cfg["name"]}-e{args.gpt_epoch}.ckpt'
    if not gpt or not sovits:
        raise ValueError('此角色任务尚无配套推理模型。先启动已有模型的界面，再登记、训练或导入新角色。')
    store.validate_pair(character_id,task_id,str(gpt),str(sovits))
    script = task.repo/('webui.py' if args.mode == 'main' else 'GPT_SoVITS/inference_webui_fast.py')
    bert = task.pretrained/'chinese-roberta-wwm-ext-large'
    hubert = task.pretrained/'chinese-hubert-base'
    for path in (task.backend, script, gpt, bert/'config.json', hubert/'config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    print('SoVITS:', sovits, '\nGPT:', gpt, '\nUI:', f'http://127.0.0.1:{args.port}', flush=True)
    if args.check:
        return 0
    with socket.socket() as listener:
        listener.settimeout(1)
        if listener.connect_ex(('127.0.0.1',args.port)) == 0:
            print(f'Port already in use. Open http://127.0.0.1:{args.port} or choose --port.',flush=True)
            if not args.no_browser:
                webbrowser.open(f'http://127.0.0.1:{args.port}')
            return 0
    reference=entry.get('reference') or {}
    (task.work/f'webui_{args.mode}.pid').write_text(str(os.getpid()),encoding='ascii')
    environment = task.environment(gpt_path=gpt, sovits_path=sovits, bert_path=bert,
        cnhubert_base_path=hubert, infer_ttswebui=9872 if args.mode == 'main' else args.port,
        is_share='False', is_half='True', language='zh_CN',
        local_sovits_weight_root=task.weights/'sovits', local_gpt_weight_root=task.weights/'gpt',
        local_webui_script=script, local_webui_port=args.port, local_webui_browser=str(not args.no_browser),
        local_ref_audio=ROOT/reference['audio_path'] if reference.get('audio_path') else '',
        local_ref_text=reference.get('text',''),local_seed=task.cfg['seed'],
        local_character=character_id,local_task=task_id,local_scripts_root=ROOT/'scripts')
    bootstrap = '''import os, runpy, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.environ['local_scripts_root'])
import config
from character_manager import CharacterStore
store=CharacterStore()
for character_id, character in store.read()['characters'].items():
    for task_id in character['tasks']:
        cfg=store.task(character_id,task_id)[3]
        config.SoVITS_weight_root.append(str(store.local(cfg['weights'])/'sovits'))
        config.GPT_weight_root.append(str(store.local(cfg['weights'])/'gpt'))
config.webui_port_main = int(os.environ['local_webui_port'])
import gradio as gr
original_launch = gr.Blocks.launch
def local_launch(self, *args, **kwargs):
    kwargs['server_name'] = '127.0.0.1'
    kwargs['inbrowser'] = os.environ['local_webui_browser'] == 'True'
    return original_launch(self, *args, **kwargs)
gr.Blocks.launch = local_launch
sys.argv = [os.environ['local_webui_script'], 'zh_CN']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    return subprocess.run([str(task.backend), '-c', bootstrap], cwd=task.repo,
                          env=environment).returncode


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
