"""Launch the bundled inference UI with explicitly selected local fine-tuned models."""
import argparse
import csv
import os
import socket
import subprocess
import sys
import webbrowser

from tts_finetune import ROOT, Task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/task.json')
    parser.add_argument('--sovits-epoch', type=int, default=20)
    parser.add_argument('--gpt-epoch', type=int, default=10)
    parser.add_argument('--mode', choices=['main', 'inference'], default='inference')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true', help='Check paths without starting the UI')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    args.port = args.port or (9874 if args.mode == 'main' else 9872)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    task = Task(args.config)
    sovits = list((task.weights/'sovits').glob(f'{task.cfg["name"]}_e{args.sovits_epoch}_s*.pth'))
    if len(sovits) != 1:
        raise ValueError('Select one existing fine-tuned SoVITS checkpoint.')
    gpt = task.weights/'gpt'/f'{task.cfg["name"]}-e{args.gpt_epoch}.ckpt'
    script = task.repo/('webui.py' if args.mode == 'main' else 'GPT_SoVITS/inference_webui_fast.py')
    bert = task.pretrained/'chinese-roberta-wwm-ext-large'
    hubert = task.pretrained/'chinese-hubert-base'
    for path in (task.backend, script, gpt, bert/'config.json', hubert/'config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    print('SoVITS:', sovits[0], '\nGPT:', gpt, '\nUI:', f'http://127.0.0.1:{args.port}', flush=True)
    if args.check:
        return 0
    with socket.socket() as listener:
        listener.settimeout(1)
        if listener.connect_ex(('127.0.0.1',args.port)) == 0:
            print(f'Port already in use. Open http://127.0.0.1:{args.port} or choose --port.',flush=True)
            if not args.no_browser:
                webbrowser.open(f'http://127.0.0.1:{args.port}')
            return 0
    with (task.work/'review.csv').open(encoding='utf-8-sig',newline='') as stream:
        reference=next(row for row in csv.DictReader(stream) if row['name']=='fem_mis_00025_000.wav')
    (task.work/f'webui_{args.mode}.pid').write_text(str(os.getpid()),encoding='ascii')
    environment = task.environment(gpt_path=gpt, sovits_path=sovits[0], bert_path=bert,
        cnhubert_base_path=hubert, infer_ttswebui=9872 if args.mode == 'main' else args.port,
        is_share='False', is_half='True', language='zh_CN',
        local_sovits_weight_root=task.weights/'sovits', local_gpt_weight_root=task.weights/'gpt',
        local_webui_script=script, local_webui_port=args.port, local_webui_browser=str(not args.no_browser),
        local_ref_audio=ROOT/reference['audio_path'], local_ref_text=reference['text'],
        local_seed=task.cfg['seed'])
    bootstrap = '''import os, runpy, sys
sys.path.insert(0, os.getcwd())
import config
config.SoVITS_weight_root.append(os.environ['local_sovits_weight_root'])
config.GPT_weight_root.append(os.environ['local_gpt_weight_root'])
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
