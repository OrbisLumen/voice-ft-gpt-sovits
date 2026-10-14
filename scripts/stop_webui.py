"""Stop only UI process trees launched by this workspace."""
from pathlib import Path
import subprocess
import sys

import psutil
from tts_finetune import Task


def main():
    task=Task('config/task.json')
    for path in task.work.glob('webui_*.pid'):
        pid=int(path.read_text(encoding='utf-8-sig').strip())
        try:
            process=psutil.Process(pid)
            command=' '.join(process.cmdline())
            if Path(process.cwd()).resolve() != task.work.parents[1] or 'tts_webui.py' not in command:
                print('Skip unrelated process:',pid)
                continue
            subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],check=True)
        except psutil.NoSuchProcess:
            print('Already stopped:',pid)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
