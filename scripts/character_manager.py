"""Character/task registry, isolated paths and local asset registration."""
import argparse
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import sys
import threading

import tts_utils as utils

LANGUAGES = {'ja': '日文', 'zh': '中文', 'en': '英文', 'ko': '韩文', 'yue': '粤语'}


class CheckpointConfig:
    """Data-only replacement for upstream's serialized utils.HParams."""
    pass


def plain_config(value):
    if isinstance(value,CheckpointConfig): value=vars(value)
    if isinstance(value,dict): return {k:plain_config(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain_config(v) for v in value]
    return value


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


class CharacterStore:
    def __init__(self, root=utils.ROOT):
        self.root=Path(root).resolve()
        self.path=self.root/'config/characters.json'
        self.lock=threading.RLock()

    def read(self):
        return utils.read_json(self.path)

    def save(self, registry):
        temporary=self.path.with_suffix('.tmp')
        utils.write_json(temporary,registry)
        temporary.replace(self.path)

    def local(self, value):
        path=(self.root/value).resolve()
        if not path.is_relative_to(self.root): raise ValueError('路径必须在项目内。')
        return path

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def task(self, character_id, task_id=None):
        registry=self.read()
        if character_id not in registry['characters']: raise ValueError('角色不存在，请刷新角色列表。')
        character=registry['characters'][character_id]
        task_id=task_id or character['active_task']
        if task_id not in character['tasks']: raise ValueError('训练任务不属于此角色。')
        entry=character['tasks'][task_id]
        config=utils.read_json(self.local(entry['config']))
        return character,task_id,entry,config

    def characters(self):
        return [(f'{v["display_name"]} · {k}',k) for k,v in self.read()['characters'].items()]

    def tasks(self, character_id):
        character,_,_,_=self.task(character_id)
        return list(character['tasks'])

    def models(self, character_id, task_id):
        _,_,_,config=self.task(character_id,task_id)
        result={}
        for stage,extension in [('gpt','.ckpt'),('sovits','.pth')]:
            directory=self.local(config['weights'])/stage
            rows=[]
            for path in directory.glob('*'+extension):
                if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()): continue
                match=re.search(r'(?:^|[_-])e(\d+)(?:[_\-.]|$)',path.name)
                epoch=int(match.group(1)) if match else -1
                label=f'{epoch}e · {path.name}' if epoch>=0 else f'导入 · {path.name}'
                rows.append((epoch,label,str(path.resolve())))
            result[stage]=[(label,path) for _,label,path in sorted(rows,reverse=True)]
        return result

    def defaults(self, character_id, task_id):
        _,_,_,config=self.task(character_id,task_id)
        models=self.models(character_id,task_id)
        result={}
        for stage in ('gpt','sovits'):
            epoch=config['training'][stage+'_epochs']
            result[stage]=next((path for label,path in models[stage] if label.startswith(f'{epoch}e · ')),
                               models[stage][0][1] if models[stage] else None)
        selection_path=self.local(config['work'])/'selected_models.json'
        if selection_path.exists():
            selection=utils.read_json(selection_path)
            for stage in ('gpt','sovits'):
                if selection.get(stage):
                    value=str(self.local(selection[stage]))
                    if value in {v for _,v in models[stage]}: result[stage]=value
        return result

    def save_selection(self,character_id,task_id,gpt,sovits):
        self.validate_pair(character_id,task_id,gpt,sovits)
        _,_,_,config=self.task(character_id,task_id)
        utils.write_json(self.local(config['work'])/'selected_models.json',
                         {'gpt':self.relative(gpt),'sovits':self.relative(sovits)})

    def validate_pair(self, character_id, task_id, gpt, sovits):
        choices=self.models(character_id,task_id)
        for stage,path in [('gpt',gpt),('sovits',sovits)]:
            if not path or str(Path(path).resolve()) not in {v for _,v in choices[stage]}:
                raise ValueError(f'{stage}权重不属于所选角色和训练任务，或尚未训练。')
        _,_,_,config=self.task(character_id,task_id)
        if config['version']!='v2Pro': raise ValueError('当前角色管理使用v2Pro架构。')
        return str(Path(gpt).resolve()),str(Path(sovits).resolve())

    def _identifier(self,value):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,47}',value):
            raise ValueError('ID使用小写英文字母开头，后接字母、数字、下划线或短横线，最长48字符。')
        if value in {'con','prn','aux','nul',*[f'com{i}' for i in range(1,10)],*[f'lpt{i}' for i in range(1,10)]}:
            raise ValueError('ID不能是Windows保留设备名。')

    def _new_task(self, registry, character_id, suffix, language):
        self._identifier(suffix)
        name=f'{character_id}_gsv_v2pro_{suffix}'
        character=registry['characters'][character_id]
        if name in character['tasks']: raise ValueError('任务已存在，不会覆盖。')
        config_path=f'config/characters/{character_id}/{suffix}.json'
        template=self.local(registry.get('task_template','config/task_template.json'))
        if not template.exists(): template=self.root/'config/task.json'
        config=copy.deepcopy(utils.read_json(template))
        config.update(name=name,character_id=character_id,language=language,
            source_manifest=f'data/manifests/{name}.json',dataset=f'data/datasets/{name}',
            work=f'runs/{name}',output=f'outputs/{name}',weights=f'models/{name}')
        if self.local(config_path).exists(): raise ValueError('任务配置文件已存在。')
        for key in ('source_manifest','dataset','work','output','weights'):
            if self.local(config[key]).exists(): raise ValueError(f'已有目录或文件，无法创建隔离任务：{config[key]}')
        utils.write_json(self.local(config_path),config)
        utils.write_json(self.local(config['source_manifest']),{'files':[]})
        for key in ('dataset','work','output','weights'):
            self.local(config[key]).mkdir(parents=True)
        for stage in ('gpt','sovits'): (self.local(config['weights'])/stage).mkdir()
        character['tasks'][name]={'config':config_path,'reference':None}
        character['active_task']=name
        return name

    def add(self, character_id, display_name, language='ja', raw_directory=None):
        self._identifier(character_id)
        if not display_name.strip(): raise ValueError('请输入角色名称。')
        if language not in LANGUAGES: raise ValueError('不支持的训练语种。')
        with self.lock:
            registry=self.read()
            if character_id in registry['characters']: raise ValueError('角色ID已存在，不会覆盖。')
            raw=raw_directory or f'data/source_library/{character_id}'
            if not self.local(raw).is_relative_to(self.root/'data/source_library'):
                raise ValueError('原素材目录必须在data/source_library内。')
            registry['characters'][character_id]={'display_name':display_name.strip(),'language':language,
                'raw_directory':raw,'active_task':'','tasks':{}}
            name=self._new_task(registry,character_id,'v1',language)
            self.local(raw).mkdir(parents=True,exist_ok=True)
            self.save(registry)
            return name

    def new_task(self, character_id, suffix):
        with self.lock:
            registry=self.read()
            character=registry['characters'][character_id]
            name=self._new_task(registry,character_id,suffix,character['language'])
            self.save(registry)
            return name

    def set_reference(self, character_id, task_id, audio, text, language):
        self.task(character_id,task_id)
        if not text.strip() or language not in LANGUAGES: raise ValueError('填写准确参考台词和语种。')
        import numpy as np
        import soundfile as sf
        audio=Path(audio)
        wave,rate=sf.read(audio,dtype='float32')
        if not 3<=len(wave)/rate<=10 or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError('参考音频需为3–10秒有效非静音音频。')
        destination=self.local(f'data/references/{character_id}/{sha256(audio)[:16]}{audio.suffix.lower()}')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): shutil.copy2(audio,destination)
        reference={'audio_path':self.relative(destination),'text':text.strip(),'language':language}
        with self.lock:
            registry=self.read()
            registry['characters'][character_id]['tasks'][task_id]['reference']=reference
            self.save(registry)
        return reference

    def import_pair(self, character_id, task_id, gpt, sovits):
        import torch
        _,_,_,config=self.task(character_id,task_id)
        paths={'gpt':Path(gpt),'sovits':Path(sovits)}
        for stage,path in paths.items():
            if path.suffix.lower()!=('.ckpt' if stage=='gpt' else '.pth'):
                raise ValueError('GPT使用.ckpt，SoVITS使用推理.pth。')
            # Upstream replaces the ZIP signature with a two-byte model-version tag.
            # Restore that signature while retaining restricted tensor-only loading.
            with path.open('rb') as stream:
                header=stream.read(2)
                if stage=='sovits' and header in (b'00',b'01',b'02',b'03',b'04',b'05',b'06',b'07',b'08'):
                    payload=BytesIO(b'PK'+stream.read())
                else:
                    payload=path
            with torch.serialization.safe_globals([(CheckpointConfig,'utils.HParams')]):
                model=torch.load(payload,map_location='cpu',weights_only=True)
            if not isinstance(model,dict) or not isinstance(model.get('weight'),dict) or 'config' not in model:
                raise ValueError('请选择推理权重，不是G/D或Lightning续训状态。')
            model['config']=plain_config(model['config'])
            if stage=='gpt' and model['config']['model'].get('vocab_size')!=1025:
                raise ValueError('GPT语音词表不兼容。')
            if stage=='sovits' and model['config']['model'].get('version')!='v2Pro':
                raise ValueError('当前任务需要v2Pro SoVITS权重。')
            del model
        with self.lock:
            destinations={}
            for stage,path in paths.items():
                destination=self.local(config['weights'])/stage/path.name
                if destination.exists() and sha256(destination)!=sha256(path):
                    raise ValueError('已有同名且不同内容的权重，请更换导入文件名。')
                destinations[stage]=destination
            for stage,path in paths.items():
                destination=destinations[stage]
                if not destination.exists(): shutil.copy2(path,destination)
        return {k:str(v) for k,v in destinations.items()}

    def add_sources(self, character_id, task_id, files, split='train'):
        if split not in ('train','holdout'): raise ValueError('split需为train或holdout。')
        character,_,_,config=self.task(character_id,task_id)
        if (self.local(config['work'])/'dataset_manifest.json').exists():
            raise ValueError('此任务已准备数据。先新建训练任务，再改变数据集。')
        import numpy as np
        import soundfile as sf
        with self.lock:
            manifest=utils.read_json(self.local(config['source_manifest']))
            existing={row['sha256']:row for row in manifest['files']}
            for value in files:
                source=Path(value).resolve()
                wave,rate=sf.read(source,dtype='float32')
                if wave.size==0 or not np.isfinite(wave).all() or not np.any(wave):
                    raise ValueError('源音频为空、静音或数值无效。')
                identity=sha256(source)
                if identity in existing:
                    if existing[identity]['split']!=split: raise ValueError('相同源音频不能同时加入训练和留出。')
                    continue
                destination=self.local(character['raw_directory'])/'incoming'/f'{identity[:12]}_{source.name}'
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists(): shutil.copy2(source,destination)
                row={'source':self.relative(destination),'destination':self.relative(destination),'split':split,
                     'user_selected':split=='train','seconds':len(wave)/rate,'sha256':identity}
                manifest['files'].append(row)
                existing[identity]=row
            utils.write_json(self.local(config['source_manifest']),manifest)
        return len(manifest['files'])

    def summary(self, character_id, task_id):
        character,_,entry,config=self.task(character_id,task_id)
        models=self.models(character_id,task_id)
        source=utils.read_json(self.local(config['source_manifest']))
        review=self.local(config['dataset'])/'review.csv'
        status='有配套推理权重' if models['gpt'] and models['sovits'] else '未训练 / 缺少配套权重'
        return (f'**{character["display_name"]} · {task_id}** — {status}\n\n'
                f'GPT {len(models["gpt"])}份 / SoVITS {len(models["sovits"])}份；源素材 {len(source["files"])}段。\n\n'
                f'任务配置：`{entry["config"]}`\n\n原素材：`{character["raw_directory"]}`\n\n'
                f'校对：`{review.relative_to(self.root).as_posix()}`\n\n'
                f'数据：`{config["dataset"]}`；训练记录：`{config["work"]}`；权重：`{config["weights"]}`')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    add=sub.add_parser('add')
    add.add_argument('id'); add.add_argument('name'); add.add_argument('--language',default='ja',choices=LANGUAGES)
    add.add_argument('--raw-directory')
    task=sub.add_parser('new-task'); task.add_argument('id'); task.add_argument('suffix')
    sources=sub.add_parser('sources'); sources.add_argument('id'); sources.add_argument('files',nargs='+')
    sources.add_argument('--task'); sources.add_argument('--split',choices=['train','holdout'],default='train')
    args=parser.parse_args(); store=CharacterStore()
    if args.command=='list': print(json.dumps(store.read(),ensure_ascii=False,indent=2))
    elif args.command=='add': print(store.add(args.id,args.name,args.language,args.raw_directory))
    elif args.command=='new-task': print(store.new_task(args.id,args.suffix))
    else:
        _,task_id,_,_=store.task(args.id,args.task)
        print('Sources:',store.add_sources(args.id,task_id,args.files,args.split))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
