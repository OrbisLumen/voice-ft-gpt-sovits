"""Role management panel plugged into the official GPT-SoVITS inference UI."""
import os
from pathlib import Path
import threading

import gradio as gr

from character_manager import CharacterStore, LANGUAGES


class CharacterPanel:
    def __init__(self, pipeline):
        self.store=CharacterStore()
        self.pipeline=pipeline
        self.lock=threading.Lock()
        self.loaded=(os.environ['local_character'],os.environ['local_task'])
        self.ready=True
        self.loaded_paths=(str(pipeline.configs.t2s_weights_path),str(pipeline.configs.vits_weights_path))
        try:
            self.store.validate_pair(*self.loaded,*self.loaded_paths)
        except ValueError:
            owner=None
            for character_id,character in self.store.read()['characters'].items():
                for task_id in character['tasks']:
                    try:
                        self.store.validate_pair(character_id,task_id,*self.loaded_paths)
                        owner=(character_id,task_id)
                        break
                    except ValueError: pass
                if owner: break
            if owner is None: raise ValueError('启动权重未登记为同一角色任务的配套模型。')
            self.loaded=owner
        _,_,entry,config=self.store.task(*self.loaded)
        reference=entry.get('reference') or {}
        self.reference_audio=str(self.store.local(reference['audio_path'])) if reference.get('audio_path') else None
        self.reference_text=reference.get('text','')
        self.reference_language=LANGUAGES.get(reference.get('language',config['language']),'日文')

    def header(self):
        character_id,task_id=self.loaded
        models=self.store.models(character_id,task_id)
        with gr.Group():
            gr.Markdown('## 选择并加载声音模型')
            gr.Markdown('① 选择角色和任务　② 选择 GPT 与 SoVITS 轮次　③ 点击加载，再到下方合成语音。')
            with gr.Row():
                self.character=gr.Dropdown(label='角色',choices=self.store.characters(),value=character_id)
                self.task=gr.Dropdown(label='训练任务',choices=self.store.tasks(character_id),value=task_id)
                self.refresh_roles=gr.Button('刷新角色列表')
            with gr.Row():
                self.gpt=gr.Dropdown(label='GPT模型列表',choices=models['gpt'],value=self.loaded_paths[0],interactive=True)
                self.sovits=gr.Dropdown(label='SoVITS模型列表',choices=models['sovits'],value=self.loaded_paths[1],interactive=True)
            with gr.Row():
                self.load_button=gr.Button('加载所选模型',variant='primary')
                self.refresh_models=gr.Button('刷新模型路径')
            self.status=gr.Markdown(self._status(character_id,task_id,'模型已就绪，可以合成语音。'))
        with gr.Accordion('管理角色、数据与参考音频',open=False):
            gr.Markdown('角色和训练任务各自拥有数据、校对、权重及输出目录。已有任务的数据准备后，请新建任务再改变源素材。')
            with gr.Row():
                self.new_id=gr.Textbox(label='新角色ID（小写英文，例如 shizuku）')
                self.new_name=gr.Textbox(label='角色显示名称')
                self.new_language=gr.Dropdown(label='训练素材语种',choices=[(v,k) for k,v in LANGUAGES.items()],value='ja')
                self.add_button=gr.Button('登记新角色')
            with gr.Row():
                self.task_suffix=gr.Textbox(label='新训练任务后缀（例如 v2）')
                self.task_button=gr.Button('为当前角色新建任务')
            with gr.Row():
                self.source_files=gr.File(label='加入当前任务的原音声',file_count='multiple',file_types=['audio'],type='filepath')
                self.source_split=gr.Dropdown(label='源级数据划分',choices=[('训练','train'),('留出','holdout')],value='train')
                self.source_button=gr.Button('保存源素材到当前任务')
            with gr.Row():
                self.import_gpt=gr.File(label='导入GPT推理权重',file_types=['.ckpt'],type='filepath')
                self.import_sovits=gr.File(label='导入SoVITS推理权重',file_types=['.pth'],type='filepath')
                self.import_button=gr.Button('导入配套模型')
            self.reference_button=gr.Button('将下方参考音频、台词与语种保存给当前角色任务')
            self.message=gr.Markdown('选择角色、任务及两份权重后，点击“加载所选模型”。')

    def _status(self,character_id,task_id,message):
        loaded=' / '.join(f'`{Path(path).name}`' for path in self.loaded_paths)
        state='当前已加载' if self.ready else '上次加载（当前不可合成）'
        return (f'**{message}**\n\n{state}：{self.loaded[0]} · {self.loaded[1]}\n\n'
                f'{loaded}')

    @staticmethod
    def generate_update(ready,reason='请先加载所选模型'):
        return gr.update(interactive=ready,value='合成语音' if ready else reason)

    def _select(self,character_id,task_id):
        with self.lock: self.ready=False
        _,task_id,entry,config=self.store.task(character_id,task_id)
        models=self.store.models(character_id,task_id)
        default=self.store.defaults(character_id,task_id)
        reference=entry.get('reference') or {}
        audio=str(self.store.local(reference['audio_path'])) if reference.get('audio_path') else None
        return (gr.update(choices=models['gpt'],value=default['gpt']),
                gr.update(choices=models['sovits'],value=default['sovits']),audio,reference.get('text',''),
                LANGUAGES.get(reference.get('language',config['language']),'日文'),
                self.generate_update(False),config['seed'],
                self._status(character_id,task_id,'尚未加载所选模型。点击“加载所选模型”；无权重的任务不能合成。'),None)

    def select_character(self,character_id):
        _,task_id,_,_=self.store.task(character_id)
        return (gr.update(choices=self.store.tasks(character_id),value=task_id),
                *self._select(character_id,task_id))

    def select_task(self,character_id,task_id):
        return self._select(character_id,task_id)

    def refresh(self,character_id,task_id):
        return self._select(character_id,task_id)

    def pending(self,character_id,task_id,gpt,sovits):
        with self.lock: self.ready=False
        selected=' / '.join(f'`{Path(path).name}`' if path else '未选择' for path in (gpt,sovits))
        return (self.generate_update(False),
                self._status(character_id,task_id,'选择已更改，请点击上方“加载所选模型”。')+
                f'\n\n待加载：{selected}')

    def loading(self,character_id,task_id):
        with self.lock: self.ready=False
        return self.generate_update(False,'模型加载中…'),self._status(character_id,task_id,'正在加载所选模型，请稍候…')

    def load(self,character_id,task_id,gpt,sovits):
        try:
            gpt,sovits=self.store.validate_pair(character_id,task_id,gpt,sovits)
            with self.lock:
                self.ready=False
                self.pipeline.init_vits_weights(sovits)
                self.pipeline.init_t2s_weights(gpt)
                self.loaded=(character_id,task_id)
                self.loaded_paths=(gpt,sovits)
                self.ready=True
                self.store.save_selection(character_id,task_id,gpt,sovits)
            _,_,entry,_=self.store.task(character_id,task_id)
            reference=entry.get('reference')
            can_generate=bool(reference)
            message='配套模型已加载。' if can_generate else '配套模型已加载；先填写下方参考信息并保存给此任务。'
            return self.generate_update(can_generate,'请先保存参考音频'),self._status(character_id,task_id,message)
        except Exception as exc:
            with self.lock: self.ready=False
            return self.generate_update(False,'模型加载失败，请重新选择并加载'),self._status(character_id,task_id,f'加载失败：{exc}')

    def run(self,character_id,task_id,inputs):
        with self.lock:
            if not self.ready or self.loaded!=(character_id,task_id):
                raise gr.Error('当前角色/任务尚未加载配套模型，请先点击“加载所选模型”。')
            self.store.validate_pair(character_id,task_id,*self.loaded_paths)
            yield from self.pipeline.run(inputs)

    def add(self,character_id,name,language):
        try:
            task_id=self.store.add(character_id.strip(),name,language)
            return gr.update(choices=self.store.characters(),value=character_id.strip()),f'已登记角色并建立独立任务：{task_id}。尚未训练。'
        except Exception as exc: return gr.update(),f'登记失败：{exc}'

    def new_task(self,character_id,suffix):
        try:
            task_id=self.store.new_task(character_id,suffix.strip())
            return gr.update(choices=self.store.tasks(character_id),value=task_id),f'新任务：{task_id}。原任务和模型保留。'
        except Exception as exc: return gr.update(),f'新建失败：{exc}'

    def sources(self,character_id,task_id,files,split):
        try:
            if not files: raise ValueError('先选择源音声文件。')
            count=self.store.add_sources(character_id,task_id,files,split)
            return f'已保存，当前任务共{count}段源素材。下一步prepare、transcribe和人工校对。',self._status(character_id,task_id,'源素材已更新。')
        except Exception as exc: return f'保存失败：{exc}',self._status(character_id,task_id,'源素材未更新。')

    def import_models(self,character_id,task_id,gpt,sovits):
        try:
            if not gpt or not sovits: raise ValueError('同时选择两份推理权重。')
            pair=self.store.import_pair(character_id,task_id,gpt,sovits)
            with self.lock: self.ready=False
            choices=self.store.models(character_id,task_id)
            return (gr.update(choices=choices['gpt'],value=pair['gpt']),
                    gr.update(choices=choices['sovits'],value=pair['sovits']),self.generate_update(False),
                    '导入成功，请点击“加载所选模型”。')
        except Exception as exc: return gr.update(),gr.update(),gr.update(),f'导入失败：{exc}'

    def save_reference(self,character_id,task_id,audio,text,language):
        try:
            if not audio: raise ValueError('先填写下方参考音频。')
            code=next((key for key,value in LANGUAGES.items() if value==language),None)
            self.store.set_reference(character_id,task_id,audio,text,code)
            with self.lock: can_generate=self.ready and self.loaded==(character_id,task_id)
            return ('已保存参考音频、台词与语种，下次选择此任务会自动填入。',
                    self._status(character_id,task_id,'参考信息已保存。'),self.generate_update(can_generate))
        except Exception as exc: return f'保存失败：{exc}',gr.update(),gr.update()

    def bind(self,gpt,sovits,audio,prompt,prompt_language,generate,seed,aux_refs,refresh):
        _,_,_,config=self.store.task(*self.loaded)
        models=self.store.models(*self.loaded)
        gpt.choices=models['gpt']
        sovits.choices=models['sovits']
        generate.interactive=bool(self.reference_audio)
        if not generate.interactive: generate.value='请先保存参考音频'
        self.quick_load=gr.Button('加载所选模型（与顶部相同）')
        gr.Markdown('更换角色或模型轮次后，请先加载；加载成功后“合成语音”按钮会恢复可用。')
        outputs=[gpt,sovits,audio,prompt,prompt_language,generate,seed,self.status,aux_refs]
        queue={'concurrency_id':'character_models','concurrency_limit':1}
        self.character.change(self.select_character,[self.character],[self.task,*outputs],**queue)
        self.task.change(self.select_task,[self.character,self.task],outputs,**queue)
        self.refresh_roles.click(lambda:gr.update(choices=self.store.characters()),[],self.character,**queue)
        refresh.click(self.refresh,[self.character,self.task],outputs,**queue)
        selection=[self.character,self.task,gpt,sovits]
        gpt.input(self.pending,selection,[generate,self.status],**queue)
        sovits.input(self.pending,selection,[generate,self.status],**queue)
        for button in (self.load_button,self.quick_load):
            button.click(self.loading,[self.character,self.task],[generate,self.status],**queue).then(
                self.load,selection,[generate,self.status],**queue)
        self.add_button.click(self.add,[self.new_id,self.new_name,self.new_language],[self.character,self.message],**queue)
        self.task_button.click(self.new_task,[self.character,self.task_suffix],[self.task,self.message],**queue)
        self.source_button.click(self.sources,[self.character,self.task,self.source_files,self.source_split],[self.message,self.status],**queue)
        self.import_button.click(self.import_models,[self.character,self.task,self.import_gpt,self.import_sovits],
                                 [gpt,sovits,generate,self.message],**queue)
        self.reference_button.click(self.save_reference,[self.character,self.task,audio,prompt,prompt_language],
                                    [self.message,self.status,generate],**queue)
