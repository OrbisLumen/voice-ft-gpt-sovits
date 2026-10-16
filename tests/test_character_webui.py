"""Model selection must remain explicit and recover after loading succeeds."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import gradio as gr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from character_manager import CharacterStore
from character_webui import CharacterPanel
from tts_utils import write_json


class CharacterPanelTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        (self.root/'config').mkdir()
        shutil.copy2(ROOT/'config/task_template.json',self.root/'config/task_template.json')
        write_json(self.root/'config/characters.json',{'schema_version':1,
            'default_character':'alpha','task_template':'config/task_template.json','characters':{}})
        self.store=CharacterStore(self.root)
        self.task=self.store.add('alpha','Alpha')
        cfg=self.store.task('alpha',self.task)[3]
        self.paths=[]
        for stage,name in [('gpt','voice-e5.ckpt'),('sovits','voice_e10_s640.pth')]:
            path=self.root/cfg['weights']/stage/name
            path.write_bytes(b'fixture')
            self.paths.append(str(path))
        registry=self.store.read()
        registry['characters']['alpha']['tasks'][self.task]['reference']={
            'audio_path':'reference.wav','text':'hello','language':'en'}
        self.store.save(registry)
        self.pipeline=Mock()
        self.pipeline.configs.t2s_weights_path,self.pipeline.configs.vits_weights_path=self.paths
        with patch('character_webui.CharacterStore',return_value=self.store), patch.dict(
                os.environ,{'local_character':'alpha','local_task':self.task}):
            self.panel=CharacterPanel(self.pipeline)

    def test_selection_loading_and_success_restore_synthesis(self):
        button,status=self.panel.pending('alpha',self.task,*self.paths)
        self.assertFalse(button['interactive'])
        self.assertIn('请先加载',button['value'])
        self.assertIn('待加载',status)
        button,status=self.panel.loading('alpha',self.task)
        self.assertFalse(button['interactive'])
        self.assertIn('加载中',button['value'])
        button,status=self.panel.load('alpha',self.task,*self.paths)
        self.assertTrue(button['interactive'])
        self.assertEqual(button['value'],'合成语音')
        self.assertIn('当前已加载',status)
        self.assertIn('voice-e5.ckpt',status)
        self.pipeline.init_t2s_weights.assert_called_once_with(self.paths[0])
        self.pipeline.init_vits_weights.assert_called_once_with(self.paths[1])

    def test_failed_load_cannot_run_partially_loaded_models(self):
        self.pipeline.init_t2s_weights.side_effect=RuntimeError('load failed')
        button,status=self.panel.load('alpha',self.task,*self.paths)
        self.assertFalse(button['interactive'])
        self.assertIn('load failed',status)
        with self.assertRaises(gr.Error):
            list(self.panel.run('alpha',self.task,{}))
        self.pipeline.run.assert_not_called()

    def test_header_contains_model_selection_and_load_together(self):
        with gr.Blocks() as app:
            self.panel.header()
        config=app.get_config_file()
        components={item['id']:item for item in config['components']}
        group=config['layout']['children'][0]
        ids=[]
        def collect(node):
            ids.append(node['id'])
            for child in node.get('children',[]): collect(child)
        collect(group)
        for component in (self.panel.gpt,self.panel.sovits,self.panel.load_button):
            self.assertIn(component._id,ids)
        self.assertEqual(components[self.panel.gpt._id]['props']['value'],self.paths[0])
        self.assertEqual(components[self.panel.sovits._id]['props']['value'],self.paths[1])


if __name__=='__main__': unittest.main()
