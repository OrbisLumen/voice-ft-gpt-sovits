"""Registry isolation and checkpoint selection across arbitrary characters/tasks."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from character_manager import CharacterStore
from tts_utils import write_json


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        (self.root/'config').mkdir()
        shutil.copy2(ROOT/'config/task_template.json',self.root/'config/task_template.json')
        write_json(self.root/'config/characters.json',{'schema_version':1,
            'default_character':'alpha','task_template':'config/task_template.json','characters':{}})
        self.store=CharacterStore(self.root)

    def weights(self,character,task,g=10,s=20):
        config=self.store.task(character,task)[3]
        result=[]
        for stage,filename in [('gpt',f'custom-e{g}.ckpt'),('sovits',f'custom_e{s}_s999.pth')]:
            path=self.root/config['weights']/stage/filename
            path.write_bytes(b'fixture')
            result.append(str(path.resolve()))
        return result

    def test_many_characters_and_tasks_are_isolated(self):
        paths=set()
        for index in range(5):
            char=f'voice{index}'
            self.store.add(char,f'角色{index}',('ja','zh','en','ko','yue')[index])
            self.store.new_task(char,'alternate')
            for task in self.store.tasks(char):
                config=self.store.task(char,task)[3]
                for key in ('source_manifest','dataset','work','output','weights'):
                    self.assertNotIn(config[key],paths)
                    paths.add(config[key])
                self.assertEqual(config['language'],('ja','zh','en','ko','yue')[index])
        self.assertEqual(len(self.store.characters()),5)

    def test_cross_character_and_task_pair_cannot_load(self):
        a=self.store.add('alpha','A')
        self.assertEqual(a,'alpha_gsv_v2pro_v1')
        b=self.store.add('beta','B')
        newer=self.store.new_task('alpha','v2')
        g,s=self.weights('alpha',a)
        other_g,other_s=self.weights('beta',b)
        self.assertEqual(self.store.validate_pair('alpha',a,g,s),(g,s))
        for role,task,gpt,sovits in [('alpha',a,g,other_s),('beta',b,g,other_s),('alpha',newer,g,s)]:
            with self.assertRaises(ValueError): self.store.validate_pair(role,task,gpt,sovits)
        self.assertEqual(self.store.defaults('alpha',newer),{'gpt':None,'sovits':None})

    def test_epoch_sort_and_persisted_selection(self):
        task=self.store.add('alpha','A')
        old=self.weights('alpha',task,2,3)
        preferred=self.weights('alpha',task)
        latest=self.weights('alpha',task,100,100)
        self.assertEqual(list(self.store.defaults('alpha',task).values()),preferred)
        self.assertEqual(self.store.models('alpha',task)['gpt'][0][1],latest[0])
        self.store.save_selection('alpha',task,*old)
        self.assertEqual(list(CharacterStore(self.root).defaults('alpha',task).values()),old)
        Path(old[0]).unlink()
        self.assertEqual(self.store.defaults('alpha',task)['gpt'],preferred[0])

    def test_ids_and_existing_tasks_never_overwrite(self):
        for value in ('../escape','con','nul','bad/name','中文','UPPER'):
            with self.assertRaises(ValueError): self.store.add(value,'test')
        self.store.add('alpha','A')
        before=self.store.path.read_bytes()
        with self.assertRaises(ValueError): self.store.add('alpha','B')
        with self.assertRaises(ValueError): self.store.new_task('alpha','v1')
        self.assertEqual(before,self.store.path.read_bytes())
        with self.assertRaises(ValueError): self.store.local('../outside')


if __name__=='__main__': unittest.main()
