"""Offline checks for reviewed-label gating and isolated TTS task configuration."""
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tts_finetune import read_review, data_signature

ROOT = Path(__file__).resolve().parents[1]


class TTSTests(unittest.TestCase):
    def review(self, **changes):
        row = dict(name='a.wav', language='ja', text='こんにちは。', reviewed='true', audio_path='a.wav')
        row.update(changes)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'review.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            return read_review(path, {'clips': [{'name': 'a.wav', 'eligible': True}]})

    def test_unreviewed_asr_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError, 'not reviewed'):
            self.review(reviewed='false')

    def test_invalid_text_and_wrong_language_rejected(self):
        for changes in ({'text': ''}, {'text': 'bad|label'}, {'language': 'zh'}, {'name': 'missing.wav'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.review(**changes)

    def test_reviewed_japanese_is_accepted(self):
        self.assertEqual(self.review()[0]['text'], 'こんにちは。')

    def test_task_separates_data_run_models_and_audio(self):
        cfg = json.loads((ROOT/'config/task.json').read_text(encoding='utf-8'))
        self.assertEqual(cfg['repository'], 'vendor/GPT-SoVITS')
        self.assertEqual(cfg['version'], 'v2Pro')
        self.assertTrue(cfg['selected_only'])
        self.assertTrue(cfg['dataset'].startswith('data/datasets/'))
        self.assertTrue(cfg['work'].startswith('runs/'))
        self.assertTrue(cfg['weights'].startswith('models/'))
        self.assertTrue(cfg['source_manifest'].startswith('data/manifests/'))
        self.assertEqual(cfg['training']['sovits_batch_size'], 2)

    def test_training_budget_changes_reuse_audio_but_selection_changes_do_not(self):
        cfg = json.loads((ROOT/'config/task.json').read_text(encoding='utf-8'))
        original = data_signature(cfg)
        cfg['training']['sovits_batch_size'] = 1
        self.assertEqual(data_signature(cfg), original)
        cfg['max_seconds'] = 8
        self.assertNotEqual(data_signature(cfg), original)

    def test_notebook_has_valid_python_and_review_gate_before_training(self):
        nb = json.loads((ROOT/'notebooks/finetune_misuzu_tts.ipynb').read_text(encoding='utf-8'))
        code = []
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                ast.parse(source)
                self.assertEqual(cell['outputs'], [])
                code.append(source)
        self.assertIn('RUN_ASR = False', '\n'.join(code))
        self.assertIn('RUN_TRAIN = True', '\n'.join(code))
        joined = '\n'.join(code)
        self.assertLess(joined.index("stage('train-gpt')"), joined.index("stage('sample')"))


if __name__ == '__main__':
    unittest.main()
