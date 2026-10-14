"""Reject confounded TTS checkpoint comparisons before processing audio."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_tts_checkpoints import validate_pair


class TTSComparisonTests(unittest.TestCase):
    def setUp(self):
        self.first = {'reference': {'audio_path': 'ref.wav', 'text': '参考音频'},
            'settings': {'t2s_weights_path': 'gpt5.ckpt', 'vits_weights_path': 'sovits10.pth', 'is_half': True},
            'results': [{'name': 'zh_daily', 'text': '同一句话', 'language': 'all_zh'}],
            'inference_controls': {'seed': 1234, 'speed_factor': 1}}
        self.second = copy.deepcopy(self.first)
        self.second['settings']['vits_weights_path'] = 'sovits20.pth'

    def test_only_sovits_can_change(self):
        self.assertEqual(set(validate_pair(self.first, self.second)[0]), {'zh_daily'})

    def test_changed_gpt_reference_text_or_seed_is_rejected(self):
        for kind in ('gpt', 'reference', 'text', 'seed'):
            with self.subTest(kind=kind):
                second = copy.deepcopy(self.second)
                if kind == 'gpt':
                    second['settings']['t2s_weights_path'] = 'gpt10.ckpt'
                elif kind == 'reference':
                    second['reference']['audio_path'] = 'different.wav'
                elif kind == 'text':
                    second['results'][0]['text'] = '另一句话'
                else:
                    second['inference_controls']['seed'] = 5678
                with self.assertRaises(ValueError):
                    validate_pair(self.first, second)

    def test_gpt_comparison_requires_fixed_sovits_and_controls(self):
        second = copy.deepcopy(self.first)
        second['settings']['t2s_weights_path'] = 'gpt10.ckpt'
        self.assertEqual(set(validate_pair(self.first, second, phase='gpt')[0]), {'zh_daily'})
        for kind in ('sovits', 'seed', 'reference', 'text'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(second)
                if kind == 'sovits':
                    changed['settings']['vits_weights_path'] = 'sovits20.pth'
                elif kind == 'seed':
                    changed['inference_controls']['seed'] = 5678
                elif kind == 'reference':
                    changed['reference']['audio_path'] = 'different.wav'
                else:
                    changed['results'][0]['text'] = '另一句话'
                with self.assertRaises(ValueError):
                    validate_pair(self.first, changed, phase='gpt')


if __name__ == '__main__':
    unittest.main()
