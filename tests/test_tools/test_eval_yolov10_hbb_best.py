import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


SCRIPT_PATH = (Path(__file__).resolve().parents[2] / 'tools' /
               'analysis_tools' / 'eval_yolov10_hbb_best.py')
SPEC = importlib.util.spec_from_file_location('eval_yolov10_hbb_best',
                                              SCRIPT_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class TestEvalYolov10HbbBest(unittest.TestCase):

    def setUp(self):
        self.root = (SCRIPT_PATH.parents[2] / '.eval_best_test_tmp' /
                     uuid.uuid4().hex)
        self.work_root = self.root / 'work_dirs'
        self.work_root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            self.root.parent.rmdir()
        except OSError:
            pass

    def test_selects_best_checkpoint_from_completed_run(self):
        work_dir = self.work_root / (
            'dota10_yolov10-h_s_640_80e_rand_b2x3_accum6_fp32')
        log_dir = work_dir / '20260612_000000'
        log_dir.mkdir(parents=True)
        log_path = log_dir / 'train.log'
        log_path.write_text(
            'The best checkpoint with 0.6488 pascal_voc/mAP at 76 epoch '
            'is saved to best_pascal_voc_mAP_epoch_76.pth.\n'
            'Epoch(val) [80][293/293] pascal_voc/mAP: 0.6459\n',
            encoding='utf-8')
        checkpoint = work_dir / 'best_pascal_voc_mAP_epoch_76.pth'
        checkpoint.write_bytes(b'checkpoint')

        selected, selected_log, epoch, best_map = (
            EVAL.find_best_checkpoint(self.work_root, 's'))

        self.assertEqual(selected, checkpoint)
        self.assertEqual(selected_log, log_path)
        self.assertEqual(epoch, 76)
        self.assertEqual(best_map, 0.6488)

    def test_ignores_incomplete_run_and_parses_metrics(self):
        incomplete = self.work_root / (
            'dota10_yolov10-h_m_640_80e_rand_b4x3_accum3_fp32')
        incomplete.mkdir(parents=True)
        (incomplete / 'train.log').write_text(
            'Epoch(val) [7][293/293] pascal_voc/mAP: 0.2\n',
            encoding='utf-8')
        (incomplete / 'best_pascal_voc_mAP_epoch_7.pth').write_bytes(b'old')

        with self.assertRaises(FileNotFoundError):
            EVAL.find_best_checkpoint(self.work_root, 'm')

        eval_log = self.root / 'eval.log'
        eval_log.write_text(
            'pascal_voc/mAP: 0.3210 pascal_voc/AP50: 0.6488 '
            'pascal_voc/AP75: 0.3000\n',
            encoding='utf-8')
        self.assertEqual(
            EVAL.parse_metrics(eval_log), {
                'mAP': 0.321,
                'AP50': 0.6488,
                'AP75': 0.3
            })


if __name__ == '__main__':
    unittest.main()
