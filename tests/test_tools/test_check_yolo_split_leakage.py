import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


SCRIPT_PATH = (Path(__file__).resolve().parents[2] / 'tools' /
               'analysis_tools' / 'check_yolo_split_leakage.py')
SPEC = importlib.util.spec_from_file_location('check_yolo_split_leakage',
                                              SCRIPT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TestCheckYoloSplitLeakage(unittest.TestCase):

    def setUp(self):
        self.root = (SCRIPT_PATH.parents[2] / '.audit_test_tmp' /
                     uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            self.root.parent.rmdir()
        except OSError:
            pass

    def _make_layout(self, data_root):
        for relative in ('images/train', 'images/val', 'labels/train',
                         'labels/val'):
            (data_root / relative).mkdir(parents=True)

    def _write_sample(self,
                      data_root,
                      split,
                      name,
                      content,
                      label='0 0.5 0.5 0.2 0.2\n'):
        (data_root / 'images' / split / f'{name}.jpg').write_bytes(content)
        (data_root / 'labels' / split / f'{name}.txt').write_text(
            label, encoding='utf-8')

    def test_clean_split_passes_and_writes_reports(self):
        data_root = self.root / 'clean_data'
        output_dir = self.root / 'reports'
        self._make_layout(data_root)
        self._write_sample(data_root, 'train', 'P0001__0_0_1024', b'train',
                           '0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n')
        self._write_sample(data_root, 'val', 'P0002__0_0_1024', b'val',
                           '1 0.5 0.5 0.2 0.2\n')

        report = AUDIT.audit_dataset(data_root, output_dir)

        self.assertTrue(report['decisions']['overall_pass'])
        self.assertEqual(report['decisions']['same_name_overlap_count'], 0)
        self.assertEqual(report['decisions']['sha256_overlap_count'], 0)
        self.assertEqual(report['decisions']['source_id_overlap_count'], 0)
        self.assertEqual(report['counts']['train']['instances'], 2)
        self.assertEqual(report['counts']['val']['instances'], 1)
        self.assertTrue((output_dir / 'audit_report.json').is_file())
        self.assertTrue((output_dir / 'class_distribution.csv').is_file())
        self.assertTrue((output_dir / 'suspicious_files.csv').is_file())

    def test_detects_name_hash_source_and_integrity_problems(self):
        data_root = self.root / 'leaky_data'
        output_dir = self.root / 'reports'
        self._make_layout(data_root)

        self._write_sample(data_root, 'train', 'P1000__0_0_1024',
                           b'name-train')
        self._write_sample(data_root, 'val', 'P1000__0_0_1024', b'name-val')
        self._write_sample(data_root, 'train', 'P2000__0_0_1024',
                           b'exact-copy')
        self._write_sample(data_root, 'val', 'P3000__0_0_1024',
                           b'exact-copy')
        self._write_sample(data_root, 'train', 'P4000__0_0_1024',
                           b'source-train')
        self._write_sample(data_root, 'val', 'P4000__200_0_1024',
                           b'source-val')

        (data_root / 'images/train/P5000__0_0_1024.jpg').write_bytes(
            b'no-label')
        (data_root / 'labels/val/P6000__0_0_1024.txt').write_text(
            '0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
        self._write_sample(data_root, 'val', 'P7000__0_0_1024',
                           b'malformed',
                           'not-an-id 0.5 0.5 0.2 0.2\n')

        report = AUDIT.audit_dataset(data_root, output_dir)

        self.assertFalse(report['decisions']['overall_pass'])
        self.assertEqual(report['decisions']['same_name_overlap_count'], 1)
        self.assertEqual(report['decisions']['sha256_overlap_count'], 1)
        self.assertEqual(report['decisions']['source_id_overlap_count'], 2)
        self.assertEqual(
            report['integrity']['train']['images_without_labels'],
            ['P5000__0_0_1024.jpg'])
        self.assertEqual(
            report['integrity']['val']['labels_without_images'],
            ['P6000__0_0_1024.txt'])
        self.assertEqual(len(report['integrity']['malformed_labels']), 1)
        self.assertEqual(
            AUDIT.main([str(data_root), '--output-dir', str(output_dir)]), 1)


if __name__ == '__main__':
    unittest.main()
