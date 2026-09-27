import argparse
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import infer_masks
from sam3_face_attributes.paths import iter_images, prepare_io


class PortableInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / 'sam3.pt'
        self.checkpoint.touch()

    def image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 10), (110, 100, 90)).save(path)
        return path

    def args(self, source, output=None):
        return argparse.Namespace(
            input=source, output_dir=output or self.root / 'masks',
            checkpoint_path=self.checkpoint, recursive=True,
            config=ROOT / 'configs/makeup_complex.json',
        )

    def test_infers_task_and_allows_override(self):
        for task in ('makeup', 'jewelry', 'Makeup'):
            folder = self.root / task
            folder.mkdir(exist_ok=True)
            self.assertEqual(infer_masks.select_task(folder), task.lower())
        source = self.image('makeup/a.png')
        self.assertEqual(infer_masks.select_task(source), 'makeup')
        self.assertEqual(infer_masks.select_task(self.root, 'jewelry'), 'jewelry')
        with self.assertRaisesRegex(ValueError, 'Cannot infer task'):
            infer_masks.select_task(self.root)

    def test_list_paths_resolve_from_list_and_expand_variables(self):
        source = self.image('data/a.png')
        listing = self.root / 'images.txt'
        listing.write_text('data/a.png\n\n$MASK_TEST_ROOT/data/a.png\n')
        with patch.dict(os.environ, {'MASK_TEST_ROOT': str(self.root)}):
            self.assertEqual(iter_images(listing), [source])
            self.assertEqual(prepare_io(self.args(listing))[0][1].name, 'a.png.png')

    def test_recursive_scan_and_filename_collisions(self):
        for name in ('makeup/a.jpg', 'makeup/a.png', 'makeup/nested/a.jpg'):
            self.image(name)
        (self.root / 'makeup/fake.jpg').mkdir()
        self.assertEqual(len(iter_images(self.root / 'makeup', False)), 2)
        jobs = prepare_io(self.args(self.root / 'makeup'))
        self.assertEqual({str(dst.relative_to(self.root / 'masks')) for _, dst in jobs},
                         {'a.jpg.png', 'a.png.png', 'nested/a.jpg.png'})

    def test_missing_checkpoint_fails_before_model_loading(self):
        source = self.image('makeup/a.png')
        self.checkpoint.unlink()
        module = importlib.import_module('infer_makeup')
        with patch.object(module, 'build_processor') as loader:
            with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint'):
                module.main(['--input', str(source), '--output-dir', str(self.root / 'out'),
                             '--checkpoint-path', str(self.checkpoint)])
            loader.assert_not_called()

    def test_rejects_empty_missing_and_nested_output(self):
        folder = self.root / 'makeup'
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, 'No supported images'):
            prepare_io(self.args(folder))
        with self.assertRaisesRegex(FileNotFoundError, 'Input does not exist'):
            prepare_io(self.args(self.root / 'missing'))
        self.image('makeup/a.png')
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare_io(self.args(folder, folder / 'masks'))

    def test_rejects_invalid_sam3_repository(self):
        source = self.image('makeup/a.png')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            infer_masks.main(['--input', str(source), '--output-dir', str(self.root / 'masks'),
                              '--checkpoint-path', str(self.checkpoint),
                              '--sam3-repo', str(self.root)])

    def test_processes_every_image_and_writes_only_binary_masks(self):
        for task in ('makeup', 'jewelry'):
            with self.subTest(task=task):
                module = importlib.import_module('infer_' + task)
                for name in ('a.jpg', 'a.png', 'nested/a.jpg', 'b.png', 'c.png', 'd.png', 'e.png'):
                    self.image(f'{task}/{name}')
                attribute = 'lip_color' if task == 'makeup' else 'nose_jewelry'
                config = self.root / (task + '.json')
                item = {'name': attribute, 'prompts': ['test'], 'threshold': 0.4,
                        'min_sam_score': 0.3, 'min_quality': 0.4,
                        'max_crop_coverage': 0.3, 'max_face_coverage': 0.2}
                config.write_text(json.dumps({task: [item]}))
                mask = np.zeros((10, 12), dtype=bool)
                mask[2:4, 3:6] = True
                common = dict(face_index=0, attribute=attribute, prompt='test', roi='full_image',
                              quality=0.9, mask=mask, crop_coverage=0.05, face_coverage=0.05)
                if task == 'makeup':
                    candidate = module.Candidate(**common, score=0.9, metrics={})
                    branch = 'run_candidate_branch'
                else:
                    candidate = module.Candidate(**common, sam_score=0.9, border_contacts=0)
                    branch = 'run_branch'
                mesh = SimpleNamespace(close=Mock())
                processor = SimpleNamespace(set_image=Mock(return_value={}))
                out = self.root / (task + '_masks')
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(module, 'build_processor', return_value=processor))
                    stack.enter_context(patch.object(module, 'load_face_mesh', return_value=mesh))
                    stack.enter_context(patch.object(module, 'landmark_faces', return_value=[]))
                    stack.enter_context(patch.object(module, 'autocast_context', side_effect=lambda *_: contextlib.nullcontext()))
                    stack.enter_context(patch.object(module, branch, return_value=([candidate], [])))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    infer_masks.main(['--input', str(self.root / task), '--output-dir', str(out),
                                      '--checkpoint-path', str(self.checkpoint), '--config', str(config)])
                files = [p for p in out.rglob('*') if p.is_file()]
                self.assertEqual(len(files), 7)
                self.assertTrue((out / 'nested/a.jpg.png').is_file())
                self.assertTrue((out / 'a.jpg.png').is_file())
                self.assertTrue((out / 'a.png.png').is_file())
                for path in files:
                    self.assertEqual(path.suffix, '.png')
                    with Image.open(path) as result:
                        self.assertEqual(result.mode, 'L')
                        self.assertEqual(result.size, (12, 10))
                        np.testing.assert_array_equal(np.asarray(result), mask.astype(np.uint8) * 255)
                mesh.close.assert_called_once()

    def test_no_limit_option_and_recursion_defaults(self):
        for task in ('makeup', 'jewelry'):
            module = importlib.import_module('infer_' + task)
            argv = ['--input', str(self.root), '--output-dir', str(self.root / 'out'),
                    '--checkpoint-path', str(self.checkpoint)]
            self.assertTrue(module.parse_args(argv).recursive)
            self.assertFalse(module.parse_args([*argv, '--no-recursive']).recursive)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.parse_args([*argv, '--limit', '5'])

    def test_shell_wrappers_forward_environment_and_arguments(self):
        recorder = self.root / 'record-python'
        recorder.write_text(f'#!{sys.executable}\nimport json, os, sys\nprint(json.dumps(dict(args=sys.argv[1:], cuda=os.getenv("CUDA_VISIBLE_DEVICES"))))\n')
        recorder.chmod(0o755)
        keys = ('INPUT_DIR', 'OUTPUT_DIR', 'CHECKPOINT', 'SAM3_REPO', 'CONFIG', 'TASK',
                'CUDA_DEVICE', 'CONDA_ENV', 'PYTHON')
        environment = {k: v for k, v in os.environ.items() if k not in keys}
        environment.update(PYTHON=str(recorder), INPUT_DIR='/new server/data/makeup',
                           OUTPUT_DIR='/new server/results', CHECKPOINT='/new server/sam3.pt',
                           SAM3_REPO='/new server/sam3', CUDA_DEVICE='2')
        for task in ('masks', 'makeup', 'jewelry'):
            output = subprocess.check_output(['bash', str(ROOT / 'scripts' / f'run_{task}_server.sh'),
                                              '--no-recursive', '--dtype', 'float32'],
                                             env=environment, text=True)
            captured = json.loads(output)
            self.assertEqual(captured['cuda'], '2')
            self.assertIn('/new server/data/makeup', captured['args'])
            self.assertEqual(captured['args'][-3:], ['--no-recursive', '--dtype', 'float32'])
            self.assertNotIn('--limit', captured['args'])
            if task != 'masks':
                self.assertEqual(captured['args'][captured['args'].index('--task') + 1], task)
        # Also exercise the launcher without any optional environment arguments.
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        clean['PYTHON'] = str(recorder)
        output = subprocess.check_output(['bash', str(ROOT / 'scripts/run_masks_server.sh'), '--help'],
                                         env=clean, text=True)
        self.assertEqual(json.loads(output)['args'][-1], '--help')


if __name__ == '__main__':
    unittest.main()
