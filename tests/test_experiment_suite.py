from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests._bootstrap import REPO_ROOT, load_module

suite_mod = load_module('utils.experiment_suite')
config_mod = load_module('engine.config')


class TestPaperSuite(unittest.TestCase):
    def setUp(self):
        self.suite = suite_mod.load_suite('configs/suites/paper_train.py')

    def test_all_ten_training_variants_share_the_full_budget(self):
        jobs = suite_mod.train_jobs(self.suite)
        self.assertEqual(len(jobs), 10)
        self.assertEqual(jobs[0]['variant'], 'full')
        cfg = jobs[0]['cfg']
        self.assertEqual(cfg['train_dataloader']['batch_size'], 4)
        self.assertEqual(cfg['train_cfg']['max_iters'], 10000)
        self.assertEqual(cfg['optimizer']['lr'], 1e-4)
        self.assertEqual(cfg['param_scheduler'][0]['end'], 500)
        self.assertEqual(cfg['param_scheduler'][1]['T_max'], 9500)
        self.assertEqual(cfg['logging']['wandb']['mode'], 'offline')
        self.assertEqual(cfg['model']['aggregation']['num_blocks'], 2)
        self.assertTrue((REPO_ROOT / 'configs/experiments/ablation').is_dir())

    def test_manual_selection_restores_architecture_and_all_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = suite_mod.load_suite('configs/suites/paper_validate.py')
            suite['baselines'] = []
            source = suite_mod.train_jobs(self.suite, ['full'])[0]['cfg']
            source['model']['aggregation']['share_window_weights_across_scales'] = False
            source = config_mod.finalize_config(source)
            checkpoint = Path(tmp) / 'iter_0003000.pth'
            checkpoint.write_bytes(b'synthetic checkpoint; no model execution')
            source['meta'] = dict(config_fingerprint=config_mod.config_fingerprint(source))
            (checkpoint.parent / 'config.json').write_text(json.dumps(source))
            suite['selections'] = [dict(variant='full', seed=42, checkpoint=str(checkpoint))]
            jobs, pending = suite_mod.validation_jobs(suite)
            self.assertEqual(pending, [])
            self.assertEqual(len(jobs), 9)
            self.assertEqual({j['checkpoint_sha256'] for j in jobs}, {suite_mod.sha256_file(checkpoint)})
            for job in jobs:
                self.assertEqual(job['cfg']['model']['pipeline'], 'reca')
                self.assertTrue(job['cfg']['model']['remoteclip']['enabled'])
                self.assertFalse(job['cfg']['model']['aggregation']['share_window_weights_across_scales'])
                self.assertEqual(job['cfg']['training']['dataset'], 'dlrsd')
                self.assertEqual(job['cfg']['evaluation']['dataset'], job['eval_dataset'])
            suite['selections'][0]['variant'] = 'retired_variant'
            with self.assertRaisesRegex(ValueError, 'Unknown or duplicate'):
                suite_mod.validation_jobs(suite)

    def test_no_automatic_checkpoint_selection(self):
        suite = suite_mod.load_suite('configs/suites/paper_validate.py')
        jobs, pending = suite_mod.validation_jobs(suite, dry_run=True)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len(pending), 10)
        self.assertTrue(all(j['pretrained'] for j in jobs))

    def test_validation_uses_target_dataset_threshold(self):
        source = suite_mod.train_jobs(self.suite)[0]['cfg']
        source['evaluation']['postprocess']['prediction_prob_threshold'] = 0.9
        original_build = suite_mod.build_effective_config

        def build_target(path):
            cfg, fingerprint = original_build(path)
            threshold = 0.35 if Path(path).stem == 'loveda' else 0.65
            cfg['evaluation']['postprocess']['prediction_prob_threshold'] = threshold
            return cfg, fingerprint

        with patch.object(suite_mod, 'build_effective_config', side_effect=build_target):
            loveda = suite_mod.evaluation_config(source, 'loveda', '/tmp/eval_loveda')
            potsdam = suite_mod.evaluation_config(source, 'potsdam', '/tmp/eval_potsdam')
        self.assertEqual(loveda['evaluation']['postprocess']['prediction_prob_threshold'], 0.35)
        self.assertEqual(potsdam['evaluation']['postprocess']['prediction_prob_threshold'], 0.65)
        self.assertEqual(source['evaluation']['postprocess']['prediction_prob_threshold'], 0.9)

    def test_retired_ablation_options_are_rejected(self):
        base = suite_mod.train_jobs(self.suite, ['full'])[0]['cfg']
        for key, value in dict(update_streams='feature_only', semantic_init='zeros', shifted_windows=True).items():
            cfg = copy.deepcopy(base)
            cfg['model']['aggregation'][key] = value
            with self.subTest(key=key), self.assertRaises(KeyError):
                config_mod.finalize_config(cfg)
        cfg = copy.deepcopy(base)
        cfg['model']['fusion']['orientations'] = [0]
        with self.assertRaises(KeyError):
            config_mod.finalize_config(cfg)

    def test_skip_resume_changed_config_and_failed_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = copy.deepcopy(self.suite)
            suite['root'] = tmp
            job = suite_mod.train_jobs(suite, ['full'])[0]
            calls = []
            def fake(command, **kwargs):
                calls.append(command)
                work = Path(job['work'])
                for name in ('latest.pth', 'iter_0010000.pth'):
                    (work / name).touch()
                saved = copy.deepcopy(job['cfg'])
                saved['meta'] = dict(config_fingerprint=job['signature'])
                (work / 'config.json').write_text(json.dumps(saved))
                return SimpleNamespace(returncode=0)
            result = suite_mod.execute_job(job, 'cpu', run_process=fake)
            self.assertEqual(result['status'], 'completed')
            result = suite_mod.execute_job(job, 'cpu', resume=True, run_process=fake)
            self.assertEqual(result['status'], 'skipped_completed')
            self.assertEqual(len(calls), 1)
            state_path = Path(job['work']) / 'suite_status.json'
            state = json.loads(state_path.read_text())
            state['status'] = 'failed'
            state_path.write_text(json.dumps(state))
            suite_mod.execute_job(job, 'cpu', resume=True, run_process=fake)
            self.assertIn('--resume-from', calls[-1])
            changed = dict(job, signature='different-config')
            with self.assertRaisesRegex(ValueError, 'Changed experiment'):
                suite_mod.execute_job(changed, 'cpu', resume=True, run_process=fake)
            with patch.object(suite_mod, 'execute_job', side_effect=[RuntimeError('one failure'), dict(status='completed')]):
                statuses = suite_mod.execute_jobs([job, job], ['cpu'])
            self.assertEqual([s['status'] for s in statuses], ['failed', 'completed'])


if __name__ == '__main__':
    unittest.main()
