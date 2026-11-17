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

    def test_all_twelve_variants_share_the_training_budget(self):
        jobs = suite_mod.train_jobs(self.suite)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job['signature'] for job in jobs}), 12)
        configs = {j['variant']: j['cfg'] for j in jobs}
        for cfg in configs.values():
            self.assertEqual(cfg['train_dataloader']['batch_size'], 4)
            self.assertEqual(cfg['train_cfg']['max_iters'], 10000)
            self.assertEqual(cfg['optimizer']['lr'], 2e-6)
            self.assertEqual(cfg['param_scheduler'][0]['end'], 500)
            self.assertEqual(cfg['param_scheduler'][1]['T_max'], 9500)
            self.assertEqual(cfg['param_scheduler'][1]['min_lr_ratio'], .01)
            self.assertEqual(cfg['logging']['wandb']['mode'], 'offline')
        self.assertEqual(configs['encoder_only_update']['model']['aggregation']['update_streams'], 'feature_only')
        self.assertEqual(configs['simple_adapter']['model']['pipeline'], 'adapter')
        self.assertEqual(configs['no_score_prior']['model']['aggregation']['semantic_init'], 'zeros')
        self.assertEqual(configs['no_distillation']['loss']['reference_distill_weight'], 0)
        combinations = [(configs[v]['model']['freeze']['remoteclip_text_finetune'],
                         configs[v]['model']['freeze']['remoteclip_image_finetune'])
                        for v in ('full', 'freeze_both', 'freeze_text', 'freeze_image')]
        self.assertEqual(len(set(combinations)), 4)

    def test_manual_selection_restores_architecture_and_all_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = suite_mod.load_suite('configs/suites/paper_validate.py')
            suite['baselines'] = []
            source = suite_mod.train_jobs(self.suite, ['simple_adapter'])[0]['cfg']
            checkpoint = Path(tmp) / 'iter_0003000.pth'
            checkpoint.write_bytes(b'synthetic checkpoint; no model execution')
            source['meta'] = dict(config_fingerprint=config_mod.config_fingerprint(source))
            (checkpoint.parent / 'config.json').write_text(json.dumps(source))
            suite['selections'] = [dict(variant='simple_adapter', seed=42, checkpoint=str(checkpoint))]
            jobs, pending = suite_mod.validation_jobs(suite)
            self.assertEqual(pending, [])
            self.assertEqual(len(jobs), 8)
            self.assertEqual({j['checkpoint_sha256'] for j in jobs}, {suite_mod.sha256_file(checkpoint)})
            for job in jobs:
                self.assertEqual(job['cfg']['model']['pipeline'], 'adapter')
                self.assertFalse(job['cfg']['model']['remoteclip']['enabled'])
                self.assertEqual(job['cfg']['training']['dataset'], 'dlrsd')
                self.assertEqual(job['cfg']['evaluation']['dataset'], job['dataset'])
            suite['selections'][0]['variant'] = 'full'
            with self.assertRaisesRegex(ValueError, 'identity'):
                suite_mod.validation_jobs(suite)

    def test_no_automatic_checkpoint_selection(self):
        suite = suite_mod.load_suite('configs/suites/paper_validate.py')
        jobs, pending = suite_mod.validation_jobs(suite, dry_run=True)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len(pending), 12)
        self.assertTrue(all(j['pretrained'] for j in jobs))

    def test_encoder_only_selection_restores_three_ablation_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = suite_mod.load_suite('configs/suites/paper_validate.py')
            suite['baselines'] = []
            source = suite_mod.train_jobs(self.suite, ['encoder_only_update'])[0]['cfg']
            checkpoint = Path(tmp) / 'iter_0005000.pth'
            checkpoint.write_bytes(b'synthetic metadata only')
            source['meta'] = dict(config_fingerprint=config_mod.config_fingerprint(source))
            (checkpoint.parent / 'config.json').write_text(json.dumps(source))
            suite['selections'] = [dict(variant='encoder_only_update', seed=42, checkpoint=str(checkpoint))]
            jobs, pending = suite_mod.validation_jobs(suite)
            self.assertEqual(pending, [])
            self.assertEqual({job['dataset'] for job in jobs}, {'loveda', 'potsdam', 'uavid'})
            for job in jobs:
                self.assertEqual(job['cfg']['model']['aggregation']['update_streams'], 'feature_only')
                self.assertEqual(job['cfg']['experiment']['variant_id'], 'encoder_only_update')

    def test_stream_choice_validation_and_experiment_identity(self):
        cfg = suite_mod.train_jobs(self.suite, ['full'])[0]['cfg']
        fingerprint = config_mod.config_fingerprint(cfg)
        previous = copy.deepcopy(cfg)
        previous['model']['aggregation'].pop('update_streams')
        self.assertEqual(fingerprint, config_mod.config_fingerprint(previous))
        cfg['model']['aggregation']['update_streams'] = 'feature_only'
        self.assertNotEqual(fingerprint, config_mod.config_fingerprint(cfg))
        cfg['model']['aggregation']['update_streams'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'update_streams'):
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
