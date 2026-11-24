"""Behavior checks for the paper variants, label protocols and single-job CLI."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tests._bootstrap import REPO_ROOT, load_module

attention = load_module('models.aggregation_attention')
aggregation = load_module('models.feature_aggregation')
core = load_module('models.sam3_image')
config = load_module('engine.config')
suites = load_module('utils.experiment_suite')


class PaperAblationPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diagonal_mask_blocks_every_other_prompt_including_aliases(self):
        torch.manual_seed(16)
        layer = attention.CrossClassAttention(feature_dim=4, semantic_dim=4, num_heads=1,
                                               dropout=0, cross_class_mode='self').eval()
        features = torch.randn(1, 3, 4, 2, 2)
        semantics = torch.randn_like(features)
        expected = layer(features, semantics)
        changed = features.clone(); changed[:, 1:] += 100
        guidance = semantics.clone(); guidance[:, 1:] *= -20
        torch.testing.assert_close(expected[:, 0], layer(changed, guidance)[:, 0])
        # With one allowed key, the result is the projected value of that prompt.
        values = features.permute(0, 3, 4, 1, 2)
        manual = layer.out_proj(layer.v_proj(values)).permute(0, 3, 4, 1, 2)
        torch.testing.assert_close(expected, manual)

    def test_no_intra_has_no_spatial_parameters(self):
        block = attention.AggregationAttentionBlock(feature_dim=4, semantic_dim=4, num_heads=1,
                                                    dropout=0, intra_class_enabled=False)
        self.assertIsNone(block.window_update)
        x = torch.randn(1, 2, 4, 3, 3, requires_grad=True)
        semantic = torch.randn_like(x)
        expected = x + block.cross_attention(attention.apply_layer_norm_bcdhw(x, block.cross_norm_feature),
                                             attention.apply_layer_norm_bcdhw(semantic, block.cross_norm_semantic))
        torch.testing.assert_close(block(x, (semantic,), ()), expected)
        expected.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(any('window' in n for n, _ in block.named_parameters()))

    def test_single_scale_is_one_window_update_without_fusion_parameters(self):
        block = attention.AggregationAttentionBlock(feature_dim=4, semantic_dim=4, num_heads=1,
                                                    guidance_dim=4, dropout=0, window_scales=[36])
        x = torch.randn(1, 2, 4, 36, 36)
        s = torch.randn_like(x); g = torch.randn_like(x)
        self.assertIsNone(block.window_update.scale_logits)
        calls = []
        window = block.window_update.windows[0]
        with window.register_forward_hook(lambda m, a, o: calls.append(a[0].shape[-1])):
            actual = block(x, (s,), (g,))
        self.assertEqual(calls, [36])
        cross = x + block.cross_attention(attention.apply_layer_norm_bcdhw(x, block.cross_norm_feature),
                                          attention.apply_layer_norm_bcdhw(s, block.cross_norm_semantic))
        torch.testing.assert_close(actual, cross + window(cross, s, g))
        actual.mean().backward()
        self.assertIsNotNone(window.attn.v_proj.weight.grad)

    def test_guidance_removal_has_no_remoteclip_modules_and_keeps_fpn_gradient(self):
        module = aggregation.RemoteCLIPGuidedAggregation(
            clip_text_encoder=None, clip_dim=None, use_remoteclip_guidance=False,
            window_scales=[36], num_blocks=1, dropout=0, use_checkpoint=True).train()
        self.assertIsNone(module.remoteclip_guidance)
        self.assertEqual(module.blocks[0].cross_attention.q_proj.in_features, 256)
        self.assertEqual(module.blocks[0].window_update.windows[0].attn.q_proj.in_features, 512)
        x = torch.randn(1, 2, 256, 72, 72, requires_grad=True)
        fpn = torch.randn(1, 256, 72, 72, requires_grad=True)
        out = module(x, None, ['car', 'tree'], sam3_fpn_72=fpn)
        self.assertEqual(set(out), {'aggregated_features'})
        out['aggregated_features'][:, :, 0].mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(fpn.grad.abs().sum().item(), 0)
        transformer = torch.nn.Identity(); transformer.d_model = 256
        model = core.Sam3Image(backbone=torch.nn.Identity(), transformer=transformer,
                              input_geometry_encoder=torch.nn.Identity(),
                              segmentation_head=torch.nn.Identity(), use_remoteclip_guidance=False)
        self.assertIsNone(model.clip_text_encoder)
        self.assertIsNone(model.clip_image_encoder)
        self.assertFalse(any('remoteclip' in n or 'clip_image' in n for n, _ in model.named_parameters()))

    def test_final_dense_encoder_preserves_gradient_and_collects_no_middle_features(self):
        encoders = load_module('models.openclip_image_encoder')
        visual = torch.nn.Module(); visual.width = visual.output_dim = 4
        visual.conv1 = torch.nn.Conv2d(3, 4, 1)
        visual.class_embedding = torch.nn.Parameter(torch.randn(4))
        visual.positional_embedding = torch.nn.Parameter(torch.randn(17, 4))
        visual.patch_dropout = visual.ln_pre = visual.ln_post = torch.nn.Identity()
        visual.proj = torch.nn.Parameter(torch.eye(4))
        visual.transformer = torch.nn.Module()
        final = torch.nn.Module(); final.ln_1 = final.ln_2 = torch.nn.Identity()
        final.attn = torch.nn.MultiheadAttention(4, 1)
        final.mlp = torch.nn.Linear(4, 4)
        visual.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), final])
        encoder = encoders.OpenCLIPImageEncoder(visual, image_size=36)
        encoder.set_enable_grad(True)
        visual.requires_grad_(True)
        out = encoder.encode_raw_images([torch.rand(3, 36, 36)], torch.device('cpu'))
        self.assertEqual(set(out), {'feat_map'})
        self.assertEqual(out['feat_map'].shape, (1, 4, 36, 36))
        out['feat_map'].square().mean().backward()
        self.assertIsNotNone(visual.conv1.weight.grad)
        self.assertFalse(hasattr(encoder, 'intermediate_layers'))

    def test_isaid_raw_labels_have_two_explicit_protocols(self):
        dataset_type = load_module('data.dataset').OVSemanticSegDataset
        labels = torch.tensor([[0, 1, 15, 255]])
        configs = []
        for target in ('isaid_15_ignbg', 'isaid_16_bg'):
            cfg, _ = config.build_effective_config(REPO_ROOT / f'configs/test/{target}.py')
            configs.append(cfg)
            data = cfg['val_dataloader']['dataset']
            instance = dataset_type.__new__(dataset_type)
            instance.ignore_index = 255; instance.reduce_zero_label = data['reduce_zero_label']
            mapped = instance._apply_reduce_zero_label(labels)
            self.assertEqual(mapped.tolist(), [[255, 0, 14, 255]] if '15_' in target else [[0, 1, 15, 255]])
            self.assertEqual(cfg['evaluation']['dataset'], 'isaid')
            self.assertEqual(len(data['classes']), 15 if '15_' in target else 16)
        self.assertNotEqual(configs[0]['evaluation']['protocol_id'], configs[1]['evaluation']['protocol_id'])
        self.assertFalse(configs[0]['val_dataloader']['dataset']['background_cfg']['enabled'])
        self.assertTrue(configs[1]['val_dataloader']['dataset']['background_cfg']['enabled'])

    def test_single_variant_single_target_cli_restores_checkpoint_architecture(self):
        suite = suites.load_suite('configs/suites/paper_validate.py')
        full_jobs = suites.train_jobs(suite)
        self.assertEqual(len(full_jobs), 10)
        self.assertEqual(len({j['signature'] for j in full_jobs}), 10)
        with tempfile.TemporaryDirectory() as tmp:
            source = next(j['cfg'] for j in full_jobs if j['variant'] == 'single_scale')
            weight = Path(tmp) / 'best.pth'; weight.write_bytes(b'planning-only fixture')
            source['meta'] = dict(config_fingerprint=config.config_fingerprint(source))
            (weight.parent / 'config.json').write_text(json.dumps(source))
            args = [sys.executable, 'tools/validate_experiments.py', 'configs/suites/paper_validate.py',
                    '--only', 'single_scale', '--checkpoint', str(weight), '--seed', '42',
                    '--datasets', 'isaid_15_ignbg', 'isaid_16_bg', '--dry-run']
            result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            jobs = json.loads(result.stdout)['jobs']
            self.assertEqual(len(jobs), 2)
            self.assertEqual(len({j['checkpoint_sha256'] for j in jobs}), 1)
            self.assertEqual(len({j['eval_protocol_id'] for j in jobs}), 2)
            self.assertEqual(len({j['work'] for j in jobs}), 2)
            chosen = suites.select_jobs(suite, ['single_scale'], ['loveda'], str(weight), 42)
            planned, pending = suites.validation_jobs(chosen, ['single_scale'])
            self.assertFalse(pending)
            self.assertEqual(len(planned), 1)
            self.assertEqual(planned[0]['cfg']['model']['aggregation']['window_scales'], [36])
            with self.assertRaises(ValueError):suites.select_jobs(suite, ['full'], ['typo'])
            with self.assertRaises(ValueError):suites.select_jobs(suite, None, checkpoint=str(weight))
            source['experiment']['variant_id'] = 'full'
            (weight.parent / 'config.json').write_text(json.dumps(source))
            with self.assertRaises(ValueError):suites.validation_jobs(chosen, ['single_scale'])


if __name__ == '__main__':
    unittest.main()
