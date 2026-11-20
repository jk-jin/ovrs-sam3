from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tests._bootstrap import load_module

encoder_mod = load_module('models.openclip_image_encoder')
core_mod = load_module('models.sam3_image')
wrapper_mod = load_module('models.reca_sam3')
config_mod = load_module('engine.config')
aggregation_mod = load_module('models.feature_aggregation')


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = self.output_dim = 1
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([nn.Identity()])


class TestRemoteCLIPRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_encoder(self):
        return encoder_mod.OpenCLIPImageEncoder(
            TinyVisual(), intermediate_layers=[0], image_size=4,
        ).eval()

    def test_encoder_preserves_rotated_dense_and_middle_features_with_gradients(self):
        encoder = self.make_encoder()
        raw = [torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48 + i for i in range(4)]
        images = encoder.preprocess_raw_images(raw, torch.device('cpu'))
        gain = nn.Parameter(torch.tensor(2.0))
        position = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        seen = []

        def encode(x, keep_intermediate_grad):
            seen.append(x)
            self.assertTrue(keep_intermediate_grad)
            # Position-dependent output detects premature inverse rotation.
            dense = x[:, :1] * gain + position
            return dict(feat_map=dense, mid_features=[dense * 3], mid_layer_indices=(0,))

        with patch.object(encoder_mod.random, 'randrange', side_effect=[0, 1, 2, 3]), \
             patch.object(encoder, 'encode_image_with_intermediate', side_effect=encode):
            output = encoder.encode_raw_images(raw, torch.device('cpu'), True, random_rotate=True)
        self.assertEqual(len(seen), 1)
        for i in range(4):
            torch.testing.assert_close(seen[0][i], torch.rot90(images[i], i, (-2, -1)))
            expected = torch.rot90(images[i, :1], i, (-2, -1)) * gain + position[0]
            torch.testing.assert_close(output['feat_map'][i], expected)
            torch.testing.assert_close(output['mid_features'][0][i], expected * 3)
        self.assertEqual(output['mid_layer_indices'], (0,))
        self.assertEqual(output['rotation_turns'], (0, 1, 2, 3))
        (output['feat_map'].square().mean() + output['mid_features'][0].square().mean()).backward()
        self.assertGreater(gain.grad.abs().item(), 0)
        self.assertTrue(output['feat_map'].is_contiguous())
        self.assertTrue(output['mid_features'][0].is_contiguous())

    def test_disabled_path_preserves_output_identity_and_rng(self):
        encoder = self.make_encoder()
        expected = dict(feat_map=torch.zeros(1, 1, 4, 4), mid_features=[], mid_layer_indices=())
        state = random.getstate()
        with patch.object(encoder, 'encode_image_with_intermediate', return_value=expected) as encode, \
             patch.object(encoder_mod.random, 'randrange', side_effect=AssertionError('unexpected RNG')):
            result = encoder.encode_raw_images([torch.zeros(3, 4, 4)], torch.device('cpu'))
        self.assertIs(result, expected)
        self.assertEqual(random.getstate(), state)
        encode.assert_called_once()

    def test_training_gate_uses_outer_model_even_when_clip_is_eval_or_frozen(self):
        class RecordingEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def encode_raw_images(self, **kwargs):
                self.calls.append((kwargs['random_rotate'], self.training))
                return dict(feat_map=torch.zeros(1, 1, 36, 36), mid_features=[],
                            mid_layer_indices=(), rotation_turns=(1,) if kwargs['random_rotate'] else None)
            def has_trainable_params(self):
                return False

        core = object.__new__(core_mod.Sam3Image)
        nn.Module.__init__(core)
        core.clip_image_encoder = RecordingEncoder()
        core.clip_text_encoder = None
        core.feature_aggregation = None
        for name in ('backbone', 'transformer', 'geometry_encoder', 'segmentation_head'):
            setattr(core, name, nn.Identity())
        wrapper = wrapper_mod.ReCASAM3(core, nn.Identity(), 'semantic')
        batch = SimpleNamespace(raw_images=[torch.zeros(3, 4, 4)])
        for enabled, training, expected in [(True, True, True), (True, False, False), (False, True, False)]:
            core.remoteclip_train_random_rotate = enabled
            wrapper.train(training)
            with torch.no_grad():
                cache = core._build_remoteclip_image_cache(batch, torch.device('cpu'))
            self.assertEqual(core.clip_image_encoder.calls[-1], (expected, False))
            self.assertEqual(cache['remoteclip_rotation_turns'], (1,) if expected else None)

    def test_embedding_convolutions_precede_inverse_rotation_and_keep_gradients(self):
        class ToyText(nn.Module):
            def __init__(self):
                super().__init__()
                self.templates = nn.Parameter(torch.randn(2, 64, 4))
            def encode_prompt_templates(self, class_names, device, **kwargs):
                return self.templates[:len(class_names)].to(device)

        class RecordingBlock(nn.Module):
            def forward(self, features, semantic, guidance):
                self.semantic = semantic
                return features + semantic

        torch.manual_seed(3)
        model = aggregation_mod.RemoteCLIPGuidedAggregation(
            ToyText(), clip_dim=4, prompt_templates=['{}'] * 64,
            num_blocks=1, use_checkpoint=False, dropout=0,
        ).train()
        model.blocks = nn.ModuleList([RecordingBlock()])
        dense = torch.randn(4, 4, 36, 36, requires_grad=True)
        turns = (0, 1, 2, 3)
        names = ['car', 'tree']
        # The reference runs the actual embedding, including both 3x3 convs,
        # in the rotated coordinates. Spatial kernels need not be equivariant.
        embedding, scores, text = model.remoteclip_guidance(names, dense)
        expected = aggregation_mod.realign_rotated_batch(embedding, turns)
        early_dense = aggregation_mod.realign_rotated_batch(dense, turns)
        early_embedding = model.remoteclip_guidance(names, early_dense)[0]
        self.assertFalse(torch.allclose(expected, early_embedding))
        out = model(
            torch.zeros(4, 2, 256, 72, 72), dense, names,
            sam3_fpn_72=torch.zeros(4, 256, 72, 72),
            remoteclip_rotation_turns=turns,
        )
        torch.testing.assert_close(out['clip_guidance_36'], expected)
        torch.testing.assert_close(out['clip_template_scores_36'],
                                   aggregation_mod.realign_rotated_batch(scores, turns))
        torch.testing.assert_close(out['remoteclip_template_text_features'], text)
        self.assertIs(model.blocks[0].semantic, out['clip_guidance_36'])
        self.assertIs(out['semantic_state_36'], out['clip_guidance_36'])
        out['clip_guidance_36'].square().mean().backward()
        for parameter in (dense, model.remoteclip_guidance.template_projection[0].weight,
                          model.remoteclip_guidance.clip_text_encoder.templates):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_cache_routes_angles_to_aggregation_and_realigns_debug_features(self):
        class RecordingAggregation(nn.Module):
            def forward(self, **kwargs):
                self.kwargs = kwargs
                spatial = torch.zeros(2, 1, 1, 36, 36)
                return dict(aggregated_features=spatial, semantic_state_36=spatial,
                            clip_guidance_36=spatial, clip_template_scores_36=spatial,
                            remoteclip_template_text_features=torch.zeros(1, 64, 1))

        core = object.__new__(core_mod.Sam3Image)
        nn.Module.__init__(core)
        core.feature_aggregation = RecordingAggregation()
        dense = torch.randn(2, 1, 36, 36)
        middle = torch.randn_like(dense)
        cache = dict(prompt_conditioned_features_72=torch.zeros(2, 1, 256, 72, 72),
                     remoteclip_feat_map=dense, remoteclip_rotation_turns=(1, 3),
                     prompt_names=['car'], backbone_fpn=[None, None, torch.zeros(2, 256, 72, 72)],
                     clip_spatial_guidance_features=[middle])
        out = core.run_aggregation_lowres_from_cache(
            cache, SimpleNamespace(find_text_batch=['car']), return_debug=True,
        )
        self.assertEqual(core.feature_aggregation.kwargs['remoteclip_rotation_turns'], (1, 3))
        self.assertIs(core.feature_aggregation.kwargs['remoteclip_feat_map'], dense)
        torch.testing.assert_close(out['clip_spatial_guidance_features'][0],
                                   aggregation_mod.realign_rotated_batch(middle, (1, 3)))
        # The combined decoding wrapper must preserve the aligned debug output.
        core.eval()
        cache['chunk_prompt_counts'] = [1]
        with patch.object(core, 'decode_aggregation_chunk_from_cache',
                          return_value=dict(final_logits=torch.zeros(2, 1, 288, 288))):
            decoded = core.run_aggregation_from_cache(
                cache, SimpleNamespace(find_text_batch=['car']), return_debug=True,
            )
        torch.testing.assert_close(decoded['clip_spatial_guidance_features'][0],
                                   out['clip_spatial_guidance_features'][0])

    def test_switch_validation_and_config_fingerprint(self):
        cfg, original = config_mod.build_effective_config('configs/experiments/main/dlrsd_loveda_full.py')
        self.assertFalse(cfg['model']['remoteclip']['train_random_rotate'])
        cfg['model']['remoteclip']['train_random_rotate'] = True
        finalized = config_mod.finalize_config(cfg)
        self.assertNotEqual(original, config_mod.config_fingerprint(finalized))
        for value in ('true', 1, None):
            cfg['model']['remoteclip']['train_random_rotate'] = value
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'train_random_rotate'):
                config_mod.finalize_config(cfg)


if __name__ == '__main__':
    unittest.main()
