from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from tests._bootstrap import load_module

attention = load_module('models.aggregation_attention')
variants = load_module('models.experiment_variants')
aggregation = load_module('models.feature_aggregation')
fusion = load_module('models.semantic_detail_fusion')


class TestExperimentVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diagonal_attention_blocks_all_other_prompts(self):
        block = attention.CrossClassAggregationBlock(8, 8, 2, 0.0, cross_class_mode='diagonal').eval()
        features, semantic = torch.randn(1, 3, 8, 2, 2), torch.randn(1, 3, 8, 2, 2)
        reference = block(features, semantic)
        changed_features, changed_semantic = features.clone(), semantic.clone()
        changed_features[:, 1:] += 100 * torch.randn_like(changed_features[:, 1:])
        changed_semantic[:, 1:] += 100 * torch.randn_like(changed_semantic[:, 1:])
        changed = block(changed_features, changed_semantic)
        for before, after in zip(reference, changed):
            torch.testing.assert_close(before[:, 0], after[:, 0])
        sum(x.sum() for x in reference).backward()
        self.assertIsNotNone(block.feature_attention.v_proj.weight.grad)
        self.assertIsNotNone(block.semantic_attention.v_proj.weight.grad)

    def test_presence_multiplies_once_in_expanded_prompt_space(self):
        semantic = torch.tensor([0., 2., -3.]).reshape(1, 3, 1, 1)
        presence = torch.tensor([0., -1., 2.]).reshape(1, 3, 1, 1)
        scores = variants.semantic_presence_logits(semantic, presence).sigmoid()
        torch.testing.assert_close(scores, semantic.sigmoid() * presence.sigmoid())
        self.assertAlmostEqual(scores[0, 0].item(), .25)
        # Two aliases must be multiplied separately before the class max.
        expected = (semantic[:, :2].sigmoid() * presence[:, :2].sigmoid()).amax(1)
        torch.testing.assert_close(scores[:, :2].amax(1), expected)

    def test_adapter_and_concat_keep_decoder_grid_and_gradients(self):
        adapter = variants.SimpleAdapter()
        x = torch.randn(1, 256, 72, 72)
        out = adapter.fuse_semantic_detail_chunk(x)
        self.assertEqual(out.shape, x.shape)
        out.square().mean().backward()
        self.assertGreater(adapter.block[0].weight.grad.abs().sum().item(), 0)
        concat = variants.ConcatFusion()
        agg = torch.randn(2, 256, 36, 36, requires_grad=True)
        out = concat(agg, x.expand(2, -1, -1, -1), x)
        self.assertEqual(out.shape, (2, 256, 72, 72))
        out.square().mean().backward()
        self.assertGreater(agg.grad.abs().sum().item(), 0)

    def test_zero_prior_keeps_visual_guidance_and_intra_bypass(self):
        kwargs = dict(clip_text_encoder=None, clip_dim=4, remoteclip_native_dim=4,
                      prompt_templates=['{}'] * 64, semantic_init='zeros', num_blocks=2,
                      num_heads=8, dropout=0, use_checkpoint=False)
        model = aggregation.RemoteCLIPGuidedAggregation(**kwargs)
        self.assertIsNone(model.remoteclip_guidance)
        self.assertIsNotNone(model.clip_spatial_guidance_proj)
        features = torch.randn(1, 2, 256, 72, 72)
        mids = [torch.randn(1, 4, 36, 36, requires_grad=True) for _ in range(2)]
        out = model(features, torch.randn(1, 4, 36, 36), ['car', 'tree'], mids, (7, 15), features[:, 0])
        out['aggregated_features'].square().mean().backward()
        self.assertGreater(sum(x.grad.abs().sum().item() for x in mids), 0)
        model = aggregation.RemoteCLIPGuidedAggregation(**kwargs, intra_class_enabled=False)
        self.assertEqual(len(model.blocks), 1)
        self.assertIsNone(model.clip_spatial_guidance_proj)

    def test_single_orientation_keeps_parameters_and_uses_one_residual(self):
        sdf = fusion.SemanticDetailFusion(orientations=[0])
        full = fusion.SemanticDetailFusion()
        self.assertEqual({k: p.shape for k, p in sdf.state_dict().items()},
                         {k: p.shape for k, p in full.state_dict().items()})
        x = torch.randn(1, 3, 8, 8)
        block = torch.nn.Conv2d(3, 3, 3, padding=1)
        torch.testing.assert_close(fusion.aggregate_orientations(x, block, [0]), x + block(x))

    def test_feature_only_blocks_keep_semantics_live_and_both_feature_windows(self):
        constructors = [
            lambda **kw: attention.CrossClassAggregationBlock(8, 8, 2, 0.0, **kw),
            lambda **kw: attention.IntraClassAggregationBlock(8, 8, 2, 2, 1, 0.0, 4, **kw),
        ]
        for factory in constructors:
            block = factory(update_streams='feature_only')
            features = torch.randn(1, 3, 8, 4, 4, requires_grad=True)
            semantic = torch.randn_like(features, requires_grad=True)
            calls = []
            if isinstance(block, attention.IntraClassAggregationBlock):
                self.assertIsNone(block.semantic_regular)
                self.assertIsNone(block.semantic_shifted)
                for name in ('feature_regular', 'feature_shifted'):
                    getattr(block, name).register_forward_hook(lambda module, args, out, name=name: calls.append(name))
                result, state = block(features, semantic, None, torch.randn(1, 3, 4, 4, 4))
                self.assertEqual(calls, ['feature_regular', 'feature_shifted'])
            else:
                self.assertIsNone(block.semantic_attention)
                result, state = block(features, semantic)
            self.assertIs(state, semantic)
            self.assertFalse(any(name.startswith('semantic_') for name, _ in block.named_parameters()))
            result.square().mean().backward()
            self.assertGreater(semantic.grad.abs().sum().item(), 0)
            self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_feature_only_prior_remains_trainable_through_checkpointed_blocks(self):
        class ToyText(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.templates = torch.nn.Parameter(torch.randn(2, 64, 4))
            def encode_prompt_templates(self, class_names, device, **kwargs):
                return self.templates[:len(class_names)].to(device)
        text = ToyText()
        model = aggregation.RemoteCLIPGuidedAggregation(
            clip_text_encoder=text, clip_dim=4, remoteclip_native_dim=4,
            prompt_templates=['{}'] * 64, update_streams='feature_only',
            num_blocks=4, num_heads=8, dropout=0, use_checkpoint=True,
        ).train()
        self.assertIsNone(model.clip_spatial_guidance_proj)
        self.assertEqual(len(model.blocks), 4)
        self.assertEqual([type(b) for b in model.blocks], [
            attention.CrossClassAggregationBlock, attention.IntraClassAggregationBlock,
            attention.CrossClassAggregationBlock, attention.IntraClassAggregationBlock,
        ])
        dense = torch.randn(1, 4, 36, 36, requires_grad=True)
        fpn = torch.randn(1, 256, 72, 72, requires_grad=True)
        # No CLIP middle features: that branch is unused in this ablation.
        out = model(torch.randn(1, 2, 256, 72, 72), dense, ['car', 'tree'], sam3_fpn_72=fpn)
        self.assertIs(out['semantic_state_36'], out['clip_guidance_36'])
        out['aggregated_features'][:, :, 0].mean().backward()
        for parameter in (text.templates, dense, fpn, model.remoteclip_guidance.template_projection[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_default_both_matches_explicit_both_initialization_and_outputs(self):
        for cls, kwargs in [
            (attention.CrossClassAggregationBlock, dict(feature_dim=8, semantic_dim=8, num_heads=2, dropout=0)),
            (attention.IntraClassAggregationBlock, dict(feature_dim=8, semantic_dim=8, num_heads=2,
                 window_size=2, shift_size=1, dropout=0, guidance_dim=4)),
        ]:
            torch.manual_seed(18)
            default = cls(**kwargs).eval()
            torch.manual_seed(18)
            explicit = cls(**kwargs, update_streams='both').eval()
            self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
            for key, value in default.state_dict().items():
                torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
            args = [torch.randn(1, 3, 8, 4, 4), torch.randn(1, 3, 8, 4, 4)]
            if cls is attention.IntraClassAggregationBlock:
                args.extend([torch.randn(1, 3, 4, 4, 4), torch.randn(1, 3, 4, 4, 4)])
            for a, b in zip(default(*args), explicit(*args)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_original_decoder_and_instance_masks_run_on_cpu(self):
        builder = load_module('model_builder').ReCASAM3ModelBuilder
        decoder = builder._create_instance_decoder().eval()
        prompt = torch.randn(3, 1, 256)
        hs, _, presence, _ = decoder(
            tgt=decoder.query_embed.weight[:, None], memory=torch.randn(4, 1, 256),
            pos=torch.randn(4, 1, 256), spatial_shapes=torch.tensor([[2, 2]]),
            valid_ratios=torch.ones(1, 1, 2), level_start_index=torch.tensor([0]),
            memory_text=prompt, text_attention_mask=torch.zeros(1, 3, dtype=torch.bool), apply_dac=False,
        )
        self.assertEqual(hs.shape, (6, 200, 1, 256))
        self.assertEqual(presence.numel(), 6)
        queries = hs.transpose(1, 2)
        scores = builder._create_instance_scorer()(queries, prompt, torch.zeros(1, 3, dtype=torch.bool))
        self.assertEqual(scores.shape, (6, 1, 200, 1))
        head = builder._create_segmentation_head(with_instance=True).eval()
        # Keep the real instance projection and query mask predictor. Replace
        # only pixel upsampling to keep this CPU check small.
        class TinyPixels(torch.nn.Module):
            def forward(self, features):
                return features[0]
        head.pixel_decoder = TinyPixels()
        outputs = head([torch.randn(1, 256, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 256, 2, 2)],
                       torch.tensor([0]), torch.randn(4, 1, 256), obj_queries=queries)
        self.assertEqual(outputs['pred_masks'].shape[1], 200)
        self.assertEqual(outputs['semantic_seg'].shape[1], 1)


if __name__ == '__main__':
    unittest.main()
