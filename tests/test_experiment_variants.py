from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from tests._bootstrap import load_module

attention = load_module('models.aggregation_attention')
variants = load_module('models.experiment_variants')
aggregation = load_module('models.feature_aggregation')
fusion = load_module('models.semantic_detail_fusion')


class TestAggregationDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_presence_multiplies_once_in_expanded_prompt_space(self):
        semantic = torch.tensor([0., 2., -3.]).reshape(1, 3, 1, 1)
        presence = torch.tensor([0., -1., 2.]).reshape(1, 3, 1, 1)
        scores = variants.semantic_presence_logits(semantic, presence).sigmoid()
        torch.testing.assert_close(scores, semantic.sigmoid() * presence.sigmoid())
        self.assertAlmostEqual(scores[0, 0].item(), .25)
        # Two aliases must be multiplied separately before the class max.
        expected = (semantic[:, :2].sigmoid() * presence[:, :2].sigmoid()).amax(1)
        torch.testing.assert_close(scores[:, :2].amax(1), expected)

    def test_complete_block_order_residual_inputs_and_semantic_gradients(self):
        block = attention.AggregationAttentionBlock(
            feature_dim=8, semantic_dim=8, num_heads=2,
            window_size=2, shift_size=1, dropout=0, guidance_dim=4,
        ).eval()
        features = torch.randn(1, 3, 8, 4, 4, requires_grad=True)
        semantic = torch.randn_like(features, requires_grad=True)
        original_semantic = semantic.detach().clone()
        guidance = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
        calls = []
        for name in ('cross_attention', 'feature_regular', 'feature_shifted'):
            getattr(block, name).register_forward_hook(
                lambda module, args, out, name=name: calls.append(name)
            )
        result = block(features, semantic, guidance)
        self.assertEqual(calls, ['cross_attention', 'feature_regular', 'feature_shifted'])
        cross = features + block.cross_attention(
            attention.apply_layer_norm_bcdhw(features, block.cross_norm_feature),
            attention.apply_layer_norm_bcdhw(semantic, block.cross_norm_semantic),
        )
        regular = cross + block.feature_regular(cross, semantic, guidance)
        expected = regular + block.feature_shifted(regular, semantic, guidance)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(semantic, original_semantic)
        self.assertFalse(any('semantic_attention' in name for name, _ in block.named_parameters()))
        result.square().mean().backward()
        for tensor in (features, semantic, guidance):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        # A class receives information from other prompts at the same location.
        changed = features.detach().clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 10
        self.assertFalse(torch.allclose(result[:, 0], block(changed, semantic, guidance)[:, 0]))

    def test_fixed_prior_remains_trainable_through_checkpointed_complete_blocks(self):
        class ToyText(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.templates = torch.nn.Parameter(torch.randn(2, 64, 4))
            def encode_prompt_templates(self, class_names, device, **kwargs):
                return self.templates[:len(class_names)].to(device)
        torch.manual_seed(14)
        text = ToyText()
        model = aggregation.RemoteCLIPGuidedAggregation(
            clip_text_encoder=text, clip_dim=4,
            prompt_templates=['{}'] * 64, num_blocks=2,
            num_heads=8, window_size=6, shift_size=3, dropout=0, use_checkpoint=True,
        ).train()
        self.assertEqual(len(model.blocks), 2)
        self.assertTrue(all(isinstance(b, attention.AggregationAttentionBlock) for b in model.blocks))
        self.assertFalse(hasattr(model, 'clip_spatial_guidance_proj'))
        dense = torch.randn(1, 4, 36, 36, requires_grad=True)
        fpn = torch.randn(1, 256, 72, 72, requires_grad=True)
        out = model(torch.randn(1, 2, 256, 72, 72), dense, ['car', 'tree'], sam3_fpn_72=fpn)
        self.assertIs(out['semantic_state_36'], out['clip_guidance_36'])
        out['aggregated_features'][:, :, 0].mean().backward()
        for parameter in (text.templates, dense, fpn, model.remoteclip_guidance.template_projection[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Recomputing checkpointed blocks must produce the same outputs and gradients.
        plain_text = ToyText()
        plain_text.load_state_dict(text.state_dict())
        plain = aggregation.RemoteCLIPGuidedAggregation(
            clip_text_encoder=plain_text, clip_dim=4, prompt_templates=['{}'] * 64,
            num_blocks=2, num_heads=8, window_size=6, shift_size=3, dropout=0, use_checkpoint=False,
        ).train()
        plain.load_state_dict(model.state_dict())
        dense_plain = dense.detach().clone().requires_grad_()
        fpn_plain = fpn.detach().clone().requires_grad_()
        encoder = torch.randn(1, 2, 256, 72, 72)
        model.zero_grad(set_to_none=True)
        dense.grad = fpn.grad = None
        checkpointed = model(encoder, dense, ['car', 'tree'], sam3_fpn_72=fpn)['aggregated_features']
        reference = plain(encoder, dense_plain, ['car', 'tree'], sam3_fpn_72=fpn_plain)['aggregated_features']
        torch.testing.assert_close(checkpointed, reference)
        checkpointed[:, :, 0].mean().backward()
        reference[:, :, 0].mean().backward()
        torch.testing.assert_close(dense.grad, dense_plain.grad)
        torch.testing.assert_close(fpn.grad, fpn_plain.grad)
        for (name, parameter), (_, expected) in zip(model.named_parameters(), plain.named_parameters()):
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, expected.grad, msg=name)

    def test_fusion_executes_each_branch_once_and_propagates_gradients(self):
        module = fusion.SemanticDetailFusion(use_checkpoint=False).eval()
        calls = []
        branch_inputs, branch_outputs = {}, {}
        for name in ('semantic_block', 'detail_block'):
            def capture(mod, args, output, name=name):
                calls.append(name)
                branch_inputs[name], branch_outputs[name] = args[0], output
            getattr(module, name).register_forward_hook(capture)
        agg = torch.randn(2, 256, 36, 36, requires_grad=True)
        encoder = torch.randn(2, 256, 72, 72, requires_grad=True)
        fpn = torch.randn(1, 256, 72, 72, requires_grad=True)
        output = module(agg, encoder, fpn)
        self.assertEqual(calls, ['semantic_block', 'detail_block'])
        self.assertEqual(output.shape, encoder.shape)
        expected = module.fusion_out_proj(
            module.semantic_out_proj(branch_inputs['semantic_block'] + branch_outputs['semantic_block'])
            + module.detail_out_proj(branch_inputs['detail_block'] + branch_outputs['detail_block'])
        )
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        for tensor in (agg, encoder, fpn):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

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
