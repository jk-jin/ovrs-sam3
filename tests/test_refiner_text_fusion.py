"""CPU regression tests; no SAM3/RemoteCLIP weights or datasets required.

Run: python -m unittest discover -s tests -p 'test_refiner_text_fusion.py'
"""
import copy
import unittest

import torch
from torch import nn

from models.encoder_refiner import ClassConditionedEncoderRefiner


class TemplateEncoder(nn.Module):
    """Small trainable stand-in for already projected template embeddings."""

    def __init__(self):
        super().__init__()
        self.templates = nn.Parameter(torch.randn(3, 64, 12))

    def has_trainable_params(self):
        return True

    def encode_prompt_templates(self, **kwargs):
        return self.templates


class RefinerTextFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.encoder = TemplateEncoder()
        self.model = ClassConditionedEncoderRefiner(
            clip_text_encoder=self.encoder,
            hidden_dim=8,
            clip_dim=12,
            score_embed_dim=8,
            num_heads=2,
            fusion_layers=2,
            dropout=0.0,
            use_checkpoint=False,
            prompt_templates=['an image of {}'] * 64,
        )
        self.inputs = dict(
            encoder_features_72=torch.randn(2, 3, 8, 72, 72),
            clip_image_feat_map=torch.randn(2, 12, 36, 36),
            sam_text_mean=torch.randn(2, 3, 8),
            class_names=['road', 'building', 'car'],
        )

    def test_shared_guidance_and_direct_text_gradients(self):
        seen = []
        fusion_calls = []
        handles = [self.model.text_fusion.register_forward_hook(
            lambda *args: fusion_calls.append(True)
        )]
        for layer in self.model.layers:
            handles.append(layer.class_attn.register_forward_pre_hook(
                lambda module, args, kwargs: seen.append(kwargs['fused_text']),
                with_kwargs=True,
            ))
        # Isolate the new text path: gradients must reach the text encoder
        # even when score embedding carries no gradient into that encoder.
        handles.append(self.model.clip_score_embed.register_forward_hook(
            lambda module, args, output: (
                output[0].detach(), output[1].detach(), output[2]
            )
        ))
        output = self.model(**self.inputs)['refiner_features_36']
        for handle in handles:
            handle.remove()
        self.assertEqual(len(fusion_calls), 1)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])
        self.assertEqual(tuple(seen[0].shape), (2, 3, 8))
        self.assertFalse(any('class_norm_text' in key for key in self.model.state_dict()))
        with torch.no_grad():
            mean = self.encoder.templates.mean(1).unsqueeze(0).expand(2, -1, -1)
            expected = self.model.text_fusion_norm(self.model.text_fusion(
                torch.cat([self.inputs['sam_text_mean'], mean], dim=-1)
            ))
        torch.testing.assert_close(seen[0], expected)
        (output * torch.randn_like(output)).mean().backward()
        for parameter in (
            self.model.text_fusion.weight,
            self.model.text_fusion_norm.weight,
            self.encoder.templates,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Averaging should distribute the text-path gradient to every template.
        grad = self.encoder.templates.grad
        torch.testing.assert_close(grad, grad[:, :1].expand_as(grad))

    def test_checkpoint_matches_forward_and_backward(self):
        checkpointed = copy.deepcopy(self.model)
        checkpointed.use_checkpoint = True
        plain = self.model(**self.inputs)['refiner_features_36']
        recomputed = checkpointed(**self.inputs)['refiner_features_36']
        torch.testing.assert_close(plain, recomputed)
        weight = torch.randn_like(plain)
        (plain * weight).mean().backward()
        (recomputed * weight).mean().backward()
        torch.testing.assert_close(
            self.model.text_fusion.weight.grad,
            checkpointed.text_fusion.weight.grad,
        )
        torch.testing.assert_close(
            self.encoder.templates.grad,
            checkpointed.clip_score_embed.clip_text_encoder.templates.grad,
        )
        checkpointed.eval()
        with torch.no_grad():
            inference = checkpointed(**self.inputs)['refiner_features_36']
        torch.testing.assert_close(plain.detach(), inference)


if __name__ == '__main__':
    unittest.main()
