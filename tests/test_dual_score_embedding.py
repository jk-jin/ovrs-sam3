"""CPU tests without pretrained weights: python -m unittest discover -s tests."""
import copy
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.score_embeddings import ClipScoreEmbedding


class TemplateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.templates = nn.Parameter(torch.randn(3, 64, 12))
        self.calls = 0

    def encode_prompt_templates(self, class_names, **kwargs):
        self.calls += 1
        return self.templates[[int(name) for name in class_names]]


class DualScoreEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.encoder = TemplateEncoder()
        self.model = ClipScoreEmbedding(
            self.encoder, ['an image of {}'] * 64,
            clip_output_dim=12, score_embed_dim=8,
        )
        self.images = torch.randn(2, 12, 36, 36, requires_grad=True)
        self.names = ['0', '1', '2']

    def test_shapes_broadcast_and_direct_condition_gradients(self):
        conditions = {}

        def capture(name):
            def hook(module, args):
                scores, condition = args
                conditions[name] = condition
                # Exclude the similarity-score path to test direct conditioning.
                return scores.detach(), condition
            return hook

        handles = [branch.register_forward_pre_hook(capture(name)) for name, branch in (
            ('image', self.model.image_branch), ('text', self.model.text_branch)
        )]
        output, scores, templates = self.model(self.names, self.images)
        for handle in handles:
            handle.remove()
        self.assertEqual(self.encoder.calls, 1)
        self.assertEqual(tuple(output.shape), (2, 3, 8, 36, 36))
        expected_scores = torch.einsum(
            'ckd,bdhw->bckhw', F.normalize(templates, dim=-1),
            F.normalize(self.images, dim=1),
        ) * 20.0
        torch.testing.assert_close(scores, expected_scores)
        image_condition = conditions['image'].reshape(2, 3, 12, 36, 36)
        text_condition = conditions['text'].reshape(2, 3, 12, 36, 36)
        torch.testing.assert_close(
            image_condition, F.normalize(self.images, dim=1)[:, None].expand_as(image_condition),
        )
        text_mean = F.normalize(templates.mean(1), dim=-1)
        torch.testing.assert_close(
            text_condition, text_mean[None, :, :, None, None].expand_as(text_condition),
        )
        (output * torch.randn_like(output)).mean().backward()
        for parameter in [self.images, self.encoder.templates, *self.model.parameters()]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.images.grad.abs().sum().item(), 0)
        self.assertGreater(self.encoder.templates.grad.abs().sum().item(), 0)
        for branch in (self.model.image_branch, self.model.text_branch):
            self.assertGreater(branch.score_stem[0].weight.grad.abs().sum().item(), 0)
        grad = self.encoder.templates.grad
        torch.testing.assert_close(grad, grad[:, :1].expand_as(grad))

    def test_batch_and_class_permutations(self):
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.names, self.images)[0]
            reordered = self.model(['2', '0', '1'], self.images.flip(0))[0]
        torch.testing.assert_close(reordered, original.flip(0)[:, [2, 0, 1]])

    def test_sam_guidance_and_checkpoint_gradients(self):
        model = ClassConditionedEncoderRefiner(
            self.encoder, hidden_dim=8, clip_dim=12, score_embed_dim=8,
            num_heads=2, fusion_layers=2, dropout=0.0, use_checkpoint=False,
            prompt_templates=['an image of {}'] * 64,
        )
        checkpointed = copy.deepcopy(model)
        checkpointed.use_checkpoint = True
        inputs = dict(
            encoder_features_72=torch.randn(2, 3, 8, 72, 72),
            clip_image_feat_map=self.images.detach(),
            sam_text_mean=torch.randn(2, 3, 8),
            class_names=self.names,
        )
        seen = []
        handles = [layer.class_attn.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs['sam_text_mean']),
            with_kwargs=True,
        ) for layer in model.layers]
        plain = model(**inputs)['refiner_features_36']
        for handle in handles:
            handle.remove()
        self.assertEqual(len(seen), len(model.layers))
        for text, layer in zip(seen, model.layers):
            torch.testing.assert_close(text, layer.class_norm_text(inputs['sam_text_mean']))
        self.assertFalse(any('text_fusion' in name for name in model.state_dict()))
        recomputed = checkpointed(**inputs)['refiner_features_36']
        torch.testing.assert_close(plain, recomputed)
        weight = torch.randn_like(plain)
        (plain * weight).mean().backward()
        (recomputed * weight).mean().backward()
        other_params = dict(checkpointed.named_parameters())
        for name, param in model.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(param.grad, other_params[name].grad)
        torch.testing.assert_close(
            self.encoder.templates.grad,
            checkpointed.clip_score_embed.clip_text_encoder.templates.grad,
        )
        checkpointed.eval()
        with torch.no_grad():
            inference = checkpointed(**inputs)['refiner_features_36']
        torch.testing.assert_close(plain.detach(), inference)

    def test_default_channels_and_autocast(self):
        model = ClipScoreEmbedding(
            self.encoder, ['an image of {}'] * 64, clip_output_dim=12,
        )
        for branch in (model.image_branch, model.text_branch):
            spatial = [m for m in branch.modules()
                       if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
            self.assertEqual(len(spatial), 1)
            self.assertEqual((spatial[0].in_channels, spatial[0].out_channels), (512, 256))
        self.assertIsNot(model.image_branch.score_stem[0].weight, model.text_branch.score_stem[0].weight)
        self.assertIsInstance(model.output_fusion, nn.Conv2d)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(['0'], self.images[:1])[0]
            loss = output.float().square().mean()
        self.assertEqual(tuple(output.shape), (1, 1, 256, 36, 36))
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(model.text_branch.condition_fusion[0].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
