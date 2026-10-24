import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.encoder_refiner import ClassConditionedEncoderRefiner, center_refiner_features


class DummyScores(nn.Module):
    def forward(self, class_names, remoteclip_feat_map):
        b, _, h, w = remoteclip_feat_map.shape
        p = len(class_names)
        score = remoteclip_feat_map.new_zeros(b, p, 256, h, w)
        return score, score[:, :, :64], remoteclip_feat_map.new_zeros(p, 64, 4)


class ClassCenteringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_single_prompt_classes_and_per_image_pixel_mean(self):
        x = torch.randn(2, 3, 4, 5, 6, dtype=torch.double)
        y = center_refiner_features(x, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(y, x - x.mean(1, keepdim=True))
        torch.testing.assert_close(y.mean(1), torch.zeros_like(y[:, 0]))
        offset = torch.randn(2, 1, 4, 5, 6, dtype=torch.double)
        torch.testing.assert_close(center_refiner_features(x + offset, [0, 1, 2]), y)

    def test_unequal_prompt_counts_class_weighting_and_permutation(self):
        x = torch.tensor([2., 4., 10.]).reshape(1, 3, 1, 1, 1)
        y = center_refiner_features(x, [0, 0, 1])
        torch.testing.assert_close(y.flatten(), torch.tensor([-4.5, -2.5, 3.5]))
        torch.testing.assert_close((y[:, :2].mean(1) + y[:, 2]) / 2, torch.zeros_like(y[:, 2]))
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(center_refiner_features(x[:, order], [1, 0, 0]), y[:, order])

    def test_single_class(self):
        x = torch.randn(2, 1, 3, 4, 4)
        torch.testing.assert_close(center_refiner_features(x, [0]), torch.zeros_like(x))
        x = torch.randn(2, 2, 3, 4, 4)
        torch.testing.assert_close(center_refiner_features(x, [0, 0]), x - x.mean(1, keepdim=True))

    def test_gradient_flows_to_other_classes(self):
        x = torch.randn(1, 3, 1, 1, 1, dtype=torch.double, requires_grad=True)
        center_refiner_features(x, [0, 0, 1])[:, 0].sum().backward()
        torch.testing.assert_close(x.grad.flatten(), torch.tensor([.75, -.25, -.5], dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(lambda v: center_refiner_features(v, [0, 0, 1]), (x,)))

    def test_proxy_chunk_backward_matches_full_backward(self):
        original = torch.randn(2, 5, 4, 3, 3, dtype=torch.double)
        target = torch.randn_like(original)
        mapping = [0, 0, 1, 2, 2]
        full = original.clone().requires_grad_()
        ((center_refiner_features(full, mapping) - target) ** 2).mean().backward()
        for chunk_size in (1, 2, 4, 5):
            x = original.clone().requires_grad_()
            centered = center_refiner_features(x, mapping)
            proxy = centered.detach().requires_grad_()
            for start in range(0, 5, chunk_size):
                loss = ((proxy[:, start:start + chunk_size] - target[:, start:start + chunk_size]) ** 2).sum() / target.numel()
                loss.backward()
            centered.backward(proxy.grad)
            torch.testing.assert_close(x.grad, full.grad)

    def test_low_precision_preserves_dtype_and_finite_gradient(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(1, 3, 2, 4, 4).to(dtype).requires_grad_()
            y = center_refiner_features(x, [0, 0, 1])
            self.assertEqual(y.dtype, dtype)
            expected_mean = (x[:, :2].float().mean(1, keepdim=True) + x[:, 2:].float()) / 2
            torch.testing.assert_close(y, x - expected_mean.to(dtype))
            y.float().square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_refiner_forward_centers_after_norm_in_train_and_eval(self):
        model = ClassConditionedEncoderRefiner(
            clip_text_encoder=nn.Linear(1, 1), clip_dim=4, clip_native_dim=2,
            prompt_templates=['{}'] * 64, fusion_layers=2, dropout=0.,
            score_intra_attn_type='local_3x3', feature_intra_attn_type='local_3x3',
            use_checkpoint=True,
        )
        model.clip_score_embed = DummyScores()
        encoder = torch.randn(1, 3, 256, 72, 72)
        kwargs = dict(encoder_features_72=encoder, clip_image_feat_map=torch.randn(1, 4, 36, 36),
                      class_names=['ship', 'vessel', 'building'], prompt_to_class_id=torch.tensor([0, 0, 1]),
                      clip_mid_features=[torch.randn(1, 2, 36, 36) for _ in range(2)],
                      clip_mid_layer_indices=(7, 15), sam_fpn_72=torch.randn(1, 256, 72, 72))
        captured = []
        hook = model.feature_output_norm.register_forward_hook(lambda m, args, out: captured.append(out))
        for training in (True, False):
            model.train(training)
            with torch.set_grad_enabled(training), patch('models.encoder_refiner.center_refiner_features', wraps=center_refiner_features) as spy:
                out = model(**kwargs)['refiner_features_36']
                self.assertEqual(spy.call_count, 1)
                normalized = captured[-1].permute(0, 1, 4, 2, 3).contiguous()
                torch.testing.assert_close(out, center_refiner_features(normalized, kwargs['prompt_to_class_id']))
                if training:
                    out[:, 0].square().mean().backward()
                    self.assertGreater(model.layers[0].class_attn.v_proj.weight.grad.abs().sum().item(), 0)
                    train_output = out.detach()
                else:
                    torch.testing.assert_close(out, train_output)
        hook.remove()

    def test_invalid_mapping(self):
        with self.assertRaises(ValueError):
            center_refiner_features(torch.zeros(1, 2, 4, 3, 3), [0])


if __name__ == '__main__':
    unittest.main()
