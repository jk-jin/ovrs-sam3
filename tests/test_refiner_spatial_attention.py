"""Small CPU regression checks; no pretrained weights or image encoder needed."""
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.decoder_input_fusion import DecoderInputFusion
    from models.encoder_refiner_attention import EncoderRefinerLayer
    from models.maskformer_segmentation import PixelDecoder
    from models.refiner_spatial_attention import GlobalFeatureAttention, LocalScoreAttention


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SpatialAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_local_attention_matches_explicit_neighbours(self):
        module = LocalScoreAttention(8, 4, 2, dropout=0).double().eval()
        feature = torch.randn(2, 2, 8, 3, 4, dtype=torch.double)
        score = torch.randn(2, 2, 4, 3, 4, dtype=torch.double)
        context = torch.randn_like(feature)
        actual = module(feature, score, context)
        f = module.norm_feature(feature.permute(0, 1, 3, 4, 2))
        s = module.norm_score(score.permute(0, 1, 3, 4, 2))
        g = module.norm_context(context.permute(0, 1, 3, 4, 2))
        q = module.q_proj(torch.cat((f, s, g), -1))
        k = module.k_proj(torch.cat((f, s, g), -1))
        vf = module.v_feature_proj(torch.cat((f, g), -1))
        vs = module.v_score_proj(s)
        expected_f, expected_s = torch.empty_like(feature), torch.empty_like(score)
        for y in range(3):
            for x in range(4):
                neighbours = [(y + dy, x + dx, (dy + 1) * 3 + dx + 1)
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                              if 0 <= y + dy < 3 and 0 <= x + dx < 4]
                query = q[:, :, y, x].reshape(2, 2, 2, 4)
                keys = torch.stack([k[:, :, yy, xx].reshape(2, 2, 2, 4) for yy, xx, _ in neighbours], -2)
                bias = module.relative_position_bias[:, [offset for _, _, offset in neighbours]]
                weights = ((query.unsqueeze(-2) * keys).sum(-1) / 2 + bias).softmax(-1)
                def aggregate(value, projection):
                    values = torch.stack([value[:, :, yy, xx].reshape(2, 2, 2, 4) for yy, xx, _ in neighbours], -2)
                    return projection((weights.unsqueeze(-1) * values).sum(-2).reshape(2, 2, 8))
                expected_f[:, :, :, y, x] = aggregate(vf, module.out_feature_proj)
                expected_s[:, :, :, y, x] = aggregate(vs, module.out_score_proj)
        torch.testing.assert_close(actual[0], expected_f)
        torch.testing.assert_close(actual[1], expected_s)

    def test_local_attention_does_not_read_distant_pixels(self):
        module = LocalScoreAttention(8, 8, 2, dropout=0).eval()
        f = torch.randn(1, 1, 8, 5, 5)
        s, g = torch.randn_like(f), torch.randn_like(f)
        baseline = module(f, s, g)
        changed = f.clone()
        changed[..., 4, 4] += torch.randn(8) * 10
        outputs = module(changed, s, g)
        for before, after in zip(baseline, outputs):
            torch.testing.assert_close(before[..., 0, 0], after[..., 0, 0])

    def test_global_bias_gradient_and_class_isolation(self):
        module = GlobalFeatureAttention(8, 2, dropout=0).eval()
        f = torch.randn(1, 2, 8, 36, 36, requires_grad=True)
        result = module(f)
        self.assertEqual(result.shape, f.shape)
        self.assertEqual(module.relative_position_bias_table.shape, (1225, 2))
        index = module.relative_position_index
        self.assertEqual(index[0, 0].item(), 612)
        self.assertEqual(index[0, -1].item(), 0)
        self.assertEqual(index[-1, 0].item(), 1224)
        changed = f.detach().clone()
        changed[:, 0] += torch.randn_like(changed[:, 0])
        torch.testing.assert_close(result[:, 1], module(changed)[:, 1])
        result.square().mean().backward()
        self.assertGreater(module.relative_position_bias_table.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_layer_computes_global_once_and_all_four_local_steps_receive_gradients(self):
        module = EncoderRefinerLayer(8, 8, 2, local_attn_steps=4, dropout=0)
        calls = []
        hook = module.global_attn.register_forward_hook(lambda *args: calls.append(1))
        f = torch.randn(1, 2, 8, 36, 36, requires_grad=True)
        s = torch.randn_like(f, requires_grad=True)
        out_f, out_s = module(f, s, torch.randn(1, 2, 8))
        (out_f.square().mean() + out_s.square().mean()).backward()
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(module.local_attns), 4)
        self.assertIsNot(module.local_attns[0], module.local_attns[1])
        for local in module.local_attns:
            self.assertGreater(local.v_feature_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(local.v_score_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.global_attn.qkv.weight.grad.abs().sum().item(), 0)

    def test_fusion_checkpoint_and_batch_class_broadcast(self):
        plain = DecoderInputFusion(8, 8, use_checkpoint=False).train()
        checked = DecoderInputFusion(8, 8, use_checkpoint=True).train()
        checked.load_state_dict(plain.state_dict())
        f = torch.randn(4, 8, 36, 36, requires_grad=True)
        f_checked = f.detach().clone().requires_grad_()
        encoder, fpn = torch.randn(4, 8, 72, 72), torch.randn(2, 8, 72, 72)
        out = plain(f, encoder, fpn)
        out_checked = checked(f_checked, encoder, fpn)
        self.assertEqual(out.shape, encoder.shape)
        torch.testing.assert_close(out, out_checked)
        out.square().mean().backward()
        out_checked.square().mean().backward()
        torch.testing.assert_close(f.grad, f_checked.grad)
        self.assertGreater(plain.fpn_proj[0].weight.grad.abs().sum().item(), 0)

    def test_frozen_pixel_decoder_preserves_student_input_gradients(self):
        decoder = PixelDecoder(8, 2).eval().requires_grad_(False)
        feature = torch.randn(2, 8, 4, 4, requires_grad=True)
        fpn = [torch.randn(1, 8, 16, 16), torch.randn(1, 8, 8, 8), feature]
        student = decoder(fpn)
        with torch.no_grad():
            teacher = decoder(fpn)
        self.assertFalse(teacher.requires_grad)
        self.assertTrue(student.requires_grad)
        student.square().mean().backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
