"""CPU checks for single-stage 72×72 semantic/detail fusion."""
from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.decoder_input_fusion import DecoderInputFusion


class DecoderFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(43)

    def inputs(self, batch=1, prompts=2):
        return (
            torch.randn(batch * prompts, 256, 36, 36),
            torch.randn(batch * prompts, 256, 72, 72),
            torch.randn(batch, 256, 72, 72),
        )

    def test_single_fusion_upsamples_once_and_routes_original_inputs(self):
        model = DecoderInputFusion(use_checkpoint=False).eval()
        self.assertFalse(hasattr(model, "layers"))
        self.assertFalse(hasattr(model, "input_fusion"))
        refiner, encoder, fpn = self.inputs()
        original_encoder, original_fpn = encoder.clone(), fpn.clone()
        branch_inputs, encoder_inputs, fpn_inputs = [], [], []
        interpolate = F.interpolate
        with ExitStack() as stack:
            resize = stack.enter_context(patch("models.decoder_input_fusion.F.interpolate", wraps=interpolate))
            for projection in (model.semantic_refiner_proj, model.detail_refiner_proj):
                stack.enter_context(projection.register_forward_pre_hook(lambda m, args: branch_inputs.append(args[0])))
            stack.enter_context(model.encoder_proj.register_forward_pre_hook(lambda m, args: encoder_inputs.append(args[0])))
            stack.enter_context(model.fpn_proj.register_forward_pre_hook(lambda m, args: fpn_inputs.append(args[0])))
            with torch.no_grad():
                result = model(refiner, encoder, fpn)
            self.assertEqual(resize.call_count, 1)
        self.assertEqual(tuple(result.shape), (2, 256, 72, 72))
        expected = interpolate(refiner, size=(72, 72), mode="bilinear", align_corners=False)
        torch.testing.assert_close(branch_inputs[0], expected)
        self.assertIs(branch_inputs[0], branch_inputs[1])
        self.assertIs(encoder_inputs[0], encoder)
        self.assertIs(fpn_inputs[0], fpn)
        torch.testing.assert_close(encoder, original_encoder, rtol=0, atol=0)
        torch.testing.assert_close(fpn, original_fpn, rtol=0, atol=0)

    def test_prompt_chunking_preserves_per_image_fpn_broadcast(self):
        model = DecoderInputFusion(use_checkpoint=False).eval()
        refiner, encoder, fpn = self.inputs(batch=2, prompts=2)
        with torch.no_grad():
            whole = model(refiner, encoder, fpn).reshape(2, 2, 256, 72, 72)
            chunks = []
            for indices in ([0, 2], [1, 3]):
                chunks.append(model(refiner[indices], encoder[indices], fpn))
        torch.testing.assert_close(whole, torch.stack(chunks, dim=1))

    def test_checkpoint_matches_outputs_and_gradients_through_both_branches(self):
        plain = DecoderInputFusion(use_checkpoint=False).train()
        checked = deepcopy(plain)
        checked.use_checkpoint = True
        data = self.inputs()
        target = torch.randn(2, 256, 72, 72)
        outputs, grads = [], []
        for model in (plain, checked):
            refiner = data[0].clone().requires_grad_()
            out = model(refiner, data[1], data[2])
            (out * target).mean().backward()
            outputs.append(out)
            grads.append(refiner.grad)
            self.assertTrue(torch.isfinite(refiner.grad).all())
            self.assertGreater(refiner.grad.abs().sum().item(), 0)
            for projection in (model.semantic_refiner_proj[0], model.detail_refiner_proj[0], model.semantic_out_proj, model.detail_out_proj):
                self.assertTrue(torch.isfinite(projection.weight.grad).all())
                self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(grads[0], grads[1])
        for (name, first), (other_name, second) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(first.grad, second.grad)
        self.assertIsNone(data[1].grad)
        self.assertIsNone(data[2].grad)

    def test_entry_shapes_are_validated(self):
        model = DecoderInputFusion(use_checkpoint=False)
        data = list(self.inputs())
        for index, shape in ((0, (2, 256, 72, 72)), (1, (1, 256, 72, 72)), (2, (3, 256, 72, 72))):
            invalid_inputs = list(data)
            invalid_inputs[index] = torch.randn(*shape)
            with self.assertRaises(ValueError):
                model(*invalid_inputs)


if __name__ == "__main__":
    unittest.main()
