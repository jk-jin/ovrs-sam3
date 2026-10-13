"""CPU checks for stacked 72×72 semantic/detail fusion."""
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

    def test_stack_upsamples_once_and_each_layer_reuses_original_encoder(self):
        model = DecoderInputFusion(use_checkpoint=False).eval()
        self.assertEqual(len(model.layers), 2)
        self.assertIsNot(model.layers[0].input_fusion, model.layers[1].input_fusion)
        refiner, encoder, fpn = self.inputs()
        original = encoder.clone()
        concatenated, mixed, branch_inputs, stage_outputs, encoder_inputs, fpn_inputs = [], [], [], [], [], []
        interpolate = F.interpolate
        with ExitStack() as stack:
            resize = stack.enter_context(patch("models.decoder_input_fusion.F.interpolate", wraps=interpolate))
            for layer in model.layers:
                stack.enter_context(layer.input_fusion.register_forward_pre_hook(lambda m, args: concatenated.append(args[0])))
                stack.enter_context(layer.input_fusion.register_forward_hook(lambda m, args, out: mixed.append(out)))
                for projection in (layer.semantic_refiner_proj, layer.detail_refiner_proj):
                    stack.enter_context(projection.register_forward_pre_hook(lambda m, args: branch_inputs.append(args[0])))
                stack.enter_context(layer.encoder_proj.register_forward_pre_hook(lambda m, args: encoder_inputs.append(args[0])))
                stack.enter_context(layer.fpn_proj.register_forward_pre_hook(lambda m, args: fpn_inputs.append(args[0])))
                stack.enter_context(layer.register_forward_hook(lambda m, args, out: stage_outputs.append(out)))
            with torch.no_grad():
                result = model(refiner, encoder, fpn)
            self.assertEqual(resize.call_count, 1)
        self.assertEqual(tuple(result.shape), (2, 256, 72, 72))
        expected_first = interpolate(refiner, size=(72, 72), mode="bilinear", align_corners=False)
        torch.testing.assert_close(concatenated[0][:, :256], expected_first)
        torch.testing.assert_close(concatenated[1][:, :256], stage_outputs[0])
        for index in range(2):
            torch.testing.assert_close(concatenated[index][:, 256:], original)
            self.assertIs(encoder_inputs[index], encoder)
            self.assertIs(fpn_inputs[index], fpn)
            self.assertIs(branch_inputs[index * 2], mixed[index])
            self.assertIs(branch_inputs[index * 2 + 1], mixed[index])
        torch.testing.assert_close(encoder, original, rtol=0, atol=0)

    def test_prompt_chunking_preserves_per_image_fpn_broadcast(self):
        model = DecoderInputFusion(use_checkpoint=False).eval()
        refiner, encoder, fpn = self.inputs(batch=2, prompts=2)
        with torch.no_grad():
            whole = model(refiner, encoder, fpn).reshape(2, 2, 256, 72, 72)
            chunks = []
            for indices in ([0, 2], [1, 3]):
                chunks.append(model(refiner[indices], encoder[indices], fpn))
        torch.testing.assert_close(whole, torch.stack(chunks, dim=1))

    def test_checkpoint_matches_outputs_and_gradients_through_both_layers(self):
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
            for layer in model.layers:
                for projection in (layer.input_fusion, layer.semantic_out_proj, layer.detail_out_proj):
                    self.assertTrue(torch.isfinite(projection.weight.grad).all())
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(grads[0], grads[1])
        for (name, first), (other_name, second) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(first.grad, second.grad)
        self.assertIsNone(data[1].grad)
        self.assertIsNone(data[2].grad)

    def test_layer_count_and_entry_shapes_are_validated(self):
        for invalid in (0, -1, 2.5, True, None):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                DecoderInputFusion(num_layers=invalid)
        for count in (1, 3):
            self.assertEqual(len(DecoderInputFusion(num_layers=count).layers), count)
        model = DecoderInputFusion(use_checkpoint=False)
        data = list(self.inputs())
        for index, shape in ((0, (2, 256, 72, 72)), (1, (1, 256, 72, 72)), (2, (3, 256, 72, 72))):
            invalid_inputs = list(data)
            invalid_inputs[index] = torch.randn(*shape)
            with self.assertRaises(ValueError):
                model(*invalid_inputs)


if __name__ == "__main__":
    unittest.main()
