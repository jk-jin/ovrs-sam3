"""CPU regressions for the residual FPN72 + learned Refiner input path."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder_refiner import ClassConditionedEncoderRefiner


class FixedTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("templates", torch.randn(1, 64, 8))

    def encode_prompt_templates(self, class_names, device, **kwargs):
        return self.templates.to(device).expand(len(class_names), -1, -1)


class CaptureLayer(nn.Module):
    def forward(self, feature_36, score_embed_36, clip_mid_36, sam_fpn_36):
        self.feature = feature_36.detach().clone()
        self.score = score_embed_36.detach().clone()
        self.fpn = sam_fpn_36.detach().clone()
        return feature_36, score_embed_36


class RefinerInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)

    def make_model(self, layers=0, mode="local_3x3"):
        return ClassConditionedEncoderRefiner(
            clip_text_encoder=FixedTextEncoder(),
            clip_dim=8,
            clip_native_dim=4,
            prompt_templates=["aerial {}"] * 64,
            fusion_layers=layers,
            dropout=0.0,
            use_checkpoint=False,
            score_intra_attn_type=mode,
            feature_intra_attn_type=mode,
        )

    def make_inputs(self, batch=1, prompts=2):
        return dict(
            encoder_features_72=torch.randn(batch, prompts, 256, 72, 72),
            sam_fpn_72=torch.randn(batch, 256, 72, 72),
            clip_image_feat_map=torch.randn(batch, 8, 36, 36),
            clip_mid_features=[torch.randn(batch, 4, 36, 36) for _ in range(2)],
            clip_mid_layer_indices=(7, 15),
            class_names=[f"class_{i}" for i in range(prompts)],
        )

    def test_residual_before_convolution_broadcasts_without_mutating_caches(self):
        model = self.make_model().eval()
        capture = CaptureLayer()
        model.layers.append(capture)
        model.feature_output_norm = nn.Identity()
        inputs = self.make_inputs(batch=2, prompts=2)
        encoder = inputs["encoder_features_72"].clone()
        fpn = inputs["sam_fpn_72"].clone()
        conv = model.feature_input_downsample
        with torch.no_grad():
            conv.weight.zero_()
            channels = torch.arange(256)
            conv.weight[channels, channels, 1, 1] = 2.0
            conv.weight[channels, channels, 0, 0] = 0.5
            conv.bias.fill_(-0.25)
            outputs = model(**inputs)

        # An asymmetric kernel checks addition at 72×72 before downsampling,
        # including padding at the top/left boundary and both prompt axes.
        residual = encoder + fpn[:, None]
        expected = 2.0 * residual[..., ::2, ::2]
        expected += 0.5 * F.pad(residual, (1, 0, 1, 0))[..., :72:2, :72:2]
        expected -= 0.25
        torch.testing.assert_close(capture.feature, expected)
        torch.testing.assert_close(outputs["refiner_features_36"], expected)
        torch.testing.assert_close(inputs["encoder_features_72"], encoder, rtol=0, atol=0)
        torch.testing.assert_close(inputs["sam_fpn_72"], fpn, rtol=0, atol=0)
        torch.testing.assert_close(capture.score, outputs["clip_score_embed_36"])
        expected_guidance = F.interpolate(fpn, (36, 36), mode="bilinear", align_corners=False)
        torch.testing.assert_close(capture.fpn, expected_guidance[:, None].expand_as(capture.fpn))

    def test_convolution_trains_with_frozen_sam_inputs(self):
        model = self.make_model().train()
        model.feature_output_norm = nn.Identity()
        inputs = self.make_inputs()
        conv = model.feature_input_downsample
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
        before = conv.weight.detach().clone()
        model(**inputs)["refiner_features_36"].square().mean().backward()
        for parameter in conv.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(inputs["encoder_features_72"].grad)
        self.assertIsNone(inputs["sam_fpn_72"].grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, conv.weight))

    def test_four_layer_checkpoint_output_and_input_conv_gradients(self):
        for mode in ("window", "local_3x3"):
            with self.subTest(mode=mode):
                model = self.make_model(layers=4, mode=mode).train()
                inputs = self.make_inputs()
                target = torch.randn(1, 2, 256, 36, 36)
                reference = None
                for use_checkpoint in (False, True):
                    model.use_checkpoint = use_checkpoint
                    model.zero_grad(set_to_none=True)
                    outputs = model(**inputs)
                    (outputs["refiner_features_36"] * target).mean().backward()
                    result = (
                        outputs["refiner_features_36"].detach().clone(),
                        outputs["score_embed_36"].detach().clone(),
                        model.feature_input_downsample.weight.grad.clone(),
                        model.feature_input_downsample.bias.grad.clone(),
                    )
                    for tensor in result:
                        self.assertTrue(torch.isfinite(tensor).all())
                    self.assertGreater(result[2].abs().sum().item(), 0)
                    if reference is None:
                        reference = result
                    else:
                        for actual, expected in zip(result, reference):
                            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
                    del outputs


if __name__ == "__main__":
    unittest.main()
