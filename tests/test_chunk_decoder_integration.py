"""Builder-to-chunk-decoder integration using real, randomly initialized heads."""
from contextlib import ExitStack
import copy
import importlib
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("ovrs_test")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("ovrs_test", package)
builder = importlib.import_module("ovrs_test.model_builder").SAM3ModelBuilder
heads = importlib.import_module("ovrs_test.models.maskformer_segmentation")
torch.set_num_threads(1)


class ChunkDecoderIntegrationTest(unittest.TestCase):
    def config(self):
        cfg = copy.deepcopy(runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))["model"])
        cfg.update(checkpoint_path=None, load_from_hf=False, device="cpu")
        cfg["openclip_cfg"]["pretrained"] = None
        return cfg

    def test_builder_rejects_invalid_attention_config(self):
        for field, value in (("post_fusion_value_dim", 255), ("post_fusion_qk_dim", 63),
                             ("post_fusion_num_heads", 0), ("post_fusion_attn_layers", -1),
                             ("post_fusion_query_chunk_size", 0), ("post_fusion_dropout", 1.0),
                             ("intra_position_encoding", "unknown")):
            cfg = self.config()
            cfg["encoder_refiner_cfg"][field] = value
            with self.assertRaises(ValueError):
                builder.build_config(**cfg)

    def test_real_head_chunk_teacher_and_student_gradients(self):
        cfg = self.config()
        cfg["encoder_refiner_cfg"].update(
            post_fusion_qk_dim=32, post_fusion_value_dim=256,
            post_fusion_query_chunk_size=128, post_fusion_dropout=0,
            intra_position_encoding="sam3_sine",
        )
        cfg = builder.build_config(**cfg)
        text, image, transformer = nn.Identity(), nn.Identity(), nn.Identity()
        text.output_dim = image.output_dim = image.native_dim = 8
        transformer.d_model = 256
        head = heads.UniversalSegmentationHead(256, 2, heads.PixelDecoder(256, 2), act_ckpt=True)
        head.requires_grad_(False).eval()
        # Only the pretrained feature encoders are replaced. Build wiring,
        # Refiner/fusion/attention, Pixel Decoder and semantic head are real.
        replacements = {
            "_create_position_encoding": None, "_create_vit_backbone": nn.Identity(),
            "_create_vit_neck": nn.Identity(), "_create_text_encoder": nn.Identity(),
            "_create_vl_backbone": nn.Identity(), "_create_geometry_encoder": nn.Identity(),
            "_create_openclip_encoders": (text, image),
            "_create_encoder_only_transformer": transformer, "_create_segmentation_head": head,
        }
        with ExitStack() as stack:
            for name, result in replacements.items():
                stack.enter_context(patch.object(builder, name, return_value=result))
            model = builder.build_semantic_core_model(cfg)
        self.assertEqual(model.encoder_refiner.post_fusion_attention[0].qk_dim, 32)
        self.assertEqual(model.encoder_refiner.post_fusion_attention[0].query_chunk_size, 128)
        self.assertIsNone(model.encoder_refiner.layers[0].window_attn_regular.relative_position_bias_table
                          if model.encoder_refiner.layers[0].intra_attn_type == "window"
                          else model.encoder_refiner.layers[0].local_attn[0].relative_position_bias_table)
        cache = {
            "cross_attended_encoder_features_72": torch.randn(1, 2, 256, 72, 72),
            "backbone_fpn": [torch.randn(1, 256, size, size) for size in (288, 144, 72)],
        }
        refiner = torch.randn(1, 2, 256, 36, 36, requires_grad=True)
        for start in range(2):
            outputs = model.decode_encoder_refiner_chunk_from_cache(
                cache, refiner[:, start:start+1], start, start+1, return_teacher_logits=True,
            )
            self.assertEqual(outputs["final_logits"].shape, (1, 1, 288, 288))
            self.assertTrue(outputs["final_logits"].requires_grad)
            self.assertFalse(outputs["sam3_teacher_logits"].requires_grad)
            with torch.no_grad():
                original = cache["cross_attended_encoder_features_72"][:, start:start+1]
                expected_teacher = head(
                    cache["backbone_fpn"], torch.tensor([0]),
                    model._feature_72_to_hidden_states(original),
                )["semantic_seg"].reshape(1, 1, 288, 288)
            torch.testing.assert_close(outputs["sam3_teacher_logits"], expected_teacher)
            outputs["final_logits"].square().mean().backward()
            del outputs
        self.assertTrue(torch.isfinite(refiner.grad).all())
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        for attention in model.encoder_refiner.post_fusion_attention:
            self.assertGreater(attention.v_proj.weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            outputs = model.decode_encoder_refiner_chunk_from_cache(cache, refiner[:, :1], 0, 1)
        self.assertEqual(set(outputs), {"final_logits"})
        self.assertEqual(outputs["final_logits"].shape, (1, 1, 288, 288))


if __name__ == "__main__":
    unittest.main()
