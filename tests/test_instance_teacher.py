"""CPU regression tests; no SAM3/RemoteCLIP checkpoint downloads required."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

# Load the checkout as a package even when its directory contains a hyphen.
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('ovrs_sam3')
package.__path__ = [str(ROOT)]
sys.modules.setdefault('ovrs_sam3', package)

from ovrs_sam3.config_dataclasses import SemanticCriterionConfig
from ovrs_sam3.losses.semantic_criterion import SemanticCriterion
from ovrs_sam3.models.instance_query_attention import InstanceQueryAttention
from ovrs_sam3.models.maskformer_segmentation import UniversalSegmentationHead
from ovrs_sam3.models.sam3_image import Sam3Image
from ovrs_sam3.models.decoder_input_fusion import DecoderInputFusion
from ovrs_sam3.models.task_modes import OUTPUT_KEYS


torch.set_num_threads(1)


class InstanceLossTests(unittest.TestCase):
    def criterion(self, weight=0.7):
        return SemanticCriterion(SemanticCriterionConfig(
            final_balanced_bce_weight=0.0, final_dice_weight=0.0,
            sam3_mask_distill_weight=0.0, sam3_instance_distill_weight=weight,
        ))

    def test_full_image_ignore_pixels_aliases_and_chunk_gradient_equivalence(self):
        criterion = self.criterion()
        labels = torch.tensor([[[0, 255], [255, 255]], [[1, 255], [0, 255]]])
        mapping = torch.tensor([0, 1, 0, 2])
        context = criterion.prepare_streaming_context(
            {'label_map': labels}, 4, mapping, 3, target_hw=(2, 2),
        )
        expected_present = torch.tensor([[True, False, True, False], [True, True, True, False]])
        self.assertTrue(torch.equal(context.instance_presence_target, expected_present))
        self.assertEqual(context.instance_distill_pixel_denom, 20)
        torch.manual_seed(7)
        logits = torch.randn(2, 4, 2, 2, requires_grad=True)
        teacher = torch.rand_like(logits, requires_grad=True)
        outputs = {OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.sam3_instance_teacher_probs: teacher}
        full = criterion.forward_chunk(outputs, context, 0, 4)
        expected = F.binary_cross_entropy_with_logits(
            logits[expected_present], teacher.detach()[expected_present], reduction='mean',
        )
        torch.testing.assert_close(full['total_loss'], expected * 0.7)
        full['total_loss'].backward()
        expected_grad = logits.grad.clone()
        self.assertIsNone(teacher.grad)
        self.assertGreater(logits.grad[0, 0, 1, 1].abs().item(), 0)  # distant ignore pixel
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)  # absent class
        logits.grad = None
        chunk_total = 0
        for start, end in [(0, 1), (1, 3), (3, 4)]:
            chunk = criterion.forward_chunk({key: value[:, start:end] for key, value in outputs.items()}, context, start, end)
            chunk_total += chunk['total_loss'].detach()
            chunk['total_loss'].backward()
        torch.testing.assert_close(chunk_total, full['total_loss'].detach())
        torch.testing.assert_close(logits.grad, expected_grad)

    def test_small_gt_object_presence_survives_downsampling(self):
        labels = torch.full((1, 4, 4), 255)
        labels[0, 1, 1] = 0  # nearest-neighbor 4->2 drops this pixel
        context = self.criterion().prepare_streaming_context(
            {'label_map': labels}, 1, torch.tensor([0]), 1, target_hw=(2, 2),
        )
        self.assertFalse(context.presence_target.item())  # original semantic loss unchanged
        self.assertTrue(context.instance_presence_target.item())
        self.assertEqual(context.instance_distill_pixel_denom, 4)

    def test_absent_classes_and_disabled_loss_need_no_teacher(self):
        for weight in (0.0, 0.7):
            criterion = self.criterion(weight)
            labels = torch.full((1, 2, 2), 255)
            context = criterion.prepare_streaming_context(
                {'label_map': labels}, 1, torch.tensor([0]), 1, target_hw=(2, 2),
            )
            logits = torch.randn(1, 1, 2, 2, requires_grad=True)
            loss = criterion.forward_chunk({OUTPUT_KEYS.final_logits: logits}, context, 0, 1)['total_loss']
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)
        criterion = self.criterion(0.0)
        context = criterion.prepare_streaming_context(
            {'label_map': torch.zeros(1, 2, 2, dtype=torch.long)}, 1, torch.tensor([0]), 1, target_hw=(2, 2),
        )
        self.assertEqual(criterion.forward_chunk({OUTPUT_KEYS.final_logits: logits}, context, 0, 1)['total_loss'].item(), 0)

    def test_instance_loss_rejects_invalid_weight_or_missing_teacher(self):
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.criterion(weight)
        criterion = self.criterion()
        context = criterion.prepare_streaming_context(
            {'label_map': torch.zeros(1, 2, 2, dtype=torch.long)}, 1, torch.tensor([0]), 1, target_hw=(2, 2),
        )
        with self.assertRaises(ValueError):
            criterion.forward_chunk({OUTPUT_KEYS.final_logits: torch.zeros(1, 1, 2, 2)}, context, 0, 1)


class QueryAndTeacherTests(unittest.TestCase):
    def test_query_attention_preserves_pair_isolation_and_gradient_boundary(self):
        torch.manual_seed(3)
        attention = InstanceQueryAttention(8, 2)
        features = torch.randn(2, 8, 3, 3, requires_grad=True)
        queries = torch.randn(2, 5, 8, requires_grad=True)
        output = attention(features, queries)
        changed = queries.detach().clone()
        changed[1] *= -20
        torch.testing.assert_close(output[0], attention(features, changed)[0])
        self.assertGreater((output[1] - attention(features, changed)[1]).abs().sum().item(), 0)
        output.square().mean().backward()
        self.assertIsNone(queries.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(attention.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(attention.query_norm.weight.grad.abs().sum().item(), 0)

    def test_instance_aggregation_matches_dense_masks_and_empty_set(self):
        torch.manual_seed(2)
        decoder = nn.Identity()
        decoder.out_dim = 8
        head = UniversalSegmentationHead(8, 1, decoder, no_dec=False).eval()
        pixels = torch.randn(3, 8, 3, 3, requires_grad=True)
        queries = torch.randn(3, 37, 8, requires_grad=True)
        keep = torch.ones(3, 37, dtype=torch.bool)
        keep[1, 3:] = False
        keep[2] = False
        presence = torch.tensor([0.9, 0.6, 0.2])
        teacher = head.predict_instance_teacher(pixels, queries, keep, presence)
        with torch.no_grad():
            dense = head.mask_predictor(queries, head.instance_seg_head(pixels)).sigmoid()
            expected = (dense * keep[..., None, None]).amax(dim=1) * presence[:, None, None]
        torch.testing.assert_close(teacher, expected)
        self.assertEqual(teacher[2].sum().item(), 0)
        self.assertFalse(teacher.requires_grad)

    def test_real_original_decoder_shapes_and_no_gradient(self):
        from ovrs_sam3.model_builder import SAM3ModelBuilder
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.transformer = SAM3ModelBuilder._create_instance_transformer().eval()
        core.dot_prod_scoring = SAM3ModelBuilder._create_dot_product_scoring().eval()
        memory = torch.randn(4, 2, 256, requires_grad=True)
        encoder_out = dict(
            memory=memory, pos_embed=torch.randn_like(memory), padding_mask=None,
            level_start_index=torch.tensor([0]), spatial_shapes=torch.tensor([[2, 2]]),
            valid_ratios=torch.ones(2, 1, 2),
        )
        prompt = torch.randn(3, 2, 256)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        queries, scores, presence = core._run_instance_decoder(encoder_out, prompt, mask)
        self.assertEqual(queries.shape, (2, 200, 256))
        self.assertEqual(scores.shape, (2, 200))
        self.assertEqual(presence.shape, (2,))
        for tensor in (queries, scores, presence):
            self.assertFalse(tensor.requires_grad)
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertIsNone(memory.grad)
        self.assertIsNone(core.transformer.decoder.compilable_cord_cache[0].grad)


class CountingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls = []

    def forward(self, backbone_feats, image_ids, encoder_hidden_states, **kwargs):
        self.calls.append((torch.is_grad_enabled(), kwargs))
        spatial = encoder_hidden_states.mean(dim=-1).T.reshape(-1, 1, 72, 72)
        result = {'semantic_seg': F.interpolate(spatial * self.weight, (288, 288))}
        if kwargs:
            presence = kwargs['instance_presence']
            keep = kwargs['instance_keep'].any(dim=1)
            result['instance_teacher_probs'] = (presence * keep)[:, None, None].expand(-1, 288, 288)
        return result


class CoreDecodeTests(unittest.TestCase):
    def make_core(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        core.instance_score_threshold = 0.5
        core.segmentation_head = CountingHead()
        core.encoder_refiner = nn.Module()
        core.encoder_refiner.fusion = DecoderInputFusion(use_checkpoint=False)
        core.encoder_refiner.instance_query_attention = InstanceQueryAttention(256, 8)
        core.encoder_refiner.fuse_decoder_input_chunk = lambda **kwargs: core.encoder_refiner.fusion(**kwargs)
        # Pair order is image0-prompt0, image0-prompt1, image1-prompt0, image1-prompt1.
        scores = torch.tensor([[[0., 0., -10.], [10., -10., 10.]], [[10., 10., 10.], [10., 10., 10.]]])
        presence = torch.tensor([[0., 10.], [-10., 10.]])
        cache = {
            'cross_attended_encoder_features_72': torch.randn(2, 2, 256, 72, 72),
            'backbone_fpn': [torch.empty(2, 1, 1, 1), torch.empty(2, 1, 1, 1), torch.randn(2, 256, 72, 72)],
            'instance_queries': torch.randn(2, 2, 3, 256),
            'instance_score_logits': scores,
            'instance_presence_logits': presence,
        }
        return core, cache

    def test_training_two_decodes_inference_one_and_student_gradients(self):
        core, cache = self.make_core()
        core.train()
        refiner = torch.randn(2, 2, 256, 36, 36, requires_grad=True)
        captured = []
        handle = core.encoder_refiner.instance_query_attention.register_forward_pre_hook(
            lambda module, args: captured.append(args[1].detach().clone())
        )
        out = core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 2, True, True)
        self.assertEqual(len(core.segmentation_head.calls), 2)
        self.assertFalse(core.segmentation_head.calls[0][0])
        self.assertTrue(core.segmentation_head.calls[1][0])
        kwargs = core.segmentation_head.calls[0][1]
        expected_keep = (cache['instance_score_logits'].sigmoid() * cache['instance_presence_logits'].sigmoid()[..., None] > .5).reshape(4, 3)
        self.assertTrue(torch.equal(kwargs['instance_keep'], expected_keep))
        torch.testing.assert_close(captured[0], cache['instance_queries'].reshape(4, 3, 256))
        self.assertFalse(out[OUTPUT_KEYS.sam3_instance_teacher_probs].requires_grad)
        self.assertFalse(out[OUTPUT_KEYS.sam3_teacher_logits].requires_grad)
        out[OUTPUT_KEYS.final_logits].square().mean().backward()
        self.assertGreater(refiner.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.instance_query_attention.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.fusion.fusion_out_proj.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(core.segmentation_head.weight.grad)
        core.eval()
        core.segmentation_head.calls.clear()
        with torch.no_grad():
            out = core.decode_encoder_refiner_chunk_from_cache(cache, refiner.detach(), 0, 2)
        self.assertEqual(len(core.segmentation_head.calls), 1)
        self.assertEqual(core.segmentation_head.calls[0][1], {})
        self.assertEqual(set(out), {OUTPUT_KEYS.final_logits})
        with self.assertRaises(RuntimeError):
            core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 2, return_instance_teacher=True)
        handle.remove()

    def test_query_decoder_reads_raw_encoder_before_prompt_attention(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        raw_memory = torch.randn(72 * 72, 1, 256)
        core._encode_prompt = lambda *args: (torch.zeros(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool), {})
        core._run_full_encoder = lambda **kwargs: {'memory': raw_memory}
        seen = []
        def decode(encoder_out, prompt, mask):
            seen.append(encoder_out['memory'])
            return torch.zeros(1, 200, 256), torch.zeros(1, 200), torch.zeros(1)
        core._run_instance_decoder = decode
        core.segmentation_head = nn.Module()
        core.segmentation_head.apply_prompt_cross_attention = lambda **kwargs: kwargs['encoder_hidden_states'] + 100
        out = core._encode_sam3_chunk({}, None, None, 1, 1)
        self.assertIs(seen[0], raw_memory)
        torch.testing.assert_close(out['encoder_features_72'], (raw_memory + 100).permute(1, 2, 0).reshape(1, 1, 256, 72, 72))


class StreamingAndCheckpointTests(unittest.TestCase):
    def test_streaming_trainer_returns_instance_metrics_and_proxy_gradients(self):
        from ovrs_sam3.engine.trainer import Trainer
        core, cache = CoreDecodeTests().make_core()
        core.train()
        cache['chunk_class_counts'] = [1, 1]
        seed = nn.Parameter(torch.randn(2, 2, 256, 36, 36))
        model = types.SimpleNamespace(
            build_encoder_refiner_cache=lambda batch: cache,
            run_encoder_refiner_lowres_from_cache=lambda *args, **kwargs: {'refiner_features_36': seed * 2},
            decode_encoder_refiner_chunk_from_cache=core.decode_encoder_refiner_chunk_from_cache,
        )
        batch = types.SimpleNamespace(
            find_targets=[types.SimpleNamespace(semantic_label_map=torch.tensor([[[0, 255], [255, 255]], [[1, 255], [255, 255]]]))],
            find_metadatas=[types.SimpleNamespace(num_prompts=2, num_classes=2, prompt_to_class_id=[0, 1])],
        )
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.cfg = types.SimpleNamespace(use_amp=False)
        trainer.device = torch.device('cpu')
        trainer.criterion = SemanticCriterion(SemanticCriterionConfig(
            sam3_mask_distill_weight=0.0, sam3_instance_distill_weight=0.4,
        ))
        trainer.scaler = torch.amp.GradScaler('cpu', enabled=False)
        metrics = trainer._compute_train_loss_streaming(batch)
        self.assertEqual(len(core.segmentation_head.calls), 4)
        self.assertGreater(seed.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.instance_query_attention.attention.in_proj_weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(metrics['loss_sam3_instance_distill_weighted'], metrics['loss_sam3_instance_distill_bce'] * .4)
        self.assertFalse(metrics['total_loss'].requires_grad)
        self.assertTrue(torch.isfinite(metrics['total_loss']))

    def test_base_checkpoint_requires_original_instance_parameters(self):
        from ovrs_sam3.model_builder import SAM3ModelBuilder
        core = nn.Module()
        core.transformer = nn.Module()
        core.transformer.decoder = nn.Linear(2, 2)
        core.dot_prod_scoring = nn.Linear(2, 2)
        core.segmentation_head = nn.Module()
        core.segmentation_head.mask_predictor = nn.Linear(2, 2)
        core.segmentation_head.instance_seg_head = nn.Conv2d(2, 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'sam3.pt')
            state = {'detector.' + key: value for key, value in core.state_dict().items()}
            torch.save({'model': state}, path)
            SAM3ModelBuilder._load_checkpoint(core, path)
            del state['detector.transformer.decoder.weight']
            torch.save({'model': state}, path)
            with self.assertRaisesRegex(RuntimeError, 'pretrained SAM3 instance branch'):
                SAM3ModelBuilder._load_checkpoint(core, path)


if __name__ == '__main__':
    unittest.main()
