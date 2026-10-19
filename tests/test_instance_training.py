"""CPU regressions; no pretrained weights or datasets are downloaded."""
import importlib
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

# The repository directory may contain a hyphen, so import it as a package alias.
ROOT = Path(__file__).resolve().parents[1]
if 'ovrs_testpkg' not in sys.modules:
    package = types.ModuleType('ovrs_testpkg')
    package.__path__ = [str(ROOT)]
    sys.modules['ovrs_testpkg'] = package


def module(name):
    return importlib.import_module('ovrs_testpkg.' + name)


cfg_module = module('config_dataclasses')
criterion_module = module('losses.semantic_criterion')
instance_utils = module('models.instance_utils')
Criterion = criterion_module.SemanticCriterion
Config = cfg_module.SemanticCriterionConfig
Sam3Image = module('models.sam3_image').Sam3Image
Builder = module('model_builder').SAM3ModelBuilder


def outputs_from_leaves(semantic, masks, scores, presence, threshold=0.5):
    union, fused = instance_utils.aggregate_instances(semantic, masks, scores, presence, threshold)
    return dict(final_logits=semantic, instance_logits=masks, query_logits=scores,
                presence_logits=presence, instance_union_log_prob=union, fused_log_prob=fused)


def attach_teacher(out, teacher, present):
    indices = present.flatten().nonzero(as_tuple=True)[0]
    if indices.numel():
        out['teacher_pair_indices'] = indices
        for output_key, teacher_key in (
            ('teacher_semantic_logits', 'semantic'), ('teacher_instance_logits', 'masks'),
            ('teacher_query_logits', 'scores'),
        ):
            out[output_key] = teacher[teacher_key].flatten(0, 1)[indices]
    return out


def distill_only(**kwargs):
    return Config(final_balanced_bce_weight=0, instance_bce_weight=0, fusion_bce_weight=0,
                  presence_bce_weight=0, absent_query_bce_weight=0, **kwargs)


class AggregationTests(unittest.TestCase):
    def test_threshold_presence_and_unweighted_kept_masks(self):
        semantic = torch.logit(torch.tensor([[[[0.2, 0.4]]]]))
        masks = torch.logit(torch.tensor([[[[[0.9, 0.1]], [[0.3, 0.95]]]]]))
        scores = torch.logit(torch.tensor([[[0.9, 0.6]]]))
        presence = torch.logit(torch.tensor([[0.8]]))
        union, fused = instance_utils.aggregate_instances(semantic, masks, scores, presence, 0.5)
        torch.testing.assert_close(union.exp(), torch.tensor([[[[0.81, 0.57]]]]))
        torch.testing.assert_close(fused.exp(), torch.tensor([[[[0.9, 0.4]]]]))
        # Presence is shared, but its multiplication filters the second query.
        _, none = instance_utils.aggregate_instances(semantic, masks, scores, presence, 1)
        torch.testing.assert_close(none.exp(), semantic.sigmoid())

    def test_equal_threshold_is_removed(self):
        _, fused = instance_utils.aggregate_instances(
            torch.full((1, 1, 1, 1), -2.), torch.full((1, 1, 1, 1, 1), 3.),
            torch.zeros(1, 1, 1), torch.zeros(1, 1), 0.25)
        torch.testing.assert_close(fused.exp(), torch.full_like(fused, -2.).sigmoid())

    def test_training_union_has_score_gradients_without_presence_or_filter(self):
        semantic = torch.full((1, 1, 1, 1), -2., requires_grad=True)
        masks = torch.ones(1, 1, 2, 1, 1, requires_grad=True)
        scores = torch.full((1, 1, 2), -6., requires_grad=True)
        presence = torch.full((1, 1), -6., requires_grad=True)
        union, fused = instance_utils.aggregate_instances(semantic, masks, scores, presence, 0.5)
        instance_utils.binary_cross_entropy_log_prob(union, torch.ones_like(union)).sum().backward()
        self.assertGreater(scores.grad.abs().sum(), 0)
        self.assertGreater(masks.grad.abs().sum(), 0)
        self.assertIsNone(presence.grad)
        self.assertIsNone(semantic.grad)
        torch.testing.assert_close(fused.exp(), semantic.sigmoid())

    def test_bce_is_stable_for_extreme_logits_and_cpu_amp(self):
        for dtype in (torch.float32, torch.bfloat16):
            masks = torch.full((1, 1, 2, 1, 1), 80., dtype=dtype, requires_grad=True)
            semantic = torch.full((1, 1, 1, 1), 80., dtype=dtype, requires_grad=True)
            scores = torch.full((1, 1, 2), 80., dtype=dtype, requires_grad=True)
            presence = torch.full((1, 1), 80., dtype=dtype)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                union, fused = instance_utils.aggregate_instances(semantic, masks, scores, presence, 0.5)
                loss = instance_utils.binary_cross_entropy_log_prob(union, torch.zeros_like(union)).sum()
                loss += instance_utils.binary_cross_entropy_log_prob(fused, torch.zeros_like(fused)).sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(masks.grad).all())
            self.assertGreater(loss, 100)
            torch.testing.assert_close(instance_utils.log_prob_to_logits(fused), semantic.float())


class CriterionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.labels = torch.tensor([[[0, 0, 255, 255], [0, 1, 1, 255],
                                     [0, 1, 255, 255], [255, 255, 255, 255]],
                                    [[2, 2, 2, 255], [2, 0, 0, 255],
                                     [2, 0, 255, 255], [255, 255, 255, 255]]])
        self.mapping = torch.tensor([0, 1, 2, 0])  # Includes an alias.
        self.leaves = [torch.randn(2, 4, 4, 4, requires_grad=True),
                       torch.randn(2, 4, 3, 4, 4, requires_grad=True),
                       torch.randn(2, 4, 3, requires_grad=True),
                       torch.randn(2, 4, requires_grad=True)]
        self.teacher = {key: torch.randn(*shape, requires_grad=True) for key, shape in (
            ('semantic', (2, 4, 4, 4)), ('masks', (2, 4, 3, 4, 4)), ('scores', (2, 4, 3)))}

    def context(self, criterion, labels=None, hw=(4, 4)):
        return criterion.prepare_streaming_context(dict(label_map=self.labels if labels is None else labels),
                                                    4, self.mapping, 3, hw)

    def test_presence_precedes_resize_and_aliases_share_labels(self):
        criterion = Criterion()
        source = torch.full((1, 8, 8), 255)
        source[0, 7, 7] = 1
        ctx = self.context(criterion, source, (1, 1))
        self.assertTrue(ctx.presence_target[0, 1])
        self.assertEqual(ctx.label_map.item(), 255)
        torch.testing.assert_close(ctx.presence_target[0, 0], ctx.presence_target[0, 3])

    def test_all_distillation_present_only_and_teacher_has_no_gradient(self):
        criterion = Criterion(distill_only())
        ctx = self.context(criterion)
        out = attach_teacher(outputs_from_leaves(*self.leaves), self.teacher, ctx.presence_target)
        result = criterion.forward_chunk(out, ctx, 0, 4)
        result['total_loss'].backward()
        absent = ~ctx.presence_target
        for leaf in self.leaves[:3]:
            self.assertEqual(leaf.grad[absent].abs().sum(), 0)
            self.assertGreater(leaf.grad[~absent].abs().sum(), 0)
        for teacher in self.teacher.values():
            self.assertIsNone(teacher.grad)
        self.assertIsNone(self.leaves[3].grad)
        self.assertNotIn('loss_final_dice', result)

    def test_full_and_chunked_loss_and_gradients_are_equivalent(self):
        criterion = Criterion()
        ctx = self.context(criterion)
        full = attach_teacher(outputs_from_leaves(*self.leaves), self.teacher, ctx.presence_target)
        expected = criterion.forward_chunk(full, ctx, 0, 4)
        expected['total_loss'].backward()
        full_grad = [leaf.grad.clone() for leaf in self.leaves]
        for leaf in self.leaves:
            leaf.grad = None
        sums = {}
        for start, end in ((0, 1), (1, 3), (3, 4)):
            chunk = outputs_from_leaves(*(leaf[:, start:end] for leaf in self.leaves))
            teacher = {key: value[:, start:end] for key, value in self.teacher.items()}
            attach_teacher(chunk, teacher, ctx.presence_target[:, start:end])
            result = criterion.forward_chunk(chunk, ctx, start, end)
            result['total_loss'].backward()
            for key, value in result.items():
                sums[key] = sums.get(key, 0) + value.detach()
        for key in expected:
            torch.testing.assert_close(sums[key], expected[key].detach())
        for got, wanted in zip(self.leaves, full_grad):
            torch.testing.assert_close(got.grad, wanted)

    def test_no_present_class_skips_teacher_but_supervises_absence(self):
        criterion = Criterion()
        ctx = self.context(criterion, torch.full_like(self.labels, 255))
        self.assertFalse(criterion.needs_teacher(ctx))
        result = criterion.forward_chunk(outputs_from_leaves(*self.leaves), ctx, 0, 4)
        for name in ('sam3_mask_distill', 'instance_mask_distill', 'instance_score_distill'):
            self.assertEqual(result['loss_' + name + '_bce'], 0)
        result['total_loss'].backward()
        self.assertGreater(self.leaves[2].grad.abs().sum(), 0)
        self.assertGreater(self.leaves[3].grad.abs().sum(), 0)
        incomplete = Criterion(Config(presence_annotations_complete=False))
        unknown = self.context(incomplete, torch.full_like(self.labels, 255))
        self.assertEqual(unknown.num_presence_pairs, 0)
        self.assertEqual(unknown.num_absent_pairs, 0)

    def test_distill_boundary_is_class_specific_and_ignore_only(self):
        ctx = self.context(Criterion(Config(sam3_mask_distill_boundary_width=1)))
        boundary = ctx.distill_outer_boundary_by_class
        self.assertFalse((boundary & ctx.valid_mask[:, None]).any())
        self.assertTrue(boundary[0, 0, 0, 2])
        self.assertFalse(boundary[0, 2].any())  # Class 2 absent from image 0.

    def test_hungarian_recovers_permutation_and_uses_only_valid_pixels(self):
        teacher = torch.tensor([[[8., -8.]], [[-8., 8.]]])
        rows, cols = criterion_module.match_instances(
            teacher.flip(0), torch.ones(2), teacher, torch.ones(2),
            torch.ones(1, 2, dtype=torch.bool), 20)
        torch.testing.assert_close(cols, torch.tensor([1, 0]))
        self.assertEqual(rows.numel(), 2)

    def test_incorrect_teacher_coverage_and_invalid_config_fail(self):
        criterion = Criterion()
        ctx = self.context(criterion)
        with self.assertRaisesRegex(ValueError, 'exactly the present pairs'):
            criterion.forward_chunk(outputs_from_leaves(*self.leaves), ctx, 0, 4)
        for kw in ({'instance_match_num_points': 0}, {'fusion_bce_weight': float('nan')},
                   {'sam3_mask_distill_boundary_width': -1}):
            with self.assertRaises(ValueError):
                Criterion(Config(**kw))


class ModelTests(unittest.TestCase):
    def test_full_config_and_missing_instance_checkpoint(self):
        raw = module('engine.config').Config.fromfile(ROOT / 'configs/train/isaid_loveda_full.py')
        # Validate config construction without requiring local RemoteCLIP weights.
        cfg = Builder._normalize_build_cfg(cfg_module.SegmentorBuildConfig(**raw['model']))
        self.assertEqual(cfg.prompt_chunk_size, 1)
        self.assertEqual(cfg.instance_score_threshold, 0.5)
        self.assertEqual(cfg.criterion_cfg.fusion_bce_weight, 0.1)
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.decoder = nn.Linear(2, 2)
        model.dot_prod_scoring = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'weights.pt')
            torch.save({'model': {}}, path)
            with self.assertRaisesRegex(RuntimeError, 'instance weights are incomplete'):
                Builder._load_checkpoint(model, path)
            torch.save({'model': model.state_dict()}, path)
            Builder._load_checkpoint(model, path)

    def test_inference_adapter_receives_fused_scores(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.eval()
        core.run_encoder_refiner_lowres_from_cache = lambda **kw: dict(
            refiner_features_36=torch.zeros(1, 2, 1, 1, 1))
        def decode(**kw):
            start = kw['class_start']
            return dict(final_logits=torch.zeros(1, 1, 288, 288),
                        fused_log_prob=torch.full((1, 1, 288, 288), (0.8 + start * 0.1)).log())
        core.decode_encoder_refiner_chunk_from_cache = decode
        output = core.run_encoder_refiner_from_cache(
            dict(chunk_class_counts=[1, 1]), object())
        torch.testing.assert_close(output['final_logits'].sigmoid()[0, :, 0, 0], torch.tensor([0.8, 0.9]))
        adapter = module('models.adapters.semantic_adapter').SemanticSegAdapter()
        metadata = types.SimpleNamespace(num_prompts=2, num_classes=2, prompt_to_class_id=[0, 1])
        adapted = adapter(output, types.SimpleNamespace(find_metadatas=[metadata]), 'infer')
        torch.testing.assert_close(adapted['raw_prompt_score_map'][0, :, 0, 0], torch.tensor([0.8, 0.9]))

    def test_actual_frozen_decoder_keeps_200_queries_and_input_gradients(self):
        transformer = Builder._create_transformer().eval().requires_grad_(False)
        scorer = Builder._create_dot_product_scoring().eval().requires_grad_(False)
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.transformer, core.dot_prod_scoring = transformer, scorer
        memory = torch.randn(16, 2, 256, requires_grad=True)
        inputs = dict(padding_mask=None, pos_embed=torch.randn_like(memory),
                      level_start_index=torch.zeros(1, dtype=torch.long),
                      spatial_shapes=torch.tensor([[4, 4]]), valid_ratios=torch.ones(2, 1, 2),
                      prompt=torch.randn(3, 2, 256), prompt_mask=torch.zeros(2, 3, dtype=torch.bool))
        queries, scores, presence = core._decode_queries(memory, inputs)
        self.assertEqual(queries.shape, (2, 200, 256))
        self.assertEqual(scores.shape, (2, 200))
        self.assertEqual(presence.shape, (2,))
        F.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores)).backward()
        self.assertGreater(memory.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in core.parameters()))
        self.assertFalse(transformer.decoder.dac)

    def test_shared_pixel_decoder_generates_both_heads_and_preserves_gradients(self):
        head_module = module('models.maskformer_segmentation')
        head = head_module.UniversalSegmentationHead(
            hidden_dim=8, upsampling_stages=3,
            pixel_decoder=head_module.PixelDecoder(hidden_dim=8, num_upsampling_stages=3),
            no_dec=False, act_ckpt=True)
        head.eval().requires_grad_(False)
        features = [torch.randn(2, 8, h, h) for h in (288, 144, 72)]
        memory = torch.randn(72 * 72, 2, 8, requires_grad=True)
        queries = torch.randn(2, 3, 8, requires_grad=True)
        calls = []
        handle = head.pixel_decoder.register_forward_hook(lambda *args: calls.append(True))
        output = head(features, torch.arange(2), memory, queries)
        self.assertEqual(len(calls), 1)
        self.assertEqual(output['pred_masks'].shape, (2, 3, 288, 288))
        loss = F.binary_cross_entropy_with_logits(output['pred_masks'], torch.zeros_like(output['pred_masks']))
        loss.backward()
        handle.remove()
        self.assertGreater(memory.grad.abs().sum(), 0)
        self.assertGreater(queries.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_encode_cache_preserves_raw_memory_separately(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        raw = torch.ones(72 * 72, 1, 256)
        core._encode_prompt = lambda *args: (torch.zeros(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool), {})
        core._run_full_encoder = lambda **kw: dict(memory=raw, padding_mask=None,
            pos_embed=raw * 0, level_start_index=torch.zeros(1, dtype=torch.long),
            spatial_shapes=torch.tensor([[72, 72]]), valid_ratios=torch.ones(1, 1, 2))
        core.segmentation_head = nn.Module()
        core.segmentation_head.apply_prompt_cross_attention = lambda **kw: kw['encoder_hidden_states'] + 10
        out = core._encode_sam3_chunk({}, None, None, 1, 1)
        self.assertEqual(out['query_inputs']['memory'].mean(), 1)
        self.assertEqual(out['encoder_features_72'].mean(), 11)
        self.assertFalse(out['encoder_features_72'].requires_grad)

    def test_teacher_pair_selection_uses_image_prompt_order_and_no_grad(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        core.instance_score_threshold = 0.5
        core.encoder_refiner = nn.Module()
        core.encoder_refiner.use_checkpoint = True
        core.encoder_refiner.fuse_decoder_input_chunk = lambda **kw: F.interpolate(
            kw['refiner_feature_36'], size=(72, 72), mode='nearest') + 100
        observed = []
        def decode(memory, inputs):
            value = memory.mean((0, 2))
            observed.append((value.detach().clone(), torch.is_grad_enabled()))
            return value[:, None, None].expand(-1, 200, 256), value[:, None].expand(-1, 200), value
        core._decode_queries = decode
        class Head(nn.Module):
            def forward(self, backbone_feats, image_ids, encoder_hidden_states, obj_queries):
                value = encoder_hidden_states.mean((0, 2))
                return dict(semantic_seg=value[:, None, None, None].expand(-1, 1, 288, 288),
                            pred_masks=value[:, None, None, None].expand(-1, 200, 288, 288))
        core.segmentation_head = Head()
        raw = torch.arange(4.).view(1, 4, 1).expand(72 * 72, 4, 256)
        query = dict(memory=raw, padding_mask=None, pos_embed=raw * 0,
                     prompt=torch.zeros(2, 4, 256), prompt_mask=torch.zeros(4, 2, dtype=torch.bool),
                     valid_ratios=torch.ones(4, 1, 2), spatial_shapes=torch.tensor([[72, 72]]),
                     level_start_index=torch.zeros(1, dtype=torch.long))
        original = (raw + 10).permute(1, 2, 0).reshape(2, 2, 256, 72, 72)
        cache = dict(cross_attended_encoder_features_72=original,
                     backbone_fpn=[torch.zeros(2, 256, 72, 72)] * 3, query_chunks={(0, 2): query})
        feature = torch.zeros(2, 2, 256, 36, 36, requires_grad=True)
        present = torch.tensor([[True, False], [False, True]])
        # Aggregation arithmetic is tested separately; avoid huge expanded mask allocations here.
        def lightweight_aggregate(semantic, masks, scores, presence, threshold):
            return F.logsigmoid(semantic), F.logsigmoid(semantic)
        with patch('ovrs_testpkg.models.sam3_image.aggregate_instances', lightweight_aggregate):
            out = core.decode_encoder_refiner_chunk_from_cache(cache, feature, 0, 2, True, present)
            out['fused_log_prob'].sum().backward()
        torch.testing.assert_close(out['teacher_pair_indices'], torch.tensor([0, 3]))
        torch.testing.assert_close(observed[0][0], torch.tensor([0., 3.]))
        self.assertFalse(observed[0][1])
        torch.testing.assert_close(observed[1][0], torch.full((4,), 100.))
        torch.testing.assert_close(out['teacher_semantic_logits'][:, 0, 0], torch.tensor([10., 13.]))
        self.assertGreater(feature.grad.abs().sum(), 0)
        self.assertFalse(out['teacher_instance_logits'].requires_grad)


class StreamingTrainerTests(unittest.TestCase):
    def test_real_trainer_proxy_backward_matches_full_prompt_chunk(self):
        Trainer = module('engine.trainer').Trainer
        labels = torch.tensor([[[0, 0, 1, 255], [0, 1, 1, 255]],
                               [[2, 2, 0, 255], [2, 0, 0, 255]]])
        mapping = torch.tensor([0, 1, 2, 0])
        batch = types.SimpleNamespace(
            find_targets=[types.SimpleNamespace(semantic_label_map=labels)],
            find_metadatas=[types.SimpleNamespace(num_prompts=4, num_classes=3,
                                                 prompt_to_class_id=mapping)])
        class ToyModel(nn.Module):
            def __init__(self, chunk_size):
                super().__init__()
                self.weight = nn.Parameter(torch.linspace(-0.4, 0.4, 8).reshape(2, 4, 1, 1, 1))
                self.chunk_size = chunk_size
                self.teacher_requests = []
            def build_encoder_refiner_cache(self, batch):
                return dict(chunk_class_counts=[self.chunk_size] * (4 // self.chunk_size))
            def run_encoder_refiner_lowres_from_cache(self, *args, **kw):
                return dict(refiner_features_36=self.weight * 2 + self.weight.mean())
            def decode_encoder_refiner_chunk_from_cache(self, encoder_refiner_cache,
                    refiner_feature_36_chunk, class_start, class_end,
                    return_teacher_logits, teacher_pair_mask):
                self.teacher_requests.append((class_start, teacher_pair_mask.clone(), return_teacher_logits))
                value = refiner_feature_36_chunk[..., 0, 0, 0]
                semantic = value[..., None, None].expand(-1, -1, 288, 288)
                offsets = torch.tensor([-0.5, 0., 0.5])[None, None, :, None, None]
                masks = semantic[:, :, None] + offsets
                scores = value[..., None] + torch.tensor([-0.3, 0., 0.3])
                out = outputs_from_leaves(semantic, masks, scores, value)
                if return_teacher_logits:
                    attach_teacher(out, dict(semantic=semantic.detach() + 0.2,
                        masks=masks.detach() + 0.2, scores=scores.detach() + 0.2), teacher_pair_mask)
                return out
        results = []
        for chunk_size in (1, 4):
            trainer = Trainer.__new__(Trainer)
            trainer.device = torch.device('cpu')
            trainer.cfg = types.SimpleNamespace(use_amp=False)
            trainer.scaler = torch.amp.GradScaler('cpu', enabled=False)
            trainer.model = ToyModel(chunk_size)
            trainer.criterion = Criterion()
            losses = trainer._compute_train_loss_streaming(batch)
            results.append((losses, trainer.model.weight.grad))
            self.assertTrue(trainer.model.teacher_requests)
            self.assertTrue(all(req[2] for req in trainer.model.teacher_requests))
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key])
        torch.testing.assert_close(results[0][1], results[1][1])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
