from __future__ import annotations

from typing import Dict, Optional

import torch

from .data_misc import BatchedDatapoint
from .geometry_encoders import Prompt
from .sam3_image import Sam3Image


class Sam3Core(Sam3Image):
    """A thin wrapper around Sam3Image that exposes *raw predictions only*.

    This class keeps all encoder/decoder/mask-head logic from your current
    SAM3 image model, but removes training-time target conversion and Hungarian
    matching from the forward path.

    Why this matters:
    - instance segmentation can still use matcher-based loss outside the model
    - semantic segmentation can skip matcher entirely
    - one shared forward supports instance / semantic / hybrid training
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Matching is intentionally moved out of the model and into criterion.
        self.matcher = None

    def forward_grounding_raw(
        self,
        backbone_out: Dict[str, torch.Tensor],
        find_input,
        geometric_prompt: Prompt,
    ) -> Dict[str, torch.Tensor]:
        with torch.profiler.record_function("Sam3Core._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )

        with torch.profiler.record_function("Sam3Core._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )

        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
            # Useful for adapters / criteria that need prompt information.
            "prompt": prompt,
            "prompt_mask": prompt_mask,
        }

        with torch.profiler.record_function("Sam3Core._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        with torch.profiler.record_function("Sam3Core._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        device = self.device

        backbone_out = self.backbone.forward_image(input.img_batch)
        backbone_out.update(
            self.backbone.forward_text(input.find_text_batch, device=device)
        )

        assert len(input.find_inputs) == 1, (
            "Current simplified trainer assumes exactly one find stage per batch."
        )
        find_input = input.find_inputs[0]

        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )
        return self.forward_grounding_raw(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
        )

    def forward_instance(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        """Alias kept for trainer readability."""
        return self.forward(input)

    def forward_semantic(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        """Same raw forward; semantic-specific aggregation happens in adapter."""
        return self.forward(input)
