from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedWindowAttention2D(nn.Module):
    """
    2D shifted-window cross-attention.

    Input:
        query_map: [B, D, H, W]
        key_map:   [B, D, H, W]
        value_map: [B, D, H, W]

    Symbol meanings:
        B means batch size.
        D means feature dimension.
        H and W mean spatial height and width.

    New design support:
        use_qkv_proj=False disables internal q/k/v linear projections.
        use_out_proj=False disables internal output linear projection.
        residual_source="query" keeps the query map as the residual branch.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        dropout: float = 0.1,
        value_preserving: bool = False,
        residual_source: str = "query",
        use_residual_norm: bool = True,
        use_rel_pos_bias: bool = True,
        use_qkv_proj: bool = True,
        use_out_proj: bool | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.value_preserving = bool(value_preserving)
        self.residual_source = str(residual_source)
        self.use_residual_norm = bool(use_residual_norm)
        self.use_rel_pos_bias = bool(use_rel_pos_bias)
        self.use_qkv_proj = bool(use_qkv_proj)

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size, "
                f"got shift_size={shift_size}, window_size={window_size}."
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        if self.residual_source not in {"query", "value"}:
            raise ValueError(
                "residual_source must be 'query' or 'value', "
                f"got {self.residual_source!r}."
            )

        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        if use_out_proj is None:
            # Backward-compatible default:
            # old value_preserving=True had no v_proj/out_proj;
            # old value_preserving=False had v_proj/out_proj.
            self.use_out_proj = self.use_qkv_proj and not self.value_preserving
        else:
            self.use_out_proj = bool(use_out_proj)

        if self.use_qkv_proj:
            self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

            if self.value_preserving:
                self.v_proj = None
            else:
                self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None

        self.out_proj = (
            nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_out_proj
            else None
        )

        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_dropout = nn.Dropout(float(dropout))

        self.out_norm = (
            nn.LayerNorm(self.hidden_dim, eps=1e-6)
            if self.use_residual_norm
            else None
        )

        if self.use_rel_pos_bias:
            num_rel_pos = (2 * self.window_size - 1) * (2 * self.window_size - 1)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(num_rel_pos, self.num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_h = torch.arange(self.window_size)
            coords_w = torch.arange(self.window_size)
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing="ij")
            )
            coords_flatten = torch.flatten(coords, 1)

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size - 1
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 0] *= 2 * self.window_size - 1

            relative_position_index = relative_coords.sum(-1)
            self.register_buffer(
                "relative_position_index",
                relative_position_index,
                persistent=False,
            )
        else:
            self.relative_position_bias_table = None
            self.register_buffer(
                "relative_position_index",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

    @staticmethod
    def _pad_to_window_size(
        x: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size

        if pad_h == 0 and pad_w == 0:
            return x, height, width

        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        return x, height, width

    def _map_to_windows(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim, height, width = x.shape
        window = self.window_size

        x = x.reshape(
            batch_size,
            dim,
            height // window,
            window,
            width // window,
            window,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(-1, window * window, dim)

    def _windows_to_map(
        self,
        x: torch.Tensor,
        batch_size: int,
        padded_h: int,
        padded_w: int,
        original_h: int,
        original_w: int,
    ) -> torch.Tensor:
        window = self.window_size
        dim = self.hidden_dim

        x = x.reshape(
            batch_size,
            padded_h // window,
            padded_w // window,
            window,
            window,
            dim,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, dim, padded_h, padded_w)
        return x[:, :, :original_h, :original_w].contiguous()

    def _build_shift_attn_mask(
        self,
        padded_h: int,
        padded_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.shift_size <= 0:
            return None

        window = self.window_size
        shift = self.shift_size

        img_mask = torch.zeros(
            (1, 1, padded_h, padded_w),
            device=device,
            dtype=dtype,
        )
        h_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )
        w_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = count
                count += 1

        mask_windows = self._map_to_windows(img_mask).squeeze(-1)
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _add_relative_position_bias(self, attn: torch.Tensor) -> torch.Tensor:
        if not self.use_rel_pos_bias:
            return attn

        num_tokens = self.window_size * self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.reshape(num_tokens, num_tokens, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()
        return attn + bias.unsqueeze(0).to(dtype=attn.dtype, device=attn.device)

    def _project_windows(
        self,
        query_windows: torch.Tensor,
        key_windows: torch.Tensor,
        value_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_qkv_proj:
            if self.q_proj is None or self.k_proj is None:
                raise RuntimeError("q_proj/k_proj must exist when use_qkv_proj=True.")

            q = self.q_proj(query_windows)
            k = self.k_proj(key_windows)

            if self.value_preserving:
                v = value_windows
            else:
                if self.v_proj is None:
                    raise RuntimeError(
                        "v_proj must exist when value_preserving=False "
                        "and use_qkv_proj=True."
                    )
                v = self.v_proj(value_windows)
        else:
            q = query_windows
            k = key_windows
            v = value_windows

        return q, k, v

    def forward(
        self,
        query_map: torch.Tensor,
        key_map: torch.Tensor,
        value_map: torch.Tensor,
    ) -> torch.Tensor:
        if query_map.dim() != 4:
            raise ValueError(
                f"query_map must be [B, D, H, W], got {tuple(query_map.shape)}."
            )
        if key_map.shape != query_map.shape:
            raise ValueError(
                "key_map must have the same shape as query_map, "
                f"got {tuple(key_map.shape)} vs {tuple(query_map.shape)}."
            )
        if value_map.shape != query_map.shape:
            raise ValueError(
                "value_map must have the same shape as query_map, "
                f"got {tuple(value_map.shape)} vs {tuple(query_map.shape)}."
            )

        batch_size, dim, original_h, original_w = query_map.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.hidden_dim}, got {dim}."
            )

        query_map, _, _ = self._pad_to_window_size(query_map, self.window_size)
        key_map, _, _ = self._pad_to_window_size(key_map, self.window_size)
        value_map, _, _ = self._pad_to_window_size(value_map, self.window_size)

        padded_h, padded_w = int(query_map.shape[-2]), int(query_map.shape[-1])

        if self.shift_size > 0:
            shifts = (-self.shift_size, -self.shift_size)
            query_map = torch.roll(query_map, shifts=shifts, dims=(-2, -1))
            key_map = torch.roll(key_map, shifts=shifts, dims=(-2, -1))
            value_map = torch.roll(value_map, shifts=shifts, dims=(-2, -1))

        query_windows = self._map_to_windows(query_map)
        key_windows = self._map_to_windows(key_map)
        value_windows = self._map_to_windows(value_map)

        num_windows_total, num_tokens, _ = query_windows.shape

        q, k, v = self._project_windows(
            query_windows=query_windows,
            key_windows=key_windows,
            value_windows=value_windows,
        )

        q = q.reshape(
            num_windows_total,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        k = k.reshape(
            num_windows_total,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        v = v.reshape(
            num_windows_total,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self._add_relative_position_bias(attn)

        attn_mask = self._build_shift_attn_mask(
            padded_h=padded_h,
            padded_w=padded_w,
            device=attn.device,
            dtype=attn.dtype,
        )
        if attn_mask is not None:
            num_windows_per_image = int(attn_mask.shape[0])
            attn = attn.reshape(
                batch_size,
                num_windows_per_image,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attn = attn + attn_mask[None, :, None, :, :]
            attn = attn.reshape(
                num_windows_total,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(num_windows_total, num_tokens, self.hidden_dim)

        if self.out_proj is not None:
            out = self.out_proj(out)

        out = self.out_dropout(out)

        residual = query_windows if self.residual_source == "query" else value_windows
        out = residual + out

        if self.out_norm is not None:
            out = self.out_norm(out)

        out_map = self._windows_to_map(
            x=out,
            batch_size=batch_size,
            padded_h=padded_h,
            padded_w=padded_w,
            original_h=padded_h,
            original_w=padded_w,
        )

        if self.shift_size > 0:
            out_map = torch.roll(
                out_map,
                shifts=(self.shift_size, self.shift_size),
                dims=(-2, -1),
            )

        return out_map[:, :, :original_h, :original_w].contiguous()