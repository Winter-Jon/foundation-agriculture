"""Opt-in Qwen3-VL vision forward compatibility for Transformers 5.9.0.

This is intentionally loaded only by setting ``PYTHONPATH`` to this directory.
It restores the 5.8.1 Qwen3VLVisionModel position/forward implementation while
leaving the installed Transformers package at 5.9.0.  It is a diagnostic
compatibility layer for ZeRO-3 gradient-checkpointing validation, not a global
site-packages modification.
"""
from __future__ import annotations

import os


if os.environ.get('AGRINET_TF590_QWEN3VL_VISION_COMPAT') == '1':
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(height, width) for _, height, width in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        pos_ids = torch.empty((sum(t * h * w for t, h, w in grid_thw_list), 2), dtype=torch.long, device=freq_table.device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size
            row_idx = (
                torch.arange(merged_h, device=freq_table.device)[:, None, None, None] * merge_size
                + torch.arange(merge_size, device=freq_table.device)[None, None, :, None]
            ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = (
                torch.arange(merged_w, device=freq_table.device)[None, :, None, None] * merge_size
                + torch.arange(merge_size, device=freq_table.device)[None, None, None, :]
            ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            pos_ids[offset : offset + coords.shape[0]] = coords
            offset += coords.shape[0]
        return freq_table[pos_ids].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list, weight_list = [[] for _ in range(4)], [[] for _ in range(4)]
        for _, height, width in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, height)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, width)
            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - h_floor, w_idxs - w_floor
            base_h, base_h_ceil = h_floor * self.num_grid_per_side, h_ceil * self.num_grid_per_side
            indices = [(base_h[None].T + w_floor[None]).flatten(), (base_h[None].T + w_ceil[None]).flatten(), (base_h_ceil[None].T + w_floor[None]).flatten(), (base_h_ceil[None].T + w_ceil[None]).flatten()]
            weights = [((1 - dh)[None].T * (1 - dw)[None]).flatten(), ((1 - dh)[None].T * dw[None]).flatten(), (dh[None].T * (1 - dw)[None]).flatten(), (dh[None].T * dw[None]).flatten()]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        patch_pos_embeds = (self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]).sum(0)
        patch_pos_embeds = patch_pos_embeds.split([height * width for height, width in zip(grid_hs, grid_ws)])
        merge_size = self.config.spatial_merge_size
        result = []
        for pos_embed, frames, height, width in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            result.append(pos_embed.repeat(frames, 1).view(frames, height // merge_size, merge_size, width // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(result)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb.reshape(seq_len, -1), rotary_pos_emb.reshape(seq_len, -1)), dim=-1)
        cu_seqlens = F.pad(torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32), (1, 0), value=0)
        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=(emb.cos(), emb.sin()), **kwargs)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature_lists.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states))
        return qwen.BaseModelOutputWithDeepstackFeatures(last_hidden_state=hidden_states, pooler_output=self.merger(hidden_states), deepstack_features=deepstack_feature_lists)

    qwen.Qwen3VLVisionModel.rot_pos_emb = rot_pos_emb
    qwen.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    qwen.Qwen3VLVisionModel.forward = forward
    print('[agrinet] enabled Transformers 5.9.0 Qwen3-VL 5.8.1-vision compatibility layer', flush=True)
