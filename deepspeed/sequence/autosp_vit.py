import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed.comm as dist

class UlyssesSPViTAttention(nn.Module):

    def __init__(self, attn: nn.Module, process_group, has_cls_token: bool = True) -> None:
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.has_cls_token = has_cls_token

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        bs, local_seq_len, hidden_dim = hidden_states.shape

        if self.has_cls_token:
            cls_token = hidden_states[:, :1, :]
            local_patches = hidden_states[:, 1:, :]
        else:
            local_patches = hidden_states

        local_patch_len = local_patches.shape[1]

        len_bufs = [torch.zeros(1, dtype=torch.long, device=local_patches.device) for _ in range(self.world_size)]
        dist.all_gather(len_bufs,
                        torch.tensor([local_patch_len], dtype=torch.long, device=local_patches.device),
                        group=self.process_group)
        all_lens = [int(t.item()) for t in len_bufs]
        max_local_len = max(all_lens)

        pad_len = max_local_len - local_patch_len
        if pad_len > 0:
            local_patches_padded = F.pad(local_patches, (0, 0, 0, pad_len))
        else:
            local_patches_padded = local_patches

        gathered = [
            torch.zeros(bs, max_local_len, hidden_dim, dtype=local_patches.dtype, device=local_patches.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, local_patches_padded.contiguous(), group=self.process_group)

        real_parts = [gathered[r][:, :all_lens[r], :] for r in range(self.world_size)]
        full_patches = torch.cat(real_parts, dim=1)

        if self.has_cls_token:
            full_input = torch.cat([cls_token, full_patches], dim=1)
        else:
            full_input = full_patches

        attn_out = self.attn(full_input, **kwargs)

        if isinstance(attn_out, (tuple, list)):
            full_out, *extra = attn_out
        else:
            full_out = attn_out
            extra = []

        if self.has_cls_token:
            cls_out = full_out[:, :1, :]
            patch_out = full_out[:, 1:, :]
        else:
            patch_out = full_out

        rank = dist.get_rank(self.process_group)
        start = sum(all_lens[:rank])
        local_out = patch_out[:, start:start + local_patch_len, :].contiguous()

        if self.has_cls_token:
            local_out = torch.cat([cls_out, local_out], dim=1)

        if extra:
            return (local_out, *extra)
        return local_out
