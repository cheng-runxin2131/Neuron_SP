import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed.comm as dist

_DEFAULT_IMAGE_TOKEN_ID = -200

class ModalityFusionSPAdapter(nn.Module):

    def __init__(self, projection: nn.Module, process_group, image_token_id: int = _DEFAULT_IMAGE_TOKEN_ID) -> None:
        super().__init__()
        self.projection = projection
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.image_token_id = image_token_id

    def forward(self, visual_features: torch.Tensor, text_embeds: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        visual_embeds = self.projection(visual_features)

        parts = [torch.zeros_like(visual_embeds) for _ in range(self.world_size)]
        dist.all_gather(parts, visual_embeds.contiguous(), group=self.process_group)
        full_visual = torch.cat(parts, dim=1)

        fused = self._splice_visual_into_text(text_embeds, full_visual, input_ids)

        total_len = fused.shape[1]
        pad = (self.world_size - total_len % self.world_size) % self.world_size
        if pad > 0:
            fused = F.pad(fused, (0, 0, 0, pad))

        rank = dist.get_rank(self.process_group)
        local_len = fused.shape[1] // self.world_size
        return fused[:, rank * local_len:(rank + 1) * local_len, :].contiguous()

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__}._splice_visual_into_text is not implemented. "
                                  "Subclass ModalityFusionSPAdapter and override this method to match "
                                  "your model's prepare_inputs_embeds logic.")

class LlavaFusionAdapter(ModalityFusionSPAdapter):

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        bs, text_len, hidden = text_embeds.shape
        device = text_embeds.device

        fused_samples = []
        for i in range(bs):
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            num_images = img_pos.numel()

            if num_images == 0:
                fused_samples.append(text_embeds[i])
                continue

            visual_chunks = torch.chunk(visual_embeds[i], num_images, dim=0)

            segments = []
            prev = 0
            for j, pos in enumerate(img_pos.tolist()):
                if pos > prev:
                    segments.append(text_embeds[i, prev:pos])
                segments.append(visual_chunks[j])
                prev = pos + 1

            if prev < text_len:
                segments.append(text_embeds[i, prev:])

            fused_samples.append(torch.cat(segments, dim=0))

        max_len = max(s.shape[0] for s in fused_samples)
        out = torch.zeros(bs, max_len, hidden, dtype=text_embeds.dtype, device=device)
        for i, s in enumerate(fused_samples):
            out[i, :s.shape[0]] = s
        return out

class InternVLFusionAdapter(ModalityFusionSPAdapter):

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        out = text_embeds.clone()
        bs = text_embeds.shape[0]

        for i in range(bs):
            ctx_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if ctx_pos.numel() == 0:
                continue
            out[i, ctx_pos] = visual_embeds[i, :ctx_pos.numel()]

        return out

class Qwen2VLFusionAdapter(nn.Module):

    def __init__(self, projection: nn.Module, process_group, vision_start_token_id: int,
                 vision_end_token_id: int) -> None:
        super().__init__()
        self.projection = projection
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    def forward(self, visual_features: torch.Tensor, text_embeds: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        visual_embeds = self.projection(visual_features)

        parts = [torch.zeros_like(visual_embeds) for _ in range(self.world_size)]
        dist.all_gather(parts, visual_embeds.contiguous(), group=self.process_group)
        full_visual = torch.cat(parts, dim=1)

        fused = self._splice_visual_into_text(text_embeds, full_visual, input_ids)

        total_len = fused.shape[1]
        pad = (self.world_size - total_len % self.world_size) % self.world_size
        if pad > 0:
            fused = F.pad(fused, (0, 0, 0, pad))

        rank = dist.get_rank(self.process_group)
        local_len = fused.shape[1] // self.world_size
        return fused[:, rank * local_len:(rank + 1) * local_len, :].contiguous()

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        out = text_embeds.clone()
        bs = text_embeds.shape[0]

        for i in range(bs):
            start_pos = (input_ids[i] == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[i] == self.vision_end_token_id).nonzero(as_tuple=True)[0]

            if start_pos.numel() == 0:
                continue

            inner_positions = []
            for s, e in zip(start_pos.tolist(), end_pos.tolist()):
                inner_positions.extend(range(s + 1, e))

            if not inner_positions:
                continue

            inner_pos = torch.tensor(inner_positions, dtype=torch.long, device=text_embeds.device)
            out[i, inner_pos] = visual_embeds[i, :len(inner_positions)]

        return out
