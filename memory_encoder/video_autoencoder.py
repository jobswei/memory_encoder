from __future__ import annotations

from pathlib import Path

import einops
import torch
from torch import nn

from .anchor_motion import AnchorMotionAutoEncoder
from .svd_vae import decode_svd_latents, encode_svd_frames, load_svd_vae


class AnchorMotionVideoAutoEncoder(nn.Module):

    def __init__(
        self,
        anchor_motion: AnchorMotionAutoEncoder,
        vae: nn.Module,
        encode_batch_size: int = 32,
    ) -> None:
        super().__init__()
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be positive")
        self.anchor_motion = anchor_motion.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False)
        self.encode_batch_size = encode_batch_size

    @classmethod
    def from_pretrained(
        cls,
        anchor_checkpoint: str | Path,
        vae_path: str | Path,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        encode_batch_size: int = 32,
    ) -> AnchorMotionVideoAutoEncoder:
        anchor_motion = AnchorMotionAutoEncoder.from_pretrained(
            anchor_checkpoint,
            map_location=device,
        )
        vae = load_svd_vae(
            vae_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        return cls(
            anchor_motion.to(device=device, dtype=torch_dtype),
            vae,
            encode_batch_size=encode_batch_size,
        )

    def _prepare_frames(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 3 and video.shape[-1] == 3:
            frames = einops.rearrange(video, "t h w c -> 1 t c h w")
        elif video.ndim == 4:
            if video.shape[-1] == 3:
                frames = einops.rearrange(video, "t h w c -> 1 t c h w")
            elif video.shape[1] == 3:
                frames = einops.rearrange(video, "t c h w -> 1 t c h w")
            else:
                raise ValueError(
                    "Video must have shape [T,H,W,C], [T,3,H,W], "
                    "[B,T,H,W,C], or [B,T,3,H,W]"
                )
        elif video.ndim == 5 and video.shape[-1] == 3:
            frames = einops.rearrange(video, "b t h w c -> b t c h w")
        elif video.ndim == 5 and video.shape[2] == 3:
            frames = video
        else:
            raise ValueError(
                "Video must have shape [T,H,W,C], [T,3,H,W], "
                "[B,T,H,W,C], or [B,T,3,H,W]"
            )
        if frames.shape[1] < 2:
            raise ValueError("Video must contain at least two frames")
        frames = frames.to(device=self.vae.device)
        return frames

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        frames = self._prepare_frames(video)
        latents = encode_svd_frames(
            self.vae,
            frames,
            self.encode_batch_size,
        )
        motion_tokens = self.anchor_motion.encode(
            latents[:, 0],
            latents[:, 1:],
        )
        return {
            "anchor_latent": latents[:, 0].contiguous(),
            "motion_tokens": motion_tokens.contiguous(),
        }

    @torch.inference_mode()
    def decode_video(
        self,
        anchor_latent: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        target_latents = self.anchor_motion.decode(
            anchor_latent,
            motion_tokens,
        )
        latents = torch.cat(
            [anchor_latent[:, None], target_latents],
            dim=1,
        )
        return decode_svd_latents(self.vae, latents)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_video(video)
