from __future__ import annotations

from pathlib import Path

import einops
import torch


def load_svd_vae(
    vae_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    from diffusers import AutoencoderKLTemporalDecoder

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(vae_path),
        subfolder="vae",
        torch_dtype=torch_dtype,
    )
    return vae.to(device=device)


def encode_svd_frames(
    vae: torch.nn.Module,
    video: torch.Tensor,
    encode_batch_size: int = 32,
) -> torch.Tensor:
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if video.ndim != 5:
        raise ValueError("video must have shape [B,T,3,H,W] or [B,T,H,W,3]")
    if video.shape[-1] == 3 and video.shape[2] != 3:
        video = einops.rearrange(video, "b t h w c -> b t c h w")
    elif video.shape[2] != 3:
        raise ValueError("video must have shape [B,T,3,H,W] or [B,T,H,W,3]")
    frames = einops.rearrange(video, "b t c h w -> (b t) c h w")
    frames = frames.to(device=vae.device)
    if frames.dtype == torch.uint8:
        frames = frames.float() / 127.5 - 1.0
    frames = frames.to(dtype=vae.dtype)
    latent_parts = []
    for start_index in range(0, frames.shape[0], encode_batch_size):
        frame_chunk = frames[start_index:start_index + encode_batch_size]
        latent_chunk = vae.encode(frame_chunk).latent_dist.mode()
        latent_parts.append(latent_chunk.mul(vae.config.scaling_factor))
    flat_latents = torch.cat(latent_parts, dim=0)
    return einops.rearrange(
        flat_latents,
        "(b t) c h w -> b t c h w",
        b=video.shape[0],
        t=video.shape[1],
    )


def decode_svd_latents(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B,T,C,H,W]")
    batch_size, num_frames = latents.shape[:2]
    flat_latents = einops.rearrange(latents, "b t c h w -> (b t) c h w")
    flat_latents = flat_latents.to(
        device=vae.device,
        dtype=vae.dtype,
    ) / vae.config.scaling_factor
    decoded = vae.decode(flat_latents, num_frames=num_frames).sample
    return einops.rearrange(
        decoded,
        "(b t) c h w -> b t h w c",
        b=batch_size,
        t=num_frames,
    )
