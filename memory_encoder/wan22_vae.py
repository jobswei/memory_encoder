from __future__ import annotations

from pathlib import Path

import einops
import torch

from .wan_video_vae import WanVideoVAE38


def load_wan22_vae(
    vae_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    path = Path(vae_path)
    if not path.is_file():
        raise ValueError(f"Wan 2.2 VAE checkpoint is required: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    input_conv_weight = checkpoint.get("encoder.conv1.weight")
    quant_conv_weight = checkpoint.get("conv1.weight")
    if input_conv_weight is None or quant_conv_weight is None:
        raise ValueError(f"Unsupported Wan 2.2 VAE checkpoint: {path}")
    if input_conv_weight.shape[1] != 12:
        raise ValueError(
            "Only official Wan 2.2 VAE checkpoints are supported"
        )
    vae = WanVideoVAE38(
        z_dim=int(quant_conv_weight.shape[0]) // 2,
        dim=int(input_conv_weight.shape[0]),
    )
    state_dict = {
        f"model.{key}": value for key, value in checkpoint.items()
    }
    vae.load_state_dict(state_dict, strict=True)
    return vae.to(device=device, dtype=torch_dtype).eval().requires_grad_(
        False
    )


def wan22_temporal_compression(vae: torch.nn.Module) -> int:
    del vae
    return 4


def _vae_device(vae: torch.nn.Module) -> torch.device:
    if hasattr(vae, "device"):
        return torch.device(vae.device)
    parameter = next(vae.parameters(), None)
    if parameter is not None:
        return parameter.device
    raise ValueError("Failed to resolve the Wan 2.2 VAE device")


def _vae_dtype(vae: torch.nn.Module) -> torch.dtype:
    if hasattr(vae, "dtype"):
        return vae.dtype
    parameter = next(vae.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    raise ValueError("Failed to resolve the Wan 2.2 VAE dtype")


def encode_wan22_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
    encode_batch_size: int = 1,
) -> torch.Tensor:
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if video.ndim != 5:
        raise ValueError("video must have shape [B,T,H,W,3] or [B,T,3,H,W]")
    if video.shape[-1] == 3:
        frames = einops.rearrange(video, "b t h w c -> b c t h w")
    elif video.shape[2] == 3:
        frames = einops.rearrange(video, "b t c h w -> b c t h w")
    else:
        raise ValueError("video must have three color channels")
    if frames.shape[-2] % 16 or frames.shape[-1] % 16:
        raise ValueError("Wan 2.2 video height and width must divide 16")
    device = _vae_device(vae)
    dtype = _vae_dtype(vae)
    frames = frames.to(device=device)
    if frames.dtype == torch.uint8:
        frames = frames.float().div(127.5).sub(1.0)
    frames = frames.to(dtype=dtype)
    latents = []
    for start_index in range(0, frames.shape[0], encode_batch_size):
        latent_chunk = vae.encode(
            frames[start_index:start_index + encode_batch_size]
        )
        latents.append(latent_chunk)
    latent_video = torch.cat(latents, dim=0)
    return einops.rearrange(
        latent_video,
        "b c t h w -> b t c h w",
    )


def decode_wan22_video(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B,T,C,H,W]")
    latent_video = einops.rearrange(latents, "b t c h w -> b c t h w")
    latent_video = latent_video.to(device=_vae_device(vae), dtype=_vae_dtype(vae))
    decoded_video = vae.decode(latent_video)
    return einops.rearrange(
        decoded_video,
        "b c t h w -> b t h w c",
    )
