from __future__ import annotations

from pathlib import Path

import einops
import torch


def load_wan_vae(
    vae_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    from diffusers import AutoencoderKLWan
    from diffusers.loaders.single_file_utils import (
        convert_wan_vae_to_diffusers,
    )

    path = Path(vae_path)
    if path.is_dir():
        subfolder = None if (path / "config.json").exists() else "vae"
        vae = AutoencoderKLWan.from_pretrained(
            str(path),
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        ).to(device=device)
        vae.upsampling_factor = (
            vae.config.scale_factor_spatial
            if vae.config.scale_factor_spatial is not None
            else vae.spatial_compression_ratio
        )
        vae.z_dim = vae.config.z_dim
        return vae.eval()

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    input_conv_weight = checkpoint.get("encoder.conv1.weight")
    quant_conv_weight = checkpoint.get("conv1.weight")
    if input_conv_weight is None or quant_conv_weight is None:
        raise ValueError(f"Unsupported Wan VAE checkpoint: {path}")
    if input_conv_weight.shape[1] != 3:
        raise ValueError(
            "Only standard Wan 2.1 VAE checkpoints are supported; "
            "Wan 2.2 uses a different architecture"
        )
    vae = AutoencoderKLWan(
        base_dim=int(input_conv_weight.shape[0]),
        z_dim=int(quant_conv_weight.shape[0]) // 2,
    )
    state_dict = convert_wan_vae_to_diffusers(checkpoint)
    vae.load_state_dict(state_dict, strict=True)
    vae = vae.to(device=device, dtype=torch_dtype).eval()
    vae.upsampling_factor = (
        vae.config.scale_factor_spatial
        if vae.config.scale_factor_spatial is not None
        else vae.spatial_compression_ratio
    )
    vae.z_dim = vae.config.z_dim
    return vae


def wan_temporal_compression(vae: torch.nn.Module) -> int:
    compression = getattr(vae.config, "scale_factor_temporal", 4)
    if compression is None:
        return 4
    return int(compression)


def encode_wan_video(
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
    frames = frames.to(device=vae.device)
    if frames.dtype == torch.uint8:
        frames = frames.float().div(127.5).sub(1.0)
    frames = frames.to(dtype=vae.dtype)
    latents = []
    for start_index in range(0, frames.shape[0], encode_batch_size):
        latent_chunk = vae.encode(
            frames[start_index:start_index + encode_batch_size]
        ).latent_dist.mode()
        latents.append(latent_chunk)
    latent_video = torch.cat(latents, dim=0)
    mean, std = _wan_latent_stats(vae, latent_video)
    latent_video = latent_video.sub(mean).div(std)
    return einops.rearrange(
        latent_video,
        "b c t h w -> b t c h w",
    )


def decode_wan_video(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B,T,C,H,W]")
    latent_video = einops.rearrange(latents, "b t c h w -> b c t h w")
    latent_video = latent_video.to(device=vae.device, dtype=vae.dtype)
    mean, standard_deviation = _wan_latent_stats(vae, latent_video)
    latent_video = latent_video.mul(standard_deviation).add(mean)
    decoded_video = vae.decode(latent_video).sample
    return einops.rearrange(
        decoded_video,
        "b c t h w -> b t h w c",
    )


def _wan_latent_stats(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    return mean, std
