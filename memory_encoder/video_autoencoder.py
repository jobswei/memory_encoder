from __future__ import annotations

import json
from pathlib import Path

import einops
import torch
from torch import nn

from .anchor_motion import AnchorMotionAutoEncoder, _load_checkpoint_config
from .svd_vae import decode_svd_latents, encode_svd_frames, load_svd_vae
from .wan_vae import (
    decode_wan_video,
    encode_wan_video,
    load_wan_vae,
    wan_temporal_compression,
)
from .wan22_vae import (
    decode_wan22_video,
    encode_wan22_video,
    load_wan22_vae,
    wan22_temporal_compression,
)


SUPPORTED_VAE_TYPES = {"svd", "wan", "wan22"}


def _vae_device(vae: nn.Module) -> torch.device:
    parameter = next(vae.parameters(), None)
    if parameter is not None:
        return parameter.device
    return vae.device


def _detect_wan_checkpoint_type(vae_path: Path) -> str | None:
    suffix = vae_path.suffix.lower()
    if not vae_path.is_file() or suffix not in {".pth", ".pt"}:
        return None
    checkpoint = torch.load(vae_path, map_location="cpu", weights_only=True)
    input_conv_weight = checkpoint.get("encoder.conv1.weight")
    if input_conv_weight is None:
        return None
    input_channels = int(input_conv_weight.shape[1])
    if input_channels == 3:
        return "wan"
    if input_channels == 12:
        return "wan22"
    return None


def _detect_diffusers_vae_type(vae_path: Path) -> str | None:
    if not vae_path.is_dir():
        return None
    config_path = (
        vae_path / "config.json"
        if (vae_path / "config.json").exists()
        else vae_path / "vae" / "config.json"
    )
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_class = str(config.get("_class_name", ""))
    if model_class == "AutoencoderKLWan":
        return "wan"
    if model_class == "AutoencoderKLTemporalDecoder":
        return "svd"
    return None


def _resolve_vae_type(
    checkpoint_config: dict,
    vae_path: str | Path,
) -> str:
    configured_type = checkpoint_config.get("base_vae_type")
    if configured_type is not None:
        if configured_type not in SUPPORTED_VAE_TYPES:
            raise ValueError(f"Unsupported VAE type: {configured_type}")
        return configured_type

    path = Path(vae_path)
    detected_type = _detect_wan_checkpoint_type(path)
    if detected_type is None:
        detected_type = _detect_diffusers_vae_type(path)
    if detected_type is None:
        raise ValueError(
            "Failed to infer the base VAE type. Record model.base_vae_type "
            f"in config.yaml or provide a recognizable vae_path: {path}"
        )
    return detected_type


class AnchorMotionVideoAutoEncoder(nn.Module):

    def __init__(
        self,
        anchor_motion: AnchorMotionAutoEncoder,
        vae: nn.Module,
        encode_batch_size: int = 32,
        vae_type: str = "svd",
    ) -> None:
        super().__init__()
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be positive")
        self.anchor_motion = anchor_motion.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False)
        self.encode_batch_size = encode_batch_size
        if vae_type not in SUPPORTED_VAE_TYPES:
            raise ValueError(f"Unsupported VAE type: {vae_type}")
        self.vae_type = vae_type
        self.temporal_compression = (
            wan_temporal_compression(vae)
            if vae_type == "wan"
            else wan22_temporal_compression(vae)
            if vae_type == "wan22"
            else 1
        )

    @classmethod
    def from_pretrained(
        cls,
        anchor_checkpoint: str | Path,
        vae_path: str | Path,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        encode_batch_size: int = 32,
    ) -> AnchorMotionVideoAutoEncoder:
        checkpoint_config = _load_checkpoint_config(Path(anchor_checkpoint))
        vae_type = _resolve_vae_type(checkpoint_config, vae_path)
        anchor_motion = AnchorMotionAutoEncoder.from_pretrained(
            anchor_checkpoint,
            map_location=device,
        )
        vae_loaders = {
            "svd": load_svd_vae,
            "wan": load_wan_vae,
            "wan22": load_wan22_vae,
        }
        vae = vae_loaders[vae_type](
            vae_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        return cls(
            anchor_motion.to(device=device, dtype=torch_dtype),
            vae,
            vae_type=vae_type,
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
        frames = frames.to(device=_vae_device(self.vae))
        return frames

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        frames = self._prepare_frames(video)
        if self.vae_type == "svd":
            latents = encode_svd_frames(
                self.vae,
                frames,
                self.encode_batch_size,
            )
        elif self.vae_type == "wan":
            latents = encode_wan_video(
                self.vae,
                frames,
                self.encode_batch_size,
            )
        else:
            latents = encode_wan22_video(
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
        num_frames: int | None = None,
    ) -> torch.Tensor:
        target_latents = self.anchor_motion.decode(
            anchor_latent,
            motion_tokens,
        )
        latents = torch.cat(
            [anchor_latent[:, None], target_latents],
            dim=1,
        )
        if self.vae_type == "svd":
            video = decode_svd_latents(self.vae, latents)
        elif self.vae_type == "wan":
            video = decode_wan_video(self.vae, latents)
        else:
            video = decode_wan22_video(self.vae, latents)
        if num_frames is None:
            return video
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if num_frames <= video.shape[1]:
            return video[:, :num_frames]
        padded_last_frame = einops.repeat(
            video[:, -1:],
            "b t h w c -> b (t repeat) h w c",
            repeat=num_frames - video.shape[1],
        )
        return torch.cat(
            [video, padded_last_frame],
            dim=1,
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_video(video)
