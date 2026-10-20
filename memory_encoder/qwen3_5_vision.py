from __future__ import annotations

from pathlib import Path

import einops
import torch
from torch import nn

QWEN3_5_PATCH_SIZE = 16
QWEN3_5_TEMPORAL_PATCH_SIZE = 2
QWEN3_5_MERGE_SIZE = 2


class Qwen35VisionTowerEncoder(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        if feature_mean is not None and feature_std is not None:
            if feature_mean.shape != feature_std.shape:
                raise ValueError("feature_mean and feature_std must share shape")
            self.register_buffer("feature_mean", feature_mean.float())
            self.register_buffer(
                "feature_std",
                feature_std.float().clamp(min=1e-6),
            )

    @property
    def normalized(self) -> bool:
        return hasattr(self, "feature_mean")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype


def load_qwen3_5_vision_tower(
    tower_path: str | Path,
    stats_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> Qwen35VisionTowerEncoder:
    from safetensors.torch import load_file
    from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel

    tower_path = Path(tower_path)
    config = Qwen3_5VisionConfig.from_pretrained(tower_path)
    model = Qwen3_5VisionModel(config)
    state_dict = load_file(tower_path / "vision_tower.safetensors")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch_dtype)
    feature_mean = None
    feature_std = None
    if stats_path is not None:
        stats = torch.load(
            Path(stats_path),
            map_location="cpu",
            weights_only=True,
        )
        feature_mean = stats["mean"]
        feature_std = stats["std"]
    return Qwen35VisionTowerEncoder(model, feature_mean, feature_std).to(
        device=device
    )


def encode_qwen3_5_frames(
    tower: Qwen35VisionTowerEncoder,
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
    batch_size, num_frames, _, height, width = video.shape
    spatial_unit = QWEN3_5_PATCH_SIZE * QWEN3_5_MERGE_SIZE
    if height % spatial_unit != 0 or width % spatial_unit != 0:
        raise ValueError(
            f"frame size ({height}, {width}) must be divisible by {spatial_unit}"
        )
    frames = einops.rearrange(video, "b t c h w -> (b t) c h w")
    # normalize on the input device (CPU) to bit-match the HF processor,
    # which also normalizes on CPU; GPU fp32 division differs by ulps
    if frames.dtype == torch.uint8:
        frames = frames.float() / 127.5 - 1.0
    frames = frames.to(device=tower.device, dtype=torch.float32)
    patches = einops.rearrange(
        frames,
        "n c (g1 m1 p1) (g2 m2 p2) -> n g1 g2 m1 m2 c p1 p2",
        m1=QWEN3_5_MERGE_SIZE,
        m2=QWEN3_5_MERGE_SIZE,
        p1=QWEN3_5_PATCH_SIZE,
        p2=QWEN3_5_PATCH_SIZE,
    )
    pixel_values = einops.repeat(
        patches,
        "n g1 g2 m1 m2 c p1 p2 -> n (g1 g2 m1 m2) (c t p1 p2)",
        t=QWEN3_5_TEMPORAL_PATCH_SIZE,
    )
    grid_height = height // QWEN3_5_PATCH_SIZE
    grid_width = width // QWEN3_5_PATCH_SIZE
    merged_height = grid_height // QWEN3_5_MERGE_SIZE
    merged_width = grid_width // QWEN3_5_MERGE_SIZE
    feature_parts = []
    for start_index in range(0, frames.shape[0], encode_batch_size):
        chunk = pixel_values[start_index:start_index + encode_batch_size]
        chunk_size = chunk.shape[0]
        grid_thw = torch.tensor(
            [[1, grid_height, grid_width]] * chunk_size,
            dtype=torch.long,
            device=chunk.device,
        )
        output = tower.model(
            chunk.to(dtype=tower.dtype),
            grid_thw=grid_thw,
            return_dict=True,
        )
        features = einops.rearrange(
            output.pooler_output,
            "(n h w) c -> n c h w",
            n=chunk_size,
            h=merged_height,
            w=merged_width,
        )
        feature_parts.append(features)
    latents = torch.cat(feature_parts, dim=0)
    if tower.normalized:
        latents = (
            (latents.float() - tower.feature_mean.view(1, -1, 1, 1))
            / tower.feature_std.view(1, -1, 1, 1)
        ).to(dtype=latents.dtype)
    return einops.rearrange(
        latents,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames,
    )
