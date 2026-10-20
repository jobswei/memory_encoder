from __future__ import annotations

import json
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


def _resolve_vision_config(tower_path: Path):
    """Vision config from a native Qwen3.5 checkpoint or a wrapped one.

    Supports both ``vision_config`` (native Qwen3.5 layout) and
    ``backbone_config.vision_config`` (wrapped layouts such as SimpleMemVLA,
    where the full Qwen3.5 model sits under a ``backbone.`` prefix).
    """
    from transformers import AutoConfig, Qwen3_5VisionConfig

    vision_config = None
    try:
        config = AutoConfig.from_pretrained(tower_path)
    except Exception:
        config = None
    if config is not None:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None and not isinstance(
            vision_config, Qwen3_5VisionConfig
        ):
            vision_config = None
    if vision_config is None:
        raw = json.loads(
            (tower_path / "config.json").read_text(encoding="utf-8")
        )
        vision_dict = raw.get("vision_config")
        if vision_dict is None:
            backbone_dict = raw.get("backbone_config")
            if isinstance(backbone_dict, dict):
                vision_dict = backbone_dict.get("vision_config")
        if vision_dict is None:
            raise ValueError(
                f"no vision_config found in {tower_path} (looked for "
                "'vision_config' and 'backbone_config.vision_config')"
            )
        vision_config = Qwen3_5VisionConfig.from_dict(vision_dict)
    return vision_config


def _resolve_visual_prefix(weight_map: dict) -> str:
    """Key prefix of the vision tower inside the checkpoint.

    Native Qwen3.5 checkpoints use ``model.visual.``; wrapped checkpoints
    such as SimpleMemVLA use ``backbone.model.visual.``.
    """
    for candidate in ("model.visual.", "backbone.model.visual."):
        if any(key.startswith(candidate) for key in weight_map):
            return candidate
    for key in weight_map:
        if ".visual." in key:
            return key[: key.index(".visual.") + len(".visual.")]
    raise ValueError(
        "no vision-tower weights ('*visual.*') found in the checkpoint"
    )


def _load_vision_tower_weights(tower_path: Path):
    tower_path = Path(tower_path)
    from safetensors import safe_open
    from safetensors.torch import load_file

    extracted_path = tower_path / "vision_tower.safetensors"
    if extracted_path.is_file():
        from transformers import Qwen3_5VisionConfig

        config = Qwen3_5VisionConfig.from_pretrained(tower_path)
        return config, load_file(extracted_path)
    config = _resolve_vision_config(tower_path)
    index_path = tower_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    elif (tower_path / "model.safetensors").is_file():
        with safe_open(
            tower_path / "model.safetensors", framework="pt"
        ) as handle:
            weight_map = {
                key: "model.safetensors" for key in handle.keys()
            }
    else:
        raise FileNotFoundError(
            f"{tower_path} contains neither vision_tower.safetensors "
            "nor a full Qwen3.5 checkpoint (model.safetensors[.index.json])"
        )
    prefix = _resolve_visual_prefix(weight_map)
    shard_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shard_keys.setdefault(shard, []).append(key)
    if not shard_keys:
        raise ValueError(
            f"no '{prefix}*' weights found in {tower_path}"
        )
    state_dict = {}
    for shard, keys in shard_keys.items():
        with safe_open(tower_path / shard, framework="pt") as handle:
            for key in keys:
                state_dict[key.removeprefix(prefix)] = handle.get_tensor(key)
    return config, state_dict


def load_qwen3_5_vision_tower(
    tower_path: str | Path,
    stats_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> Qwen35VisionTowerEncoder:
    from transformers import Qwen3_5VisionModel

    config, state_dict = _load_vision_tower_weights(Path(tower_path))
    model = Qwen3_5VisionModel(config)
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
