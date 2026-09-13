from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn as nn
import yaml

from .config import AnchorMotionConfig


def _load_checkpoint_config(checkpoint_directory: Path) -> dict[str, Any]:
    config_paths = [
        checkpoint_directory / "config.yaml",
        checkpoint_directory.parent / "config.yaml",
    ]
    for config_path in config_paths:
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return config.get("model", config)
    raise FileNotFoundError(
        "Missing model config; expected config.yaml in "
        f"{checkpoint_directory}, or in its parent directory"
    )


class SpatialCoordinateEncoder(nn.Module):

    def __init__(self, hidden_dim: int, num_harmonics: int) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics
        self.projection = nn.Linear(
            2 * (1 + 2 * num_harmonics),
            hidden_dim,
        )

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        frequencies = (
            torch.arange(
                1,
                self.num_harmonics + 1,
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            * torch.pi
        )
        axis_features = []
        for axis_index in range(coordinates.shape[-1]):
            axis = coordinates[..., axis_index : axis_index + 1]
            axis_features.append(
                torch.cat(
                    [
                        axis,
                        torch.sin(axis * frequencies),
                        torch.cos(axis * frequencies),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(axis_features, dim=-1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(coordinates))

    def grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height_axis = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        width_axis = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        grid = torch.stack(
            torch.meshgrid(height_axis, width_axis, indexing="ij"),
            dim=-1,
        )
        grid = einops.rearrange(grid, "h w c -> (h w) c")
        return self.forward(
            einops.repeat(grid, "n c -> b n c", b=batch_size)
        )


class FrameTimeEncoder(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=frame_indices.device, dtype=frame_indices.dtype)
            / half_dim
        )
        angles = frame_indices[..., None] * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.hidden_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])],
                dim=-1,
            )
        return embedding


class MotionEncoderBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_queries = nn.LayerNorm(hidden_dim)
        self.cross_norm_memory = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.query_norm(queries)
        queries = queries + self.query_self_attention(
            normalized,
            normalized,
            normalized,
        )[0]
        normalized_queries = self.cross_norm_queries(queries)
        normalized_memory = self.cross_norm_memory(memory)
        queries = queries + self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
        )[0]
        return queries + self.feed_forward(queries)


class AnchorContextProjection(nn.Module):

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        context_size: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((context_size, context_size)),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.SiLU(),
        )

    def forward(self, anchor_latent: torch.Tensor) -> torch.Tensor:
        features = self.network(anchor_latent)
        return einops.rearrange(
            features,
            "b d h w -> b (h w) d",
        )


class CausalTemporalMotionBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        anchor_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        context = torch.cat([anchor_tokens, queries], dim=1)
        normalized_queries = self.query_norm(queries)
        normalized_context = self.context_norm(context)
        hidden = queries + self.attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return hidden + self.feed_forward(hidden)


class AnchorMotionEncoder(nn.Module):

    def __init__(self, config: AnchorMotionConfig) -> None:
        super().__init__()
        self.num_motion_tokens = config.num_motion_tokens
        self.anchor_context_size = config.anchor_context_size
        self.input_projection = nn.Sequential(
            nn.Conv2d(config.latent_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, config.hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.SiLU(),
        )
        self.position_encoder = SpatialCoordinateEncoder(
            hidden_dim=config.hidden_dim,
            num_harmonics=config.num_position_harmonics,
        )
        self.motion_queries = nn.Embedding(
            config.num_motion_tokens,
            config.hidden_dim,
        )
        self.frame_time_encoder = FrameTimeEncoder(config.hidden_dim)
        self.blocks = nn.ModuleList(
            MotionEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.temporal_attention = config.temporal_attention
        self.temporal_insertion_indices = tuple(
            config.temporal_insertion_indices,
        )
        if self.temporal_attention != "none":
            self.anchor_context_projection = AnchorContextProjection(
                latent_channels=config.latent_channels,
                hidden_dim=config.hidden_dim,
                context_size=config.anchor_context_size,
            )
            self.temporal_blocks = nn.ModuleList(
                CausalTemporalMotionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in self.temporal_insertion_indices
            )

    @staticmethod
    def _build_causal_attention_mask(
        num_targets: int,
        num_motion_tokens: int,
        num_anchor_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_frames = torch.arange(num_targets, device=device)
        query_frames = einops.repeat(
            query_frames,
            "t -> (t k)",
            k=num_motion_tokens,
        )
        key_frames = torch.arange(num_targets, device=device)
        key_frames = einops.repeat(
            key_frames,
            "t -> (t k)",
            k=num_motion_tokens,
        )
        anchor_frames = torch.full(
            (num_anchor_tokens,),
            -1,
            device=device,
        )
        key_frames = torch.cat([anchor_frames, key_frames])
        return key_frames[None, :] > query_frames[:, None]

    @staticmethod
    def _build_context_padding_mask(
        target_padding_mask: torch.Tensor | None,
        num_motion_tokens: int,
        num_anchor_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if target_padding_mask is None:
            return None
        motion_padding = einops.repeat(
            target_padding_mask.to(device=device, dtype=torch.bool),
            "b t -> b (t k)",
            k=num_motion_tokens,
        )
        anchor_padding = torch.zeros(
            (target_padding_mask.shape[0], num_anchor_tokens),
            device=device,
            dtype=torch.bool,
        )
        return torch.cat([anchor_padding, motion_padding], dim=1)

    def forward(
        self,
        anchor_latent: torch.Tensor,
        target_latents: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_targets, _, height, width = target_latents.shape
        differences = target_latents - anchor_latent[:, None]
        differences = einops.rearrange(
            differences,
            "b t c h w -> (b t) c h w",
        )
        features = self.input_projection(differences)
        feature_height, feature_width = features.shape[-2:]
        tokens = einops.rearrange(features, "b d h w -> b (h w) d")
        tokens = tokens + self.position_encoder.grid(
            batch_size=tokens.shape[0],
            height=feature_height,
            width=feature_width,
            device=tokens.device,
            dtype=tokens.dtype,
        )

        queries = einops.repeat(
            self.motion_queries.weight.to(dtype=tokens.dtype),
            "k d -> (b t) k d",
            b=batch_size,
            t=num_targets,
        )
        if frame_indices is None:
            frame_indices = torch.arange(
                1,
                num_targets + 1,
                device=target_latents.device,
            )[None, :].expand(batch_size, -1)
        frame_indices = frame_indices.to(device=target_latents.device, dtype=tokens.dtype)
        frame_embeddings = self.frame_time_encoder(frame_indices)
        frame_embeddings = einops.rearrange(
            frame_embeddings,
            "b t d -> (b t) 1 d",
        )
        queries = queries + frame_embeddings
        anchor_tokens = None
        attention_mask = None
        context_padding_mask = None
        if self.temporal_attention != "none":
            anchor_tokens = self.anchor_context_projection(anchor_latent)
            attention_mask = self._build_causal_attention_mask(
                num_targets=num_targets,
                num_motion_tokens=self.num_motion_tokens,
                num_anchor_tokens=anchor_tokens.shape[1],
                device=queries.device,
            )
            anchor_tokens = anchor_tokens + self.position_encoder.grid(
                batch_size=batch_size,
                height=self.anchor_context_size,
                width=self.anchor_context_size,
                device=anchor_tokens.device,
                dtype=anchor_tokens.dtype,
            )
            context_padding_mask = self._build_context_padding_mask(
                target_padding_mask,
                num_motion_tokens=self.num_motion_tokens,
                num_anchor_tokens=anchor_tokens.shape[1],
                device=queries.device,
            )

        temporal_block_index = 0
        for layer_index, block in enumerate(self.blocks):
            queries = block(queries, tokens)
            if layer_index not in self.temporal_insertion_indices:
                continue
            motion_queries = einops.rearrange(
                queries,
                "(b t) k d -> b (t k) d",
                b=batch_size,
                t=num_targets,
            )
            motion_queries = self.temporal_blocks[temporal_block_index](
                motion_queries,
                anchor_tokens,
                attention_mask,
                context_padding_mask,
            )
            temporal_block_index += 1
            queries = einops.rearrange(
                motion_queries,
                "b (t k) d -> (b t) k d",
                b=batch_size,
                t=num_targets,
            )
        return einops.rearrange(
            queries,
            "(b t) k d -> b t k d",
            b=batch_size,
            t=num_targets,
        )


class AnchorDecoderBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        queries = queries + self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
        )[0]
        return queries + self.feed_forward(queries)


class AnchorMotionDecoder(nn.Module):

    def __init__(self, config: AnchorMotionConfig) -> None:
        super().__init__()
        self.anchor_projection = nn.Conv2d(
            config.latent_channels,
            config.hidden_dim,
            3,
            padding=1,
        )
        self.position_encoder = SpatialCoordinateEncoder(
            hidden_dim=config.hidden_dim,
            num_harmonics=config.num_position_harmonics,
        )
        self.blocks = nn.ModuleList(
            AnchorDecoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_channels),
        )

    def forward(
        self,
        anchor_latent: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_targets = motion_tokens.shape[:2]
        height, width = anchor_latent.shape[-2:]
        anchor_features = self.anchor_projection(anchor_latent)
        anchor_tokens = einops.rearrange(
            anchor_features,
            "b d h w -> b (h w) d",
        )
        anchor_tokens = anchor_tokens + self.position_encoder.grid(
            batch_size=batch_size,
            height=height,
            width=width,
            device=anchor_tokens.device,
            dtype=anchor_tokens.dtype,
        )
        anchor_tokens = einops.repeat(
            anchor_tokens,
            "b n d -> (b t) n d",
            t=num_targets,
        )
        motion_tokens = einops.rearrange(
            motion_tokens,
            "b t k d -> (b t) k d",
        )
        hidden = anchor_tokens
        for block in self.blocks:
            hidden = block(hidden, motion_tokens)
        residuals = self.residual_head(hidden)
        return anchor_latent[:, None] + einops.rearrange(
            residuals,
            "(b t) (h w) c -> b t c h w",
            b=batch_size,
            h=height,
            w=width,
        )


class AnchorMotionAutoEncoder(nn.Module):

    def __init__(self, config: AnchorMotionConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.encoder = AnchorMotionEncoder(config)
        self.decoder = AnchorMotionDecoder(config)

    @staticmethod
    def _validate_config(config: AnchorMotionConfig) -> None:
        positive_fields = (
            "latent_channels",
            "hidden_dim",
            "num_motion_tokens",
            "num_layers",
            "num_heads",
            "num_position_harmonics",
            "anchor_context_size",
        )
        for field_name in positive_fields:
            if getattr(config, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.temporal_attention not in {"none", "anchor_conditioned"}:
            raise ValueError(
                "temporal_attention must be none or anchor_conditioned"
            )
        if config.temporal_attention == "none":
            return
        insertion_indices = config.temporal_insertion_indices
        if not insertion_indices or len(set(insertion_indices)) != len(
            insertion_indices
        ):
            raise ValueError(
                "temporal_insertion_indices must contain unique values",
            )
        if any(
            index < 0 or index >= config.num_layers
            for index in insertion_indices
        ):
            raise ValueError("temporal insertion index is out of range")

    def encode(
        self,
        anchor_latent: torch.Tensor,
        target_latents: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor_latent.dim() != 4 or target_latents.dim() != 5:
            raise ValueError(
                "anchor_latent must have shape [B, C, H, W] and "
                "target_latents must have shape [B, T, C, H, W]"
            )
        if target_latents.shape[2:] != anchor_latent.shape[1:]:
            raise ValueError("target_latents and anchor_latent shapes must match")
        if frame_indices is not None and frame_indices.shape != target_latents.shape[:2]:
            raise ValueError("frame_indices must have shape [B, T]")
        if (
            target_padding_mask is not None
            and target_padding_mask.shape != target_latents.shape[:2]
        ):
            raise ValueError("target_padding_mask must have shape [B, T]")
        return self.encoder(
            anchor_latent,
            target_latents,
            frame_indices,
            target_padding_mask,
        )

    def decode(
        self,
        anchor_latent: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_latent.dim() != 4 or motion_tokens.dim() != 4:
            raise ValueError(
                "anchor_latent must have shape [B, C, H, W] and "
                "motion_tokens must have shape [B, T, K, D]"
            )
        return self.decoder(anchor_latent, motion_tokens)

    def forward(
        self,
        anchor_latent: torch.Tensor,
        target_latents: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        motion_tokens = self.encode(
            anchor_latent,
            target_latents,
            frame_indices,
            target_padding_mask,
        )
        reconstructed_latents = self.decode(anchor_latent, motion_tokens)
        return {
            "motion_tokens": motion_tokens,
            "reconstructed_latents": reconstructed_latents,
        }

    def save_pretrained(self, directory: str | Path) -> None:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "config.yaml").write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        try:
            from safetensors.torch import save_file

            save_file(self.state_dict(), output_directory / "model.safetensors")
        except ImportError:
            torch.save(self.state_dict(), output_directory / "model.pt")

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> AnchorMotionAutoEncoder:
        checkpoint_directory = Path(directory)
        config = AnchorMotionConfig.from_dict(
            _load_checkpoint_config(checkpoint_directory)
        )
        model = cls(config)
        safetensors_path = checkpoint_directory / "model.safetensors"
        torch_path = checkpoint_directory / "model.pt"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise RuntimeError(
                    "Checkpoint requires safetensors. Install memory-encoder[safetensors]."
                ) from error
            state_dict = load_file(safetensors_path, device=str(map_location))
        elif torch_path.exists():
            state_dict = torch.load(
                torch_path,
                map_location=map_location,
                weights_only=True,
            )
        else:
            raise FileNotFoundError(f"Missing weights in {checkpoint_directory}")
        model.load_state_dict(state_dict, strict=True)
        return model
