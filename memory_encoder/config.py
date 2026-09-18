from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnchorMotionConfig:
    latent_channels: int = 4
    hidden_dim: int = 256
    num_motion_tokens: int = 8
    num_layers: int = 4
    num_heads: int = 8
    num_position_harmonics: int = 8
    dropout: float = 0.0
    temporal_attention: str = "none"
    temporal_insertion_indices: list[int] = field(
        default_factory=lambda: [2, 3],
    )
    anchor_context_size: int = 4
    query_source: str = "delta"
    auxiliary_heads: list[str] = field(default_factory=list)
    dynamic_mask_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AnchorMotionConfig:
        field_names = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in field_names})


VALID_QUERY_SOURCES = {"delta", "delta_target_anchor"}
VALID_AUXILIARY_HEADS = {"dynamic_mask", "anchor_flow"}
