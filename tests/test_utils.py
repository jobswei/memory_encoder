from __future__ import annotations

from pathlib import Path

import pytest
import torch

from memory_encoder.utils import load_video, to_uint8


def test_to_uint8_contracts_dynamic_range() -> None:
    video = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    frames = to_uint8(video)

    assert frames.dtype == torch.uint8
    assert frames.tolist() == [0, 128, 255]


def test_load_video_rejects_invalid_requests(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"

    with pytest.raises(FileNotFoundError):
        load_video(video_path, 2)

    video_path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="num_frames must be at least 2"):
        load_video(video_path, 1)
