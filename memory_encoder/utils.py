from __future__ import annotations

from pathlib import Path

import torch


__all__ = ["load_video", "to_uint8"]


def load_video(
    video_path: str | Path,
    num_frames: int,
    return_fps: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    try:
        from decord import VideoReader
    except ImportError as error:
        raise RuntimeError(
            "Video loading requires decord. Install memory-encoder[example]."
        ) from error

    video_reader = VideoReader(str(video_path), num_threads=0)
    if len(video_reader) < num_frames:
        raise ValueError(
            f"Video has {len(video_reader)} frames, but {num_frames} were requested"
        )
    frames = torch.from_numpy(
        video_reader.get_batch(list(range(num_frames))).asnumpy()
    ).to(torch.uint8)
    if return_fps:
        return frames, float(video_reader.get_avg_fps())
    return frames


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    return (
        video.add(1)
        .div(2)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
    )
