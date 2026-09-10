from __future__ import annotations

import argparse
from pathlib import Path

import einops
import numpy as np
import torch
from PIL import Image, ImageDraw

from memory_encoder import AnchorMotionVideoAutoEncoder
from memory_encoder.utils import load_video, to_uint8


DEFAULT_VIDEO_PATH = Path(__file__).resolve().parent / "data" / "demo.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and decode a video with Anchor Motion."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Input video path. Default: {DEFAULT_VIDEO_PATH}",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of contiguous frames to read from the video.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Anchor Motion checkpoint directory.",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        required=True,
        help="Stable Video Diffusion checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work_dirs/results/anchor_motion_comparison.mp4"),
        help="Output comparison video path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Model dtype.",
    )
    return parser.parse_args()


def save_comparison_video(
    original_video: torch.Tensor,
    reconstructed_video: torch.Tensor,
    output_path: Path,
    fps: float,
) -> None:
    if output_path.suffix != ".mp4":
        raise ValueError("Comparison output must be an .mp4 file")
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "Saving comparison videos requires imageio and ffmpeg. "
            "Install memory-encoder[example]."
        ) from error

    reconstructed_uint8 = to_uint8(reconstructed_video)
    normalized_original = (
        original_video.to(reconstructed_video.device)
        .float()
        .div(127.5)
        .sub(1.0)
    )
    absolute_error = (
        reconstructed_video.subtract(normalized_original)
        .div(2)
        .abs()
        .mul(5)
        .clamp(0, 1)
        .mean(dim=-1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    absolute_error = einops.repeat(
        absolute_error,
        "t h w -> t h w rgb",
        rgb=3,
    )

    separator = torch.full(
        (original_video.shape[0], original_video.shape[1], 4, 3),
        245,
        dtype=torch.uint8,
    )
    video_frames = torch.cat(
        [
            original_video.cpu(),
            separator,
            reconstructed_uint8,
            separator,
            absolute_error,
        ],
        dim=2,
    )
    header_image = Image.new(
        "RGB",
        (video_frames.shape[2], 32),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(header_image)
    panel_width = original_video.shape[2]
    labels = ("Original", "Reconstructed", "Abs error x5")
    for label_index, label in enumerate(labels):
        draw.text((label_index * (panel_width + 4) + 8, 8), label, fill="white")
    header = torch.from_numpy(np.array(header_image)).to(torch.uint8)
    comparison = torch.cat(
        [
            einops.repeat(
                header,
                "h w c -> t h w c",
                t=video_frames.shape[0],
            ),
            video_frames,
        ],
        dim=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        quality=9,
    ) as writer:
        for frame in comparison:
            writer.append_data(frame.numpy())


def main() -> None:
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)
    video, input_fps = load_video(args.video_path, args.num_frames, return_fps=True)
    model = AnchorMotionVideoAutoEncoder.from_pretrained(
        args.checkpoint,
        vae_path=args.vae_path,
        device=args.device,
        torch_dtype=torch_dtype,
    ).eval()

    with torch.inference_mode():
        memory = model.encode_video(video)
        reconstructed_video = model.decode_video(
            memory["anchor_latent"],
            memory["motion_tokens"],
        )

    anchor_latent = memory["anchor_latent"]
    motion_tokens = memory["motion_tokens"]
    print(
        "Input video shape "
        f"(frame, height, width, channel): {tuple(video.shape)}"
    )
    print(
        "Anchor latent shape "
        f"(batch, channel, latent_height, latent_width): "
        f"{tuple(anchor_latent.shape)}"
    )
    print(
        "Motion tokens shape "
        f"(batch, target_frame, motion_token, channel): "
        f"{tuple(motion_tokens.shape)}"
    )
    print(f"Total motion token number: {motion_tokens.numel()}")
    save_comparison_video(
        video,
        reconstructed_video[0],
        args.output,
        input_fps,
    )


if __name__ == "__main__":
    main()
