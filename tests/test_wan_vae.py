from __future__ import annotations

import einops
import pytest
import torch

from memory_encoder import AnchorMotionAutoEncoder, AnchorMotionVideoAutoEncoder
from memory_encoder.config import AnchorMotionConfig
from memory_encoder.wan22_vae import encode_wan22_video
from memory_encoder.wan_vae import (
    decode_wan_video,
    encode_wan_video,
    wan_temporal_compression,
)


class FakeWanVAE(torch.nn.Module):

    device = torch.device("cpu")
    dtype = torch.float32
    config = type(
        "Config",
        (),
        {
            "latents_mean": [0.0, 0.0, 0.0],
            "latents_std": [1.0, 1.0, 1.0],
            "scale_factor_temporal": 4,
            "scale_factor_spatial": 8,
        },
    )()

    def encode(self, video: torch.Tensor):
        latent = video[:, :, ::4]

        class LatentDist:

            def mode(self) -> torch.Tensor:
                return latent

        return type("EncoderOutput", (), {"latent_dist": LatentDist()})()

    def decode(self, latents: torch.Tensor):
        video = einops.repeat(
            latents,
            "b c t h w -> b c (t repeat) h w",
            repeat=4,
        )
        return type("DecoderOutput", (), {"sample": video})()


class FakeWan22VAE(torch.nn.Module):

    device = torch.device("cpu")
    dtype = torch.float32

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        return videos[:, :, ::4, ::16, ::16]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        video = einops.repeat(
            latents,
            "b c t h w -> b c (t repeat) (h hrepeat) (w wrepeat)",
            repeat=4,
            hrepeat=16,
            wrepeat=16,
        )
        return video


def test_encode_wan_video_supports_uint8_and_channels_first() -> None:
    vae = FakeWanVAE()
    uint8_video = torch.randint(
        0,
        256,
        (2, 9, 8, 8, 3),
        dtype=torch.uint8,
    )

    latents = encode_wan_video(vae, uint8_video)
    channels_first = einops.rearrange(
        uint8_video,
        "b t h w c -> b t c h w",
    )
    channels_first_latents = encode_wan_video(vae, channels_first)

    assert latents.shape == (2, 3, 3, 8, 8)
    assert torch.allclose(latents, channels_first_latents)


def test_wan_latent_roundtrip_shape() -> None:
    vae = FakeWanVAE()
    latents = torch.randn(2, 3, 3, 8, 8)

    video = decode_wan_video(vae, latents)

    assert video.shape == (2, 12, 8, 8, 3)


def test_encode_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="encode_batch_size"):
        encode_wan_video(FakeWanVAE(), torch.zeros(1, 2, 8, 8, 3), 0)


def test_wan_temporal_compression() -> None:
    assert wan_temporal_compression(FakeWanVAE()) == 4


def test_wan_video_autoencoder_matches_requested_frame_count() -> None:
    anchor_motion = AnchorMotionAutoEncoder(
        AnchorMotionConfig(
            latent_channels=3,
            hidden_dim=16,
            num_motion_tokens=2,
            num_layers=1,
            num_heads=2,
            num_position_harmonics=1,
        )
    )
    model = AnchorMotionVideoAutoEncoder(
        anchor_motion,
        FakeWanVAE(),
        encode_batch_size=1,
        vae_type="wan",
    )
    video = torch.randint(0, 256, (1, 10, 8, 8, 3), dtype=torch.uint8)
    memory = model.encode_video(video)

    decoded_video = model.decode_video(
        memory["anchor_latent"],
        memory["motion_tokens"],
        num_frames=10,
    )

    assert decoded_video.shape == (1, 10, 8, 8, 3)


def test_encode_wan22_video_supports_raw_layout() -> None:
    video = torch.randint(
        0,
        256,
        (2, 9, 32, 32, 3),
        dtype=torch.uint8,
    )

    latents = encode_wan22_video(FakeWan22VAE(), video)

    assert latents.shape == (2, 3, 3, 2, 2)


def test_wan22_video_autoencoder_accepts_raw_video() -> None:
    anchor_motion = AnchorMotionAutoEncoder(
        AnchorMotionConfig(
            latent_channels=3,
            hidden_dim=16,
            num_motion_tokens=2,
            num_layers=1,
            num_heads=2,
            num_position_harmonics=1,
        )
    )
    model = AnchorMotionVideoAutoEncoder(
        anchor_motion,
        FakeWan22VAE(),
        encode_batch_size=1,
        vae_type="wan22",
    )
    video = torch.randint(0, 256, (1, 9, 32, 32, 3), dtype=torch.uint8)

    memory = model.encode_video(video)
    reconstructed_video = model.decode_video(
        memory["anchor_latent"],
        memory["motion_tokens"],
        num_frames=9,
    )

    assert memory["anchor_latent"].shape == (1, 3, 2, 2)
    assert memory["motion_tokens"].shape == (1, 2, 2, 16)
    assert reconstructed_video.shape == (1, 9, 32, 32, 3)
