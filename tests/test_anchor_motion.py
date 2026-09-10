from pathlib import Path

import yaml

import pytest
import torch

from memory_encoder import (
    AnchorMotionAutoEncoder,
    AnchorMotionConfig,
    AnchorMotionVideoAutoEncoder,
    encode_svd_frames,
)


def build_model() -> AnchorMotionAutoEncoder:
    return AnchorMotionAutoEncoder(
        AnchorMotionConfig(
            latent_channels=8,
            hidden_dim=32,
            num_motion_tokens=4,
            num_layers=2,
            num_heads=4,
            num_position_harmonics=2,
        )
    )


def test_forward_shapes() -> None:
    model = build_model()
    anchor_latent = torch.randn(2, 8, 6, 8)
    target_latents = torch.randn(2, 4, 8, 6, 8)
    frame_indices = torch.arange(1, 5)[None, :].expand(2, -1)
    outputs = model(anchor_latent, target_latents, frame_indices)
    assert outputs["motion_tokens"].shape == (2, 4, 4, 32)
    assert outputs["reconstructed_latents"].shape == (2, 4, 8, 6, 8)
    assert torch.isfinite(outputs["reconstructed_latents"]).all()


def test_invalid_shapes() -> None:
    model = build_model()
    anchor_latent = torch.randn(2, 8, 6, 8)
    target_latents = torch.randn(2, 4, 8, 6, 7)
    with pytest.raises(ValueError):
        model.encode(anchor_latent, target_latents)


def test_svd_encoder_normalizes_uint8_frames() -> None:
    class EncoderOutput:

        def __init__(self, frames):
            class LatentDist:

                def mode(self):
                    return frames

            self.latent_dist = LatentDist()

    class FakeSVDVAE(torch.nn.Module):
        dtype = torch.float32
        device = torch.device("cpu")
        config = type("Config", (), {"scaling_factor": 1.0})()

        def encode(self, frames):
            return EncoderOutput(frames)

    video = torch.zeros((1, 1, 2, 1, 3), dtype=torch.uint8)
    video[0, 0, 0, 0] = torch.tensor([0, 128, 255])
    latents = encode_svd_frames(FakeSVDVAE(), video)
    assert torch.allclose(
        latents[0, 0, :, 0, 0],
        torch.tensor([-1.0, 128 / 127.5 - 1.0, 1.0]),
        atol=1e-6,
    )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_model()
    checkpoint_directory = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint_directory)
    loaded_model = AnchorMotionAutoEncoder.from_pretrained(
        checkpoint_directory
    )
    anchor_latent = torch.randn(1, 8, 4, 5)
    target_latents = torch.randn(1, 2, 8, 4, 5)
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        original = model(anchor_latent, target_latents)[
            "reconstructed_latents"
        ]
        loaded = loaded_model(
            anchor_latent,
            target_latents,
            torch.arange(1, 3)[None, :],
        )["reconstructed_latents"]
    assert torch.allclose(original, loaded, atol=1e-6)


def test_checkpoint_loads_training_config(tmp_path: Path) -> None:
    model = build_model()
    training_output = tmp_path / "training_output"
    checkpoint_directory = training_output / "checkpoint-1"
    checkpoint_directory.mkdir(parents=True)
    config = {
        "model": {
            "config_class": "AnchorMotionModelConfig",
            "type": "AnchorMotionPipeline",
            "latent_channels": 8,
            "hidden_dim": 32,
            "num_motion_tokens": 4,
            "num_layers": 2,
            "num_heads": 4,
            "num_position_harmonics": 2,
            "dropout": 0.0,
        }
    }
    with (training_output / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file)
    torch.save(model.state_dict(), checkpoint_directory / "model.pt")
    loaded_model = AnchorMotionAutoEncoder.from_pretrained(checkpoint_directory)
    assert loaded_model.config == model.config


def test_video_autoencoder_accepts_raw_video() -> None:
    class LatentDist:

        def __init__(self, frames):
            self.frames = frames

        def mode(self):
            latent = self.frames.new_zeros(
                self.frames.shape[0],
                8,
                self.frames.shape[2],
                self.frames.shape[3],
            )
            latent[:, :3] = self.frames
            return latent

    class EncoderOutput:

        def __init__(self, frames):
            self.latent_dist = LatentDist(frames)

    class DecoderOutput:

        def __init__(self, sample):
            self.sample = sample

    class FakeSVDVAE(torch.nn.Module):

        dtype = torch.float32
        device = torch.device("cpu")
        config = type("Config", (), {"scaling_factor": 1.0})()

        def encode(self, frames):
            return EncoderOutput(frames)

        def decode(self, latents, num_frames):
            sample = latents[:, :3]
            return DecoderOutput(sample)

    model = AnchorMotionVideoAutoEncoder(build_model(), FakeSVDVAE())
    video = torch.randint(
        0,
        256,
        (2, 5, 8, 6, 3),
        dtype=torch.uint8,
    )
    outputs = model.encode_video(video)
    assert outputs["anchor_latent"].shape == (2, 8, 8, 6)
    assert outputs["motion_tokens"].shape == (2, 4, 4, 32)
    reconstructed_video = model.decode_video(
        outputs["anchor_latent"],
        outputs["motion_tokens"],
    )
    assert reconstructed_video.shape == (2, 5, 8, 6, 3)

    single_video = video[0]
    single_outputs = model.encode_video(single_video)
    assert single_outputs["anchor_latent"].shape == (1, 8, 8, 6)
    assert single_outputs["motion_tokens"].shape == (1, 4, 4, 32)
