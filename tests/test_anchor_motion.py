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
from memory_encoder.video_autoencoder import _resolve_vae_type


class FakeCheckpointWanVAE(torch.nn.Module):

    device = torch.device("cpu")
    dtype = torch.float32
    config = type(
        "Config",
        (),
        {"scale_factor_temporal": 4, "scale_factor_spatial": 8},
    )()


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


def build_temporal_model() -> AnchorMotionAutoEncoder:
    return AnchorMotionAutoEncoder(
        AnchorMotionConfig(
            latent_channels=8,
            hidden_dim=32,
            num_motion_tokens=4,
            num_layers=2,
            num_heads=4,
            num_position_harmonics=2,
            temporal_attention="anchor_conditioned",
            temporal_insertion_indices=[1],
            anchor_context_size=2,
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


def test_temporal_attention_forward_shapes() -> None:
    model = build_temporal_model().eval()
    anchor_latent = torch.randn(2, 8, 6, 8)
    target_latents = torch.randn(2, 4, 8, 6, 8)

    outputs = model(anchor_latent, target_latents)

    assert outputs["motion_tokens"].shape == (2, 4, 4, 32)
    assert outputs["reconstructed_latents"].shape == (2, 4, 8, 6, 8)
    assert torch.isfinite(outputs["motion_tokens"]).all()


def test_temporal_attention_is_causal() -> None:
    model = build_temporal_model().eval()
    anchor_latent = torch.randn(1, 8, 6, 8)
    target_latents = torch.randn(1, 4, 8, 6, 8)
    changed_latents = target_latents.clone()
    changed_latents[:, 2:] += 2.0

    with torch.inference_mode():
        original_tokens = model.encode(anchor_latent, target_latents)
        changed_tokens = model.encode(anchor_latent, changed_latents)

    assert torch.allclose(
        original_tokens[:, :2],
        changed_tokens[:, :2],
        atol=1e-6,
    )
    assert not torch.allclose(
        original_tokens[:, 2:],
        changed_tokens[:, 2:],
        atol=1e-6,
    )


def test_temporal_attention_ignores_padded_future() -> None:
    model = build_temporal_model().eval()
    anchor_latent = torch.randn(1, 8, 6, 8)
    target_latents = torch.randn(1, 4, 8, 6, 8)
    padding_mask = torch.tensor([[False, False, True, True]])

    with torch.inference_mode():
        full_tokens = model.encode(
            anchor_latent,
            target_latents,
            target_padding_mask=padding_mask,
        )
        short_tokens = model.encode(
            anchor_latent,
            target_latents[:, :2],
        )

    assert torch.allclose(
        full_tokens[:, :2],
        short_tokens,
        atol=1e-6,
    )


def test_default_model_keeps_legacy_state_dict() -> None:
    model = build_model()

    assert not any(
        "temporal" in key or "anchor_context" in key
        for key in model.state_dict()
    )


def build_dynamic_source_model() -> AnchorMotionAutoEncoder:
    return AnchorMotionAutoEncoder(
        AnchorMotionConfig(
            latent_channels=8,
            hidden_dim=32,
            num_motion_tokens=4,
            num_layers=2,
            num_heads=4,
            num_position_harmonics=2,
            temporal_attention="anchor_conditioned",
            temporal_insertion_indices=[1],
            anchor_context_size=2,
            query_source="delta_target_anchor",
            auxiliary_heads=["dynamic_mask", "anchor_flow"],
            dynamic_mask_size=12,
        )
    )


def test_dynamic_source_and_auxiliary_shapes() -> None:
    model = build_dynamic_source_model()
    anchor_latent = torch.randn(2, 8, 6, 8)
    target_latents = torch.randn(2, 4, 8, 6, 8)

    outputs = model(anchor_latent, target_latents)

    assert outputs["motion_tokens"].shape == (2, 4, 4, 32)
    assert outputs["reconstructed_latents"].shape == (2, 4, 8, 6, 8)
    assert outputs["dynamic_mask"].shape == (2, 4, 1, 12, 12)
    assert outputs["anchor_flow"].shape == (2, 4, 2, 6, 8)
    assert torch.isfinite(outputs["dynamic_mask"]).all()
    assert torch.isfinite(outputs["anchor_flow"]).all()


def test_dynamic_source_keeps_causal_temporal_order() -> None:
    model = build_dynamic_source_model().eval()
    anchor_latent = torch.randn(1, 8, 6, 8)
    target_latents = torch.randn(1, 4, 8, 6, 8)
    changed_latents = target_latents.clone()
    changed_latents[:, 2:] += 2.0

    with torch.inference_mode():
        original_tokens = model.encode(anchor_latent, target_latents)
        changed_tokens = model.encode(anchor_latent, changed_latents)

    assert torch.allclose(
        original_tokens[:, :2],
        changed_tokens[:, :2],
        atol=1e-6,
    )
    assert not torch.allclose(
        original_tokens[:, 2:],
        changed_tokens[:, 2:],
        atol=1e-6,
    )


def test_dynamic_source_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_dynamic_source_model()
    checkpoint_directory = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint_directory)
    loaded_model = AnchorMotionAutoEncoder.from_pretrained(checkpoint_directory)
    model.eval()
    loaded_model.eval()
    anchor_latent = torch.randn(1, 8, 4, 5)
    target_latents = torch.randn(1, 2, 8, 4, 5)

    with torch.no_grad():
        outputs = model(anchor_latent, target_latents)
        loaded_outputs = loaded_model(anchor_latent, target_latents)

    assert torch.allclose(
        outputs["motion_tokens"],
        loaded_outputs["motion_tokens"],
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["anchor_flow"],
        loaded_outputs["anchor_flow"],
        atol=1e-6,
    )


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
    assert (checkpoint_directory / "config.yaml").exists()
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


def test_video_autoencoder_detects_wan_training_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = build_model()
    training_output = tmp_path / "training_output"
    checkpoint_directory = training_output / "checkpoint-1"
    checkpoint_directory.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint_directory / "model.pt")
    with (training_output / "config.yaml").open("w") as file:
        yaml.safe_dump(
            {
                "model": {
                    **model.config.to_dict(),
                    "base_vae_type": "wan",
                }
            },
            file,
        )

    monkeypatch.setattr(
        "memory_encoder.video_autoencoder.load_wan_vae",
        lambda _path, device, torch_dtype: FakeCheckpointWanVAE(),
    )
    loaded_model = AnchorMotionVideoAutoEncoder.from_pretrained(
        checkpoint_directory,
        vae_path="fake-wan-vae",
    )

    assert loaded_model.vae_type == "wan"


def test_base_vae_type_is_inferred_from_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def build_checkpoint(input_channels: int) -> torch.Tensor:
        return {
            "encoder.conv1.weight": torch.zeros(
                1,
                input_channels,
                1,
                1,
                1,
            )
        }

    wan_path = tmp_path / "Wan_VAE.pth"
    wan22_path = tmp_path / "vae.pt"
    wan_path.touch()
    wan22_path.touch()
    monkeypatch.setattr(
        "memory_encoder.video_autoencoder.torch.load",
        lambda path, map_location, weights_only: build_checkpoint(
            3 if Path(path) == wan_path else 12
        ),
    )
    assert _resolve_vae_type({}, wan_path) == "wan"
    assert _resolve_vae_type({}, wan22_path) == "wan22"

    diffusers_wan = tmp_path / "wan-diffusers"
    (diffusers_wan / "vae").mkdir(parents=True)
    (diffusers_wan / "vae" / "config.json").write_text(
        '{"_class_name": "AutoencoderKLWan"}',
        encoding="utf-8",
    )
    diffusers_svd = tmp_path / "svd"
    diffusers_svd.mkdir()
    (diffusers_svd / "config.json").write_text(
        '{"_class_name": "AutoencoderKLTemporalDecoder"}',
        encoding="utf-8",
    )

    assert _resolve_vae_type({}, diffusers_wan) == "wan"
    assert _resolve_vae_type({}, diffusers_svd) == "svd"
    assert _resolve_vae_type({"base_vae_type": "wan22"}, wan_path) == "wan22"

    with pytest.raises(ValueError, match="infer the base VAE type"):
        _resolve_vae_type({}, tmp_path / "unknown")
