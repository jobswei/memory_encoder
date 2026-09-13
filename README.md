# Memory Encoder

Anchor-frame motion tokenizer for world-model history compression. This package
provides a raw-video inference API around a trainable latent model and frozen
SVD, Wan 2.1, or Wan 2.2 VAEs. Training and data preparation stay in the host
framework.

## Install

```bash
pip install -e ".[example]"
```

Or install directly from the repository after it is published:

```bash
pip install git+<repository-url>
```

### Download SVD VAE

Only the official VAE files are required:

```bash
huggingface-cli download stabilityai/stable-video-diffusion-img2vid \
  vae/config.json \
  vae/diffusion_pytorch_model.safetensors \
  --local-dir work_dirs/Checkpoints/stable-video-diffusion-img2vid
```

The resulting directory must contain `vae/config.json` and
`vae/diffusion_pytorch_model.safetensors`; pass it as `vae_path`.

### Download Wan 2.1 VAE

The raw official Wan 2.1 VAE is supported:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth \
  --local-dir work_dirs/Checkpoints/Wan2.1-T2V-1.3B
```

Pass `Wan2.1_VAE.pth` as `vae_path`. A Diffusers Wan VAE directory containing
`config.json` and weights is also supported. Wan 2.2 uses a different checkpoint
and architecture from Wan 2.1.

### Download Wan 2.2 VAE

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir work_dirs/Checkpoints/Wan2.2-TI2V-5B
```

Pass `Wan2.2_VAE.pth` as `vae_path`. Wan 2.2 produces 48-channel latents with
`4x` temporal and `16x` spatial compression. Its checkpoint must be a raw
official Wan 2.2 VAE file.

### Video API

```python
import torch

from memory_encoder import AnchorMotionVideoAutoEncoder

model = AnchorMotionVideoAutoEncoder.from_pretrained(
    "path/to/anchor-motion-checkpoint",
    vae_path="path/to/stable-video-diffusion-img2vid",
).eval()

with torch.inference_mode():
    # uint8 video: [T,H,W,C] or [B,T,H,W,C]
    memory = model(video)
    reconstructed_video = model.decode_video(
        memory["anchor_latent"],
        memory["motion_tokens"],
    )
```

The checkpoint contains only the trainable Anchor Motion model. The frozen base
VAE is loaded from its official checkpoint by `vae_path`. Its type is inferred
from `model.base_vae_type` in the training config, or from the base VAE
checkpoint/config when that field is absent. The package also includes minimal
official SVD/Wan VAE adapters shared by this raw-video API and the host
framework's cache preparation.
Checkpoints can optionally enable anchor-conditioned interleaved causal
attention through their saved model config; the raw-video and latent APIs do not
change.
Dataset, cache, and training implementations remain outside this repository.

### Latent API

If a host application already has SVD latents, use
`AnchorMotionAutoEncoder.from_pretrained` and pass `[B,C,H,W]` anchors plus
`[B,T,C,H,W]` target latents directly.

## Checkpoint format

`save_pretrained` writes:

- `config.yaml`: model configuration
- `model.safetensors`: model weights when `safetensors` is installed
- `model.pt`: fallback weight format

`from_pretrained` also accepts VAM training checkpoints and reads model fields
from `config.yaml` in the checkpoint's parent directory.

## Example

Run a minimal raw-video encode/decode demo with a bundled complete video:

```bash
python examples/run_demo.py \
  --checkpoint path/to/checkpoint-100000 \
  --vae-path path/to/stable-video-diffusion-img2vid \
  --num-frames 8
```

The script prints the anchor latent, motion token, and reconstructed video
shapes, then writes a side-by-side comparison video: original, reconstructed,
and absolute error multiplied by five. Use `--video-path` to process another
video.


## Development

```bash
pytest
```
