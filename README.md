# Memory Encoder

Anchor-frame motion tokenizer for world-model history compression. This package
provides a raw-video inference API around a trainable latent model and a frozen
official Diffusers SVD VAE. Training and data preparation stay in the host
framework.

## Install

```bash
pip install -e ".[svd,example]"
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

The checkpoint contains only the trainable Anchor Motion model. The frozen SVD
VAE is loaded directly from its official Diffusers checkpoint by `vae_path`.
The package also includes the minimal official-SVD-VAE adapter shared by this
raw-video API and the host framework's cache preparation. Dataset, cache, and
training implementations remain outside this repository.

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
