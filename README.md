# Memory Encoder

Anchor-frame motion tokenizer for world-model history compression. This package
provides a raw-video inference API around a trainable latent model and a frozen
official Diffusers SVD VAE. Training and data preparation stay in the host
framework.

## Install

```bash
pip install -e "third_party/memory_encoder[svd]"
```

Or install directly from the repository after it is published:

```bash
pip install git+<repository-url>
```

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

- `config.json`: model configuration
- `model.safetensors`: model weights when `safetensors` is installed
- `model.pt`: fallback weight format

`from_pretrained` also accepts VAM training checkpoints and reads model fields
from `config.yaml` in the checkpoint's parent directory.


## Development

```bash
pytest
```
