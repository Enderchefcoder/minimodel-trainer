# Image models

The vision pipeline mirrors the text one: prepare a corpus, train from a
recipe, sample. Same config system, same checkpoint layout, same CLI shape.

```bash
minimodel vision quickstart      # whole pipeline on synthetic sprites, ~1 min
minimodel vision models          # architectures + templates with exact params
minimodel vision datasets        # the image dataset catalogue
```

## Choosing an architecture

| Family | Generates | Pick it when |
| --- | --- | --- |
| `pixelgpt` | palette-indexed pixel art, autoregressively | sprites/pixel art - exact palette output, controllable via temperature |
| `dit` | continuous images via rectified flow | general generation with >~20K training images |
| `unet` | same objective, convolutional | small datasets (<~20-50K images) - the conv prior earns its keep |
| `image-edit` | edited images from (source, instruction) | InstructPix2Pix-style editing |
| `vae` | latents | prerequisite for diffusion beyond ~64px |

**Why PixelGPT is autoregressive, not diffusion:** pixel art *is* a discrete
palette. Diffusion works in continuous colour and quantises back, smearing
the hard colour boundaries that define the style; an AR model over palette
indices emits exactly-on-palette pixels by construction, and 24x24 = 576
tokens is a trivial sequence length. The bundled `pixelgpt_24x24_10m`
(~9.9M params) is sized for `unstonio/pixelgpt-24x24-20k`.

## Pixel art, end to end

```bash
# 1. Corpus: quantize sprites to a shared palette (exact palette detected
#    automatically when the art uses fewer colours than the budget - lossless)
minimodel vision data prepare --dataset pixelgpt-24x24 --mode palette \
    --size 24 --palette-size 64 -o data/images/sprites

# 2. Train (~10M params; a few GPU-hours, or overnight CPU)
minimodel vision train --config configs/vision/pixelgpt_24x24.yaml

# 3. Sample a sheet of sprites (temperature controls how adventurous)
minimodel vision sample -m runs/pixelgpt-24x24/model -n 16 \
    --temperature 0.8 --top-p 0.9 -o sprites.png --scale 8
```

Details that matter: the model gets **separate row and column embeddings**
(so 2-D adjacency and the vertical symmetry of sprites are learnable from
step one), class conditioning when the corpus has labels, and horizontal-flip
augmentation on by default (symmetric sprites make it free data). Partial
sprites can be completed by passing the fixed pixels as a `prompt` to
`model.generate`. Watch `val_pixel_accuracy` and `bits_per_pixel` in the
logs.

## Diffusion (DiT / UNet)

Training uses **rectified flow** by default: noise and image are connected by
a straight line, the model predicts the constant velocity, loss is plain MSE.
Straight paths integrate accurately in few steps - 20-50 Euler steps versus
DDPM's hundreds - and there is no noise-schedule weighting to tune (`ddpm`
objective is available for comparison). Timesteps sample logit-normal,
concentrating gradient where denoising is actually hard. An **EMA** of the
weights (decay 0.999+) is kept and used for sampling - visibly cleaner than
the raw weights.

```bash
minimodel vision data prepare --dataset cifar10 --size 32 -o data/images/cifar
minimodel vision train --config configs/vision/dit_cifar.yaml
minimodel vision sample -m runs/dit-cifar/model -n 16 --label 3 \
    --guidance 3.0 --steps 50 --sampler euler -o cats.png
```

Conditioning: `class` (label embedding with a null slot) or `text` (a small
jointly-trained encoder over the same BPE tokenizer as the language models -
no external CLIP). Both drop their condition for 10% of training samples,
which is what makes **classifier-free guidance** possible at sampling time
(`--guidance 2-4`; conditional and unconditional branches run in one batched
forward). Samplers: `euler` (default), `heun` (2x cost, better under ~20
steps), `ddim`/`ddpm` for epsilon models.

## Instruction editing

The edit model is a DiT whose input is the noisy target **channel-concatenated
with the clean source image**, plus the instruction embedding. Concatenation
(not cross-attention) is the point: output is spatially aligned with input
for almost every edit, so the model can learn "copy unless told otherwise".

```bash
minimodel vision data prepare --dataset instructpix2pix --size 64 -o data/images/ip2p
minimodel vision train --config configs/vision/image_edit.yaml
minimodel vision edit -m runs/edit-64/model -i photo.png \
    --instruction "make it night" -t artifacts/tokenizer.json -o night.png
```

Editing needs **two guidance scales** (both exposed): image guidance (~1.5)
controls faithfulness to the source, text guidance (~5-7) controls how hard
the instruction is applied. Training recipe: pretrain on the big synthetic
InstructPix2Pix set, fine-tune on human-annotated MagicBrush. Flip
augmentation stays **off** (it breaks "move it left").

## Latents (VAE)

Past ~64px, train `vae_f8_64` first (L1 reconstruction + a *tiny* KL, weight
`1e-6` - the KL exists to bound the latent scale, not to make the VAE
generative), freeze it, then train diffusion in its 8x-downsampled latent
space via `encode_for_diffusion` / `decode_from_diffusion`.

## Storage format

Image corpora are memmapped `uint8` shards + `index.json` (+
`palette.json` / `captions.jsonl` / paired source shards where relevant).
Fixed-size records mean a random batch is one fancy-index - no JPEG decode on
the training path, and a 20K-sprite corpus is ~34MB that lives in page cache.
`minimodel vision data info <dir>` prints the stats.
