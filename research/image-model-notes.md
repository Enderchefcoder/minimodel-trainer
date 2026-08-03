# Image-model notes

## Objectives

- **Rectified flow / flow matching** (Liu et al., arXiv:2209.03003; Lipman et
  al., arXiv:2210.02747; validated at scale by SD3, arXiv:2403.03206) — our
  default: straight paths ⇒ 20–50 Euler steps, MSE loss, no schedule
  weighting. Logit-normal timestep sampling is also from the SD3 report.
- **DDPM/DDIM** (arXiv:2006.11239 / arXiv:2010.02502) kept as `objective:
  ddpm` for comparison and external checkpoints; cosine ᾱ from iDDPM
  (arXiv:2102.09672).

## Architectures

- **DiT** (Peebles & Xie, arXiv:2212.09748) — AdaLN-Zero conditioning (gates
  init to 0 ⇒ identity at init) is the load-bearing detail; taken exactly.
- **UNet** (arXiv:2006.11239 lineage) — kept because below ~20–50K images the
  conv prior beats DiT at matched params (matches our synthetic-sprite runs).
- **Classifier-free guidance** (Ho & Salimans, arXiv:2207.12598) — 10%
  condition dropout at train time; batched cond+uncond at sampling.
- **InstructPix2Pix** (Brooks et al., arXiv:2211.09800) — channel-concat the
  source (alignment prior) + *dual* guidance scales; both in `edit.py`.
- **Latent diffusion** (Rombach et al., arXiv:2112.10752) — the f8 VAE with a
  tiny KL (1e-6) and the 0.18215 latent scale convention.
- **EMA weights for sampling** — standard since DDPM; decay 0.999–0.9995.

## Autoregressive pixels

- **PixelRNN/PixelCNN** (arXiv:1601.06759) → **ImageGPT** (Chen et al., 2020)
  — AR over pixels is exact-likelihood and palette-native. Dead at high
  resolution, *ideal* at 24×24/576 tokens: our PixelGPT is ImageGPT with a
  modern decoder block, factored row+col embeddings, and a palette from
  median-cut/k-means (Heckbert, 1982 lineage).
- Why not diffusion for sprites: continuous colour + quantise-back smears the
  hard edges that define pixel art; AR emits on-palette by construction.

## Evaluation honesty

FID needs an Inception net and thousands of samples; below that regime the
honest tools are a fixed-seed sample grid per checkpoint (`sample_every`),
val loss, and — for PixelGPT — bits/pixel and pixel accuracy. That's what the
trainers log.
