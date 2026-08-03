# Cookbook

Copy-paste workflows for concrete goals. All assume `pip install -e ".[hf]"`
(HF datasets access) unless marked offline.

## 1. Prove everything works (offline, 2 minutes)

```bash
minimodel quickstart
minimodel vision quickstart
python scripts/smoke_e2e.py
```

## 2. A storyteller in an evening (~12M, one GPU or a patient CPU)

```bash
minimodel data pull tinystories --limit 500000
minimodel tokenizer train --dataset tinystories --limit 50000 --vocab-size 8192 -o artifacts/tokenizer.json
minimodel data tokenize tinystories -t artifacts/tokenizer.json
minimodel train -c configs/pretrain/demo_tiny.yaml \
    --set model.template=dense_12m \
    --set data.train=data/tokenized/tinystories \
    --set training.max_steps=20000 --set training.seq_len=512
minimodel generate -m runs/demo-tiny/model -p "Once upon a time" --max-new-tokens 200 --min-p 0.05
```

## 3. A 30M assistant (the full lifecycle)

```bash
# Data + tokenizer
minimodel data pull smollm-corpus --limit 3000000          # fineweb-edu + cosmopedia, split by weight
minimodel tokenizer train --dataset fineweb-edu-dedup --limit 100000 --vocab-size 16384 -o artifacts/tokenizer.json
minimodel data tokenize fineweb-edu-dedup -t artifacts/tokenizer.json
minimodel data tokenize cosmopedia-v2     -t artifacts/tokenizer.json

# Pretrain (recipe already points at the mixture)
minimodel train -c configs/pretrain/dense_30m.yaml

# SFT
minimodel data pull smoltalk --limit 200000 && minimodel data pull casual-conversation
minimodel data tokenize smoltalk -t artifacts/tokenizer.json
minimodel posttrain -c configs/sft/instruct.yaml

# Preferences
minimodel data pull ultrafeedback-binarized --limit 20000
minimodel data tokenize ultrafeedback-binarized -t artifacts/tokenizer.json --format preference -o data/tokenized/prefs
minimodel posttrain -c configs/rl/dpo.yaml

# Soften the DPO edge, measure, document, chat
minimodel merge runs/dense-30m-instruct/model runs/dense-30m-dpo/model -o runs/final --method slerp --t 0.6
minimodel bench -m runs/final -o bench.json
minimodel card --run runs/dense-30m --model runs/final --benchmark bench.json \
    --name mm-30m-chat --stage instruct --dataset fineweb-edu-dedup cosmopedia-v2 smoltalk -o runs/final/README.md
minimodel chat -m runs/final
```

## 4. A small reasoner

```bash
# Start from the SFT model of recipe 3
minimodel data pull openr1-math --limit 50000
minimodel data tokenize openr1-math -t artifacts/tokenizer.json --max-length 1024
minimodel posttrain -c configs/cot/reasoning.yaml \
    --set data.train=data/tokenized/openr1-math \
    --set training.reasoning_loss_weight=0.5

# Then RLVR on GSM8K to sharpen it
minimodel data pull gsm8k-rlvr
minimodel posttrain -c configs/rl/rlvr_gsm8k.yaml \
    --set model.checkpoint=runs/dense-30m-reasoning/model
```

Inference with a thinking budget:

```python
from minimodel.inference import load_for_inference, generate_with_reasoning
lm = load_for_inference("runs/dense-30m-rlvr/model")
generate_with_reasoning(lm, "A train leaves at 7 and arrives at 10...", max_reasoning_tokens=256)
```

## 5. The 1.4M looped model (the supra2 spec)

```bash
minimodel tokenizer train --dataset fineweb-edu-10bt --limit 50000 --vocab-size 4096 -o artifacts/tokenizer.json
minimodel data pull fineweb-edu-10bt --limit 1000000
minimodel data tokenize fineweb-edu-10bt -t artifacts/tokenizer.json
minimodel train -c configs/pretrain/supra2_1.4m.yaml

# The party trick: quality scales with inference-time loops
for k in 2 4 8 12; do
  minimodel generate -m runs/supra2-1.4m/model -p "The" --loops $k --temperature 0 --max-new-tokens 40
done
```

## 6. Pixel-art sprites (~10M)

```bash
minimodel vision data prepare --dataset pixelgpt-24x24 --mode palette --size 24 --palette-size 64 -o data/images/sprites
minimodel vision train -c configs/vision/pixelgpt_24x24.yaml
minimodel vision sample -m runs/pixelgpt-24x24/model -n 16 --temperature 0.8 -o sprites.png --scale 8
```

Offline variant: swap the first line for
`minimodel vision data prepare --synthetic --size 24 --mode palette -o data/images/sprites`.

## 7. Class-conditional CIFAR diffusion

```bash
minimodel vision data prepare --dataset cifar10 --size 32 -o data/images/cifar
minimodel vision train -c configs/vision/dit_cifar.yaml
minimodel vision sample -m runs/dit-cifar/model -n 16 --label 5 --guidance 3 --steps 50 -o dogs.png
```

## 8. An image editor

```bash
minimodel vision data prepare --dataset instructpix2pix --size 64 --limit 100000 -o data/images/ip2p
minimodel vision train -c configs/vision/image_edit.yaml
# fine-tune on human-annotated edits
minimodel vision data prepare --dataset magicbrush --size 64 -o data/images/magicbrush
minimodel vision train -c configs/vision/image_edit.yaml \
    --set model.checkpoint=runs/edit-64/model --set data.train=data/images/magicbrush \
    --set run_name=edit-64-mb --set training.max_steps=20000
minimodel vision edit -m runs/edit-64-mb/model -i photo.png --instruction "make it snowy" \
    -t artifacts/tokenizer.json -o snowy.png
```

## 9. Compare a size ladder

```bash
for t in dense_3m dense_12m dense_30m; do
  minimodel train -c configs/pretrain/_base.yaml \
      --set model.template=$t --set run_name=$t --set training.max_steps=20000
  minimodel bench -m runs/$t/model --name $t -o bench_$t.json
done
minimodel compare bench_*.json -o ladder.md
```

## 10. Resume after an interruption

Nothing to do: rerun the same `minimodel train -c ...` command. `resume: true`
finds the latest checkpoint and restores model, optimizer, schedule and data
order. To *branch* instead (e.g. decay from a WSD plateau), point
`model.checkpoint` at the checkpoint directory and give the run a new
`run_name`.
