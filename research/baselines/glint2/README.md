---
license: mit
language:
- en
---
Check out this for infer! https://huggingface.co/spaces/Glint-Research/glint-2-effort-explorer

one transformer block, looped 8 times, 1.06 million parameters. the pure-loop model: we deleted every unique layer and kept one wide block that iterates. the release: weights, tokenizer, two inference scripts, a 3.5 KB corrective probe, this readme. the sft chat variant arrices soon in the same repo.

## what it actually is

- 1,065k parameters total. one shared block of about 645k (dim 96, ffn hidden 2112, 8 heads), iterated 8 times with a tiny per-loop lora and a loop embedding telling it which iteration it is on. tied embeddings over a 4096-token bpe vocab make up most of the rest.
- zero unique layers before or after the loop. the loop is the model.
- effort built in: the same 645k weights get 8 passes at every token. depth without the parameter bill.

## the one rule: loops=8

the model trains at 8 loops. the checkpoint config says 16, which is the table capacity and a trap. run it at 16 and you get gibberish. `generate.py` in this repo defaults to 8 and you should not touch the `--loops` flag unless you enjoy watching a model dissolve.

## the numbers, against every glint before it

house rule, same as always: real scores, no dressing. comparison scores are from the tiny-ml leaderboard. glint-0.1 and 0.2 are excluded because their wikitext perplexities are in the millions and they break the y-axis, and honestly they break my heart too.

| model | params | blimp | arc-easy | wikitext-2 ppl |
|---|---|---|---|---|
| glint-0.3 | 1M | 47.3 | 25.5 | 7.87 |
| glint-0.4 | 1M | 58.5 | 31.0 | 5.01 |
| glint-1 | 1M | 61.2 | 32.0 | 4.45 |
| glint-1.3 (merged) | 982K | 68.7 | 32.5 | 3.08 |
| **glint-2** | **1.06M** | **73.96** | **36.80** | **3.09** |

![blimp comparison](img/blimp.png)

![arc-easy comparison](img/arc_easy.png)

![wikitext-2 comparison](img/wikitext.png)

blimp 73.96 beats the 1.3 merge's 68.7. arc-easy 36.80 is the best any glint has posted, by a wide margin. wikitext 3.09 matches 1.3's 3.08. single checkpoint, no soup, no tuning to the eval. this is the base model running at the setup it was trained on.

## sample output

default sampling, unedited:

> "Once upon a time, there was a little girl named Lily. She loved to play with her friends and have fun. One day, Lily's friend Billy came over to play with her. Lily saw a big, red ball in the park."

that is the good prompt. on encyclopedic prompts it drifts within two sentences ("The Battle of the Middle Ages is the first to be the first of the"). this is a 1M parameter model. expect a research artifact, and you will be happy.

## effort levels

`effort.py` runs the same 1.06M weights at six effort levels. the levels scale search compute, never the model: low is one sample, ultra is ten independent beam searches with the best final output winning. loops stays 8 at every level. here is the full ladder on one simple prompt, "The sun is", 60 tokens, unedited, timed on cpu:

**low** (1 sample, 0.2s)

> The sun is a little bit of the day. "Look, Mommy! I found it in the park!" Tom said. He was curious and wanted to see what was inside. "Mommy, can we go?" Lily asked

**medium** (1 sample, tuned config, 0.2s)

> The sun is a little bit of the day. "Wow, look at that!" Lily said. "It's so pretty and nice." But then, her mom came into the room and saw what was happening.

**high** (6 proposals, reranked, 0.9s)

> The sun is the best way to find out. One day, a little girl named Lily went to the park with her mom and dad. They saw many animals in the woods.

**xhigh** (8 instances x 2 beams, 24-token chunks, 1.5s)

> The sun is the best way to find out. One day, a little girl named Lily went to the park with her mom and dad. They saw a big box in the park. It was red and

**max** (8 instances x 4 beams, 2.8s)

> same output as xhigh on this prompt. more beams only pay off when the proposals disagree.

**ultra** (10 independent max searches, 25.8s)

> The sun is in the sky. "Look, a beautiful flower!" Tom says. He runs to the garden and sees that it has many colors of colorful flowers.

read the openings in order. low and medium start with "the sun is a little bit of the day", which is nothing. high and xhigh swap it for different nothing. ultra opens with a true sentence about the sun, found by brute force: a hundred and sixty candidate continuations scored and discarded to buy four honest words. that is what effort scaling looks like at 1M parameters. the ceiling does not move, but the model reaches it far more often.

## the corrective probe

`effort_probe.pt` is a 3.5 KB neural net that rides along with the search: one linear layer over the model's own mean-pooled hidden state, trained to output P(text is real) by separating real fineweb-edu passages from the model's own generations.

**where it comes online:** low and medium never rerank. from high upward, every rerank decision blends `2.0 * P(real)` into the score. high uses it to pick between 6 finished continuations; xhigh, max, and ultra consult it every 24-token round, steering the beam frontier continuously.

**why it exists:** the base rerank score is the model's own logprob, and a model grading its own homework Goodharts under heavy search. under search, the model drifts toward whatever scores high: memorized wikitext boilerplate. the probe is trained against exactly that failure, so confident garbage stops winning. two measurements from this exact release:

the probe's raw verdicts, same prompt, real text vs model text:

```
P(real) on a real encyclopedia sentence about the sun:  0.575
P(real) on the model's own continuation of it:          0.000
```

and the search outcome it flips, xhigh on "The history of the United States":

```
without probe: The history of the United States and the Battle of Saratoga .
               = = = Forti and the High Powers = = =
               The Corps of the House of Commons was a majority of the

with probe:    The history of the United States, and its first time in the
               19th century. Competition of the House of Lords is a majority
               of the world's most famous and well-known, unusual and fascist.
```

without the probe the search dives straight into wikitext section-header markup, because headers are the single most predictable pattern the model knows. with the probe it writes prose — wrong prose, gloriously incoherent prose, but prose. the probe cannot make a 1M model correct. stopping the search from rewarding boilerplate is worth 3.5 KB.

## files

```
checkpoints/glint-2.pt   the weights (17 MB)
tokenizer.json           4096-vocab bpe tokenizer
generate.py              plain sampling, one file, no dependencies beyond torch + tokenizers
effort.py                the effort ladder, low through ultra
effort_probe.pt          the 3.5 KB corrective probe (delete it and effort.py degrades gracefully)
img/                     the comparison charts above
README.md                you are here
```

## quickstart

```
pip install torch tokenizers huggingface_hub
hf download Glint-Research/Glint-2 --local-dir Glint-2
cd Glint-2
python generate.py "Once upon a time"
python effort.py low "The sun is"
python effort.py ultra "The sun is"        # ~30s on cpu, worth it once
```

runs on cpu. the model is 17 MB. your browser tab is bigger.

/lane
glint research, 2026, one block, eight loops, 1.06M parameters, six effort levels, one 3.5 KB probe keeping the search honest, sft variant inbound