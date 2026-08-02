# training/rl/

Preference optimization and reinforcement learning, all subclasses of the
core `Trainer`:

- `dpo.py` — Direct Preference Optimization (sigmoid/IPO/hinge/CPO losses,
  frozen reference, reward-accuracy diagnostics).
- `rlvr.py` — RL from Verifiable Rewards using **GRPO** (group-relative
  advantages, clipped ratio, k3 KL). Reward functions live in the `VERIFIERS`
  registry: numeric, exact, boxed-LaTeX, expression, length — add yours with a
  decorator.
- `spin.py` — Self-Play fIne-tuNing: the model's own generations become the
  rejected side of DPO pairs, regenerated each iteration.

Which to use when + the knobs that matter: [docs/rl.md](../../docs/rl.md).
