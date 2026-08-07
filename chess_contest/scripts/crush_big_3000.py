"""Crush-big: ~30M SwarmNet v3 -> 3000 Elo ladder (SF teacher only).

Play never calls Stockfish. Pipeline:
1) Expand MultiPV soft-label set with full-strength SF
2) Cold-start SwarmNet v3 (256x12 + SE + wide heads, ~30M params)
3) OOD early-stop training
4) Long-think IDAS ladder vs SF UCI_Elo 2200-3100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.scripts.crush_3000 import (  # noqa: E402
    generate_dataset,
    train_swarm,
    write_reports,
)
from chess_contest.scripts.sprint_3000 import policy_match, sprint_ladder  # noqa: E402
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.search import set_policy_sprint, set_swarm  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.swarm_net import (  # noqa: E402
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    IN_CH,
    SwarmNet,
    try_load_swarm,
)
from chess_contest.stigmergy.weights import default_weights  # noqa: E402


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    p.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    p.add_argument("--positions", type=int, default=120000)
    p.add_argument("--extra-positions", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--sf-depth", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--games-per", type=int, default=6)
    p.add_argument("--stig-ms", type=int, default=5000)
    p.add_argument("--stig-depth", type=int, default=12)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "crush_big.log"
    data_path = out / "crush_dataset.npz"
    net_path = out / "swarm_net.pt"
    best_path = out / "swarm_net_best_ood.pt"
    legacy = out / "swarm_net_v2_15m_legacy.pt"

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    # Archive incompatible v2 tower if present (no SE layers).
    if net_path.is_file() and try_load_swarm(net_path) is None:
        if not legacy.is_file():
            net_path.rename(legacy)
            _log(log, f"archived incompatible checkpoint → {legacy}")
        else:
            net_path.unlink()
            _log(log, "removed incompatible swarm_net.pt (legacy already archived)")

    net = SwarmNet(channels=args.channels, blocks=args.blocks, in_ch=IN_CH)
    n_params = net.count_params()
    _log(
        log,
        f"=== CRUSH-BIG net={args.channels}x{args.blocks} params={n_params:,} "
        f"({n_params / 1e6:.1f}M) ep={args.epochs} ===",
    )
    if n_params < 10_000_000:
        raise SystemExit(f"crush-big requires ≥10M params, got {n_params}")

    weights = default_weights()
    weights.format_version = 4
    weights.book.clear()
    weights.trails.clear()
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "swarm_net": str(net_path),
            "crush_big": True,
            "swarm_params": n_params,
            "swarm_arch": f"{args.channels}x{args.blocks}",
        }
    )

    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=512))
    try:
        if args.probe_only:
            loaded = try_load_swarm(net_path) or try_load_swarm(best_path)
            if loaded is None:
                raise SystemExit("no crush-big checkpoint to probe")
            net = loaded
            _log(log, f"probe net {net.channels}x{net.blocks} params={net.count_params():,}")
            set_swarm(net)
            eng = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
            probe = sprint_ladder(
                eng,
                sf,
                log,
                games_per=args.games_per,
                targets=[1320, 1600, 2000, 2200, 2500, 2700, 2800, 2900, 3000, 3100],
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                use_policy_sprint=False,
            )
            match = policy_match(net, sf, n=100)
            probe["policy_match_ood"] = round(match, 4)
            probe["swarm_params"] = net.count_params()
            write_reports(out, probe, args)
            _log(log, f"PROBE Elo≈{probe['estimated_elo']} OOD={match:.0%} crush={probe['crush_3000']}")
            return 0

        # Dataset: reuse + expand.
        data: list[tuple[np.ndarray, np.ndarray, float]] = []
        if data_path.is_file():
            blob = np.load(data_path)
            data = [
                (np.asarray(pl, dtype=np.float32), np.asarray(s, dtype=np.float32), float(v))
                for pl, s, v in zip(blob["planes"], blob["soft"], blob["values"], strict=True)
            ]
            _log(log, f"loaded crush dataset {len(data)}")

        need = max(0, args.positions - len(data))
        if not args.skip_gen and (need > 0 or args.extra_positions > 0):
            n_gen = need + (0 if args.skip_gen else args.extra_positions)
            if n_gen > 0:
                extra = generate_dataset(
                    sf, log, n_positions=n_gen, depth=args.sf_depth, weights=None
                )
                data = (data + extra)[-max(args.positions, len(data) + len(extra)) :]
                np.savez_compressed(
                    data_path,
                    planes=np.stack([d[0] for d in data]),
                    soft=np.stack([d[1] for d in data]),
                    values=np.asarray([d[2] for d in data], dtype=np.float32),
                )
                _log(log, f"dataset size={len(data)}")

        if len(data) < 8000:
            raise SystemExit(f"need ≥8000 labeled positions, have {len(data)}")

        # Optional warm-start only if arch matches.
        existing = try_load_swarm(net_path)
        if existing is not None and existing.channels == args.channels and existing.blocks == args.blocks:
            net = existing
            _log(log, f"warm-start {net_path} params={net.count_params():,}")
        else:
            _log(log, f"cold-start {args.channels}x{args.blocks} params={n_params:,}")

        best_ood = policy_match(net, sf, n=80)
        _log(log, f"baseline OOD={best_ood:.0%}")
        net.save(best_path)
        net.save(net_path)

        for ep in range(1, args.epochs + 1):
            train_swarm(
                net,
                data,
                log,
                epochs=1,
                batch_size=args.batch_size,
                lr=3.0e-4 if ep <= 2 else 1.0e-4,
                save_path=net_path,
            )
            ood = policy_match(net, sf, n=120)
            _log(log, f"epoch-ood {ep}/{args.epochs} OOD={ood:.0%} best={best_ood:.0%}")
            if ood >= best_ood - 0.01:
                if ood > best_ood:
                    best_ood = ood
                    net.save(best_path)
                    net.save(net_path)
                    _log(log, f"new best OOD={best_ood:.0%}")
            else:
                _log(log, f"OOD drop {ood:.0%} < {best_ood:.0%}; early-stop")
                net.load(best_path)
                break

        set_swarm(net)
        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(net)
        probe = sprint_ladder(
            eng,
            sf,
            log,
            games_per=args.games_per,
            targets=[1320, 1600, 2000, 2200, 2500, 2700, 2800, 2900, 3000, 3100],
            stig_ms=args.stig_ms,
            stig_depth=args.stig_depth,
            use_policy_sprint=False,
        )
        probe["policy_match_ood"] = round(best_ood, 4)
        probe["swarm_params"] = net.count_params()
        probe["swarm_arch"] = f"{net.channels}x{net.blocks}"
        write_reports(out, probe, args)
        _log(
            log,
            f"CRUSH-BIG Elo≈{probe['estimated_elo']} crush3000={probe['crush_3000']} "
            f"OOD={best_ood:.0%} params={net.count_params():,}",
        )

        if probe["estimated_elo"] < args.floor:
            _log(log, "below floor - longer confirm ladder")
            confirm = sprint_ladder(
                eng,
                sf,
                log,
                games_per=max(8, args.games_per),
                targets=[2200, 2500, 2700, 2800, 2900, 3000, 3100],
                stig_ms=max(args.stig_ms, 12000),
                stig_depth=max(args.stig_depth, 16),
                use_policy_sprint=False,
            )
            confirm["policy_match_ood"] = round(best_ood, 4)
            confirm["swarm_params"] = net.count_params()
            write_reports(out, confirm, args)
            _log(log, f"CONFIRM Elo≈{confirm['estimated_elo']} crush={confirm['crush_3000']}")
    finally:
        sf.close()
        set_swarm(None)
        set_policy_sprint(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
