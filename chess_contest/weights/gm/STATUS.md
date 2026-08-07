# Rocket 3000 — on-policy SF-MAX trails, no runtime Stockfish

Estimated Elo **2591.0** (sequential=2591.0, MLE=1946.0; floor 3000.0; crush_3000=False).

`choose_move` never calls Stockfish. Strength = offline on-policy float64 trails (908,757 positions) + short IDAS. Swarm disabled on this ladder so trails are not SEE-gated away.

Think: 40 ms. trail_first=true, stockfish_at_play=false.
