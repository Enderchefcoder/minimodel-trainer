# Crush path - no runtime Stockfish

Estimated Elo **1666.0** (floor target 3000.0; crush-3000=False).

`choose_move` never calls Stockfish. SwarmNet v2 distilled from full-strength SF offline; policy-sprint/policy-first + IDAS.

Think budget: 600 ms / move.
policy_sprint=False ood_match=0.2
