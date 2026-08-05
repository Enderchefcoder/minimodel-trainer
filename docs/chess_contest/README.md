# Chess contest framework

Scoring for unique architecture entries (Elo + bracket + uniqueness).

```python
from chess_contest.stigmergy.uniqueness import score_uniqueness, composite_contest_score
from chess_contest.stigmergy.elo import estimate_elo
from chess_contest.stigmergy.bracket import round_robin
from chess_contest.stigmergy.weights import uniqueness_fingerprint

uniq = score_uniqueness(uniqueness_fingerprint())
# ... play ladder / bracket ...
composite_contest_score(elo=1550, bracket_winrate=0.72, uniqueness=uniq.score)
```

See [architecture.md](architecture.md) and `chess_contest/README.md`.
