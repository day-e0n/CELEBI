"""Future-aware page cache hints: what the runner knows about the upcoming queries.

Before each query the runner calls sirius_set_cache_hints with the columns the query reads
and, for every column a later query reads, how many queries ahead that next use is. The page
store then evicts the column needed furthest ahead first (mode "evict") and, in mode "admit",
also refuses to cache a column no later query reads.

horizon "execution": the cache sees only the rest of the current pass over the batch -- each
execution is a batch that arrives on its own. horizon "run": it also sees the passes that
follow, i.e. it is told the batch repeats.
"""
from __future__ import annotations

from functools import lru_cache

from cache_aware_query_reorder import query_signature


@lru_cache(maxsize=None)
def _columns(qnum: int) -> frozenset[str]:
    return frozenset(column for _, column in query_signature(qnum, "all_columns"))


def hint_spec(mode: str, qnums: list[int], executions: int, execution: int, position: int,
              horizon: str) -> str:
    """The sirius_set_cache_hints argument for the query at 1-based (execution, position)."""
    n = len(qnums)
    if horizon == "run":
        sequence = qnums * executions
        index = (execution - 1) * n + (position - 1)
    else:
        sequence = qnums
        index = position - 1
    next_use: dict[str, int] = {}
    for j in range(index + 1, len(sequence)):
        for column in _columns(sequence[j]):
            next_use.setdefault(column, j - index)
    current = ",".join(sorted(_columns(sequence[index])))
    upcoming = ",".join(f"{c}:{d}" for c, d in sorted(next_use.items()))
    return f"{mode};cur={current};next={upcoming}"
