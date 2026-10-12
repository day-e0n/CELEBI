"""Print the small-first-big-last reorder of the 41-query natural ClickBench order, exactly as
run_clickbench_breakdown.py would compute it for a 6 GB cache budget."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "test" / "tpch_performance"))
from performance_test import load_query_set
import cache_aware_query_reorder as reorder

load_query_set("clickbench_queries")
reorder.load_column_set("clickbench_pin_columns")
reorder.load_column_bytes(sys.argv[1])
natural = [q for q in range(1, 44) if q not in (28, 29)]
cfg = reorder.ReorderConfig(policy="small-first-big-last", scope="fixed_width", window=0,
                            keep_first=False, resident_column_budget=0, cache_budget_bytes=6 * 2 ** 30)
print(",".join(map(str, reorder.reorder_query_sequence(natural, cfg).reordered_queries)))
