"""Verify E6 content_match consistency - 3 runs."""
import os, json, time, re
from pathlib import Path
import warnings as _w; _w.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from enhanced_strategies import strategy_content_match, run_strategy

queries = json.load(open('guidelines/public_test_set.json'))
for run in range(3):
    print(f'\n=== RUN {run+1} ===')
    r = run_strategy(f'E6_run{run+1}', strategy_content_match, queries, verbose=False)
    print(f'Result: Hit={r["hit3"]:.0f}% MRR={r["mrr"]:.4f}')
    # Show ranks
    def norm(s): return re.sub(r'[:\s]+', '', str(s).upper())
    for qr in r["results"]:
        exp_n = norm(qr["expected"][0])
        ret_n = [norm(s) for s in qr["retrieved"]]
        pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 0
        print(f'  {qr["id"]}: rank={pos}')
