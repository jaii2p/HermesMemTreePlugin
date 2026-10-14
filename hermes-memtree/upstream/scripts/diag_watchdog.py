"""One real query, end to end through retrieval and delivery -- no model needed.

Prints: the query words, each word's posting length and idf, the final score of
every candidate, which node holds the answer and where it ranks, the whisper
gate decision, and whether the answer survives into the rendered memory block.

    python scripts/diag_watchdog.py --haystack haystack.txt --words 3000 --depth 0.5 --seed 0
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_recall import make_trial  # noqa: E402

from memtree.runtime import file_user_turn, render_memory, whisper_query, window_budgets  # noqa: E402
from memtree.store import LogGate, MemoryStore, words_of  # noqa: E402
from memtree.watchdog import Watchdog, vitality  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack", default="haystack.txt")
    ap.add_argument("--words", type=int, default=3000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args(argv)
    hay = open(a.haystack, errors="ignore").read().split()
    rng = random.Random(a.seed)
    s = rng.randrange(0, max(1, len(hay) - a.words))
    tr = make_trial(rng, " ".join(hay[s:s + a.words]), a.depth)
    bud = window_budgets(a.window)
    st = MemoryStore(); gate = LogGate(st); dog = Watchdog(st)
    shown, qkey, tail = file_user_turn(gate, "me", 1, tr["doc"] + "\n\n" + tr["question"],
                                       bud["tail_words"], bud["span_words"])
    b, idx = st.branches["me"], st.words["me"]
    n = len(b)
    q = whisper_query(tail)
    qw = words_of(q)
    center = {x.key for x in dog.context_view("me")}
    holders = [k for k in b.index if tr["answer"] in str(b.get(k)[0].value)]
    print(f"needle: {tr['needle']}\nmemories n={n}  center={len(center)}  answer held by: {holders or 'NO NODE (severed)'}"
          f"{'  [IN CENTER: excluded from whispers]' if set(holders) & center else ''}")
    print(f"\nquery ({len(q.split())} words): {q[:160]}...\n\nword   posting   idf     used")
    for w in sorted(qw, key=lambda w: len(idx.get(w, ()))):
        df = len(idx.get(w, ()))
        print(f"{w:6s} {df:7d} {math.log(1.0 + n / df) if df else 0:7.2f}   {'yes' if df and df <= 512 else ('NO (df>max_df)' if df else 'not indexed')}")
    found = dog.search("me", q, limit=50, exclude={qkey} | center)
    print(f"\ncandidates after vitality weighting (top {a.show} of {len(found)}):")
    rank = None
    for i, (nd, sc) in enumerate(found):
        mark = "  <== HOLDS THE ANSWER" if nd.key in holders else ""
        if nd.key in holders:
            rank = i + 1
        if i < a.show or mark:
            print(f"  #{i + 1:<3d} {nd.key:16s} score={sc:6.2f} vitality={vitality(nd, b.clock):.2f}{mark}")
    floor = 0.1 * math.log(1.0 + max(n, 1))
    top = found[0][1] if found else 0
    print(f"\ngate: floor={floor:.2f}  best={top:.2f}  rel-bar={0.35 * top:.2f}  answer rank={rank}")
    ws = dog.whispers("me", q, exclude={qkey})
    blk = render_memory(dog, "me", entry_chars=bud["entry_chars"], whispers=ws, aside=None)
    print(f"whispers delivered: {[w.key for w in ws]}")
    print("answer survives into the rendered memory block:", tr["answer"] in blk)
    verdict = ("severed before storage (split)" if not holders else
               "excluded as center, clipped" if set(holders) & center else
               "retrieval: never matched" if rank is None else
               "retrieval: ranked below the whisper limit" if tr["answer"] not in blk and rank > 2 else
               "delivery: matched but the snippet lost it" if tr["answer"] not in blk else "OK")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
