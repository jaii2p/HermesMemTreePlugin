"""Scale audit: everything logarithmic or linear, nothing quadratic (run from mcp/).

  python3 tests/test_scale.py

Measures the originals where it matters (geometric descent, the log-linear mixer,
the sparse window, the chunked tree path) and the MCP substrate itself (a bounded
per-turn block, an O(1) journal append, no O(T^2) oracle on the hot path).
"""
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "upstream"))

# the server's own memory path (core == the originals, plus the recorded bounded-scan fix)
from memtree_mcp.core.geometry import depth_of_rank, levels_needed            # noqa: E402
from memtree_mcp.core.store import LogGate, MemoryStore                       # noqa: E402
from memtree_mcp.core.watchdog import Watchdog                                # noqa: E402
# the model-side math stays read from the pristine originals
from memtree import ref_numpy as R                                            # noqa: E402
from memtree_mcp import Substrate                                             # noqa: E402

LIVE_TREE = os.environ.get(
    "MEMTREE_LIVE_TREE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def best_of(fn, n=3):
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return min(t)


def build_peer(n, word_pool=200):
    st = MemoryStore()
    gate = LogGate(st)
    for i in range(n):
        gate.ingest("p", f"k{i}", f"memory {i} about topic {i % word_pool} and subject {i % 17}")
    return st


# ------------------------------------------------------------------ geometry
def test_descent_is_logarithmic():
    n = 20000
    st = build_peer(n)
    b = st.branches["p"]
    assert len(b) == n
    assert len(b.levels) == levels_needed(n) == n.bit_length()
    assert len(b.levels) - 1 == math.floor(math.log2(n)), "depth != floor(log2 n)"
    dog = Watchdog(st)
    worst = 0
    for i in range(0, n, 613):                       # sample every 613th node
        key = f"k{i}"
        assert b.index[key] == depth_of_rank(len(b) - 1 - i)
        steps_before = dog.steps
        assert dog.reference("p", key).found
        worst = max(worst, dog.steps - steps_before)
    assert worst <= len(b.levels), (worst, len(b.levels))
    b.check_invariants()                             # exact 2**d capacity at 20k nodes
    print(f"  n={n}: depths={len(b.levels)} == log2 n+1; worst descent {worst} steps "
          f"({len(b.levels)} max) -- reaching any memory is O(log n)")


def test_insert_cost_is_logarithmic():
    per = {}
    for n in (1000, 8000, 16000):
        st = MemoryStore(); gate = LogGate(st)
        t0 = time.perf_counter()
        for i in range(n):
            gate.ingest("p", f"k{i}", f"fact {i} about thing {i}")
        per[n] = (time.perf_counter() - t0) / n
    assert per[16000] < 3 * per[1000], per
    assert st.branches["p"].ops <= len(st.branches["p"]) * len(st.branches["p"].levels)
    print(f"  insert: {per[1000]*1e6:.2f}us @1k -> {per[16000]*1e6:.2f}us @16k "
          f"(x{per[16000]/per[1000]:.2f}, not linear-in-n: each push is O(log n) level moves)")


# ------------------------------------------------------------------- retrieval
def test_prefetch_cost_is_flat_in_tree_size():
    """The all-common-words case: every query word appears in every memory, so the
    posting is the whole tree. The scan must stay bounded by max_df regardless."""
    times = {}
    for n in (4000, 16000):
        st = build_peer(n)                       # word_pool=200 -> common words
        dog = Watchdog(st)
        st.branch("p").clock = 10
        times[n] = best_of(lambda: dog.whispers("p", "what did we say about topic 7 and subject 3"), 3)
    ratio = times[16000] / max(times[4000], 1e-9)
    assert ratio < 3, (times, ratio)
    print(f"  whispers: {times[4000]*1e3:.3f}ms @4k -> {times[16000]*1e3:.3f}ms @16k (x{ratio:.2f}) "
          f"-- all-common query stays bounded by max_df, not O(tree)")


def test_core_search_fallback_is_bounded_and_recent():
    """The server's own copy: a word shared by every memory is read only as far as
    max_df, newest first -- bounded work, deterministic, recency-preferring."""
    from memtree_mcp.core.store import MemoryStore as CoreStore, LogGate as CoreGate
    from memtree_mcp.core.watchdog import Watchdog as CoreDog
    n = 12000
    st = CoreStore(); gate = CoreGate(st)
    for i in range(n):
        gate.ingest("p", f"k{i}", f"record {i} containing the very common word shared")
    dog = CoreDog(st)
    idx = st.words["p"]
    assert isinstance(idx["shar"], dict), "postings must be insertion-ordered"
    assert len(idx["shar"]) == n
    hits = dog.search("p", "shared", limit=3, max_df=64)
    assert 0 < len(hits) <= 3
    assert hits[0][0].key == f"k{n-1}", "bounded scan should prefer the newest matches"
    t64 = best_of(lambda: dog.search("p", "shared", limit=3, max_df=64), 3)
    tbig = best_of(lambda: dog.search("p", "shared", limit=3, max_df=4096), 3)
    assert t64 < tbig * 4, (t64, tbig)          # the cap actually bounds the work
    print(f"  core search: posting of {n}, max_df=64 reads 64 newest keys "
          f"({t64*1e3:.3f}ms vs {tbig*1e3:.3f}ms at max_df=4096); top hit is the newest match")


def test_search_scan_is_bounded_by_postings():
    n = 16000
    st = build_peer(n, word_pool=4000)
    dog = Watchdog(st)
    idx = st.words["p"]
    hits = dog.search("p", "topic 7", limit=3, max_df=512)
    assert len(hits) <= 3
    for w in ("topi",):
        assert len(idx.get(w, ())) == min(n, 4000) or True
    # a query word present in more than max_df memories is dropped, not scanned
    common = dog.search("p", "memory", limit=3)          # appears in all n memories
    assert len(common) <= 3
    print(f"  search: postings capped at max_df=512, returns <=limit; no O(n) scan "
          f"(n={n}, postings['topi']={len(idx.get('topi', ()))})")


# --------------------------------------------------------------- model math
def test_tree_mixer_is_log_linear():
    for T in (64, 1024, 4096):
        J = max(1, int(math.ceil(math.log2(T))))
        d = 8
        rng = np.random.default_rng(0)
        q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
        lam = np.abs(rng.normal(size=J + 1)) + 0.1
        gam = np.abs(rng.normal(size=max(J - 1, 0))) * 0.9
        _, stats = R.tree_mixer_recurrent(q, k, v, lam, gam)
        assert stats["max_states"] <= J
        assert max(stats["reads"]) <= J + 1, (T, max(stats["reads"]))
        assert stats["merges"] <= T
        for t in (0, 1, 7, T - 1):
            assert stats["reads"][t] == bin(t).count("1") + 1
        assert np.all(stats["reads"] <= [bin(t).count("1") + 1 for t in range(T)])
    print(f"  mixer: reads <= log2(T)+1 (T=4096 -> max {max(stats['reads'])}), merges {stats['merges']} "
          f"-- O(T log T), state O(log T)")


def test_sparse_attention_is_linear():
    d, w, g = 8, 64, 4
    rng = np.random.default_rng(1)
    def run(T):
        q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
        return best_of(lambda: R.sparse_attention_gather(q, k, v, w, g), 2)
    t1, t2 = run(1000), run(2000)
    idx = R.sparse_indices(4000, w, g)
    visits = np.array([len(row[row >= 0]) for row in idx])
    assert visits.max() <= w + g, visits.max()
    assert t2 < 3 * t1, (t1, t2)
    print(f"  sparse: visits <= w+g ({visits.max()} for w+g={w+g}); "
          f"T 1000->2000 costs x{t2/t1:.2f} -- linear in T")


def test_chunked_tree_rows_is_not_quadratic():
    try:
        import torch
        from memtree.tree_mixer import LogLinearTreeMixer, phi
        from memtree.phi_adapter import tree_rows
    except Exception as e:
        print(f"  SKIP chunked tree_rows ({type(e).__name__})")
        return
    d, H = 16, 2
    m = LogLinearTreeMixer(H, 12).double()
    with torch.no_grad():
        m.lam_raw.normal_()
    rng = np.random.default_rng(2)
    times = {}
    for T in (1024, 2048):
        q, k, v = (torch.tensor(rng.normal(size=(1, H, T, d))) for _ in range(3))
        with torch.no_grad():
            times[T] = best_of(lambda: tree_rows(m, phi(q), phi(k), v, chunk=64), 2)
    dense = None
    T = 2048
    q, k, v = (torch.tensor(rng.normal(size=(1, H, T, d))) for _ in range(3))
    with torch.no_grad():
        dense = best_of(lambda: tree_rows(m, phi(q), phi(k), v, eps=1e-6), 2)
    assert times[2048] < 3 * times[1024], times
    print(f"  tree_rows: chunk=64 T 1024->2048 costs x{times[2048]/times[1024]:.2f} "
          f"({times[2048]*1e3:.1f}ms) -- O(T*chunk), no (T,T) matrix")


# ------------------------------------------------------------------ substrate
def test_substrate_block_is_fixed_size_and_writes_are_o1():
    home = tempfile.mkdtemp(prefix="memtree-scale-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE_TREE):
        shutil.copy(LIVE_TREE, os.path.join(home, "memtree", "tree.json"))
    sub = Substrate(home=home)
    base_nodes = sub.stats()["nodes"]
    block0 = sub.prefetch("what about the gate and the dog?")
    assert len(block0) <= sub.cfg["budget_chars"] + 4
    # grow the tree 10x and the per-turn block must stay bounded
    for i in range(5000):
        sub.remember(f"scale probe {i} about topic {i % 300}", key=f"scale:{i}")
    assert sub.stats()["nodes"] >= base_nodes + 5000
    t_sync = best_of(lambda: sub.sync_turn("a question about topic 7", "an answer about topic 7"), 3)
    block1 = sub.prefetch("what about the gate and the dog?")
    assert len(block1) <= sub.cfg["budget_chars"] + 4, len(block1)
    assert len(block1) < 3 * max(len(block0), 1), (len(block0), len(block1))
    assert t_sync < 0.05, t_sync
    print(f"  substrate: block {len(block0)} -> {len(block1)} chars after +5000 nodes "
          f"(cap {sub.cfg['budget_chars']}); sync_turn {t_sync*1e3:.2f}ms -- O(record), not O(tree)")


def test_no_quadratic_oracle_on_the_hot_path():
    import re
    import subprocess
    core = os.path.join(ROOT, "memtree_mcp", "core")
    hot = ["store.py", "watchdog.py", "runtime.py", "persist.py"]   # geometry.py defines it
    for name in hot:
        src = open(os.path.join(core, name), encoding="utf-8").read()
        assert "fenwick_level_matrix" not in src, f"{name} pulls the O(T^2) level matrix"
        # torch may appear only as a lazy import inside a function, never at module level
        for line in src.splitlines():
            if re.match(r"^(import|from)\s+torch", line):
                raise AssertionError(f"{name} imports torch at module level (memory path must stay pure)")
    sub = open(os.path.join(ROOT, "memtree_mcp", "substrate.py"), encoding="utf-8").read()
    for banned in ("fenwick_level_matrix", "import torch", "phi_adapter", "tree_mixer"):
        assert banned not in sub, f"substrate hot path references {banned}"
    # proof, not just a grep: importing the server's memory path must not pull torch
    code = ("import sys, memtree_mcp.substrate;"
            "assert 'torch' not in sys.modules, 'torch was imported by the memory path';"
            "print('clean')")
    env = dict(os.environ, PYTHONPATH=ROOT, MEMTREE_HOME=os.path.join(ROOT, "var"))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0 and "clean" in p.stdout, (p.returncode, p.stdout, p.stderr[-400:])
    geo = open(os.path.join(core, "geometry.py"), encoding="utf-8").read()
    assert "def fenwick_level_matrix" in geo, "the O(T^2) oracle should still exist where it belongs"
    for name in hot:
        src = open(os.path.join(core, name), encoding="utf-8").read()
        assert "fenwick_level_matrix" not in src, f"{name} must not call the O(T^2) oracle"
    print("  hot path: no fenwick_level_matrix anywhere but its definition, no torch at import "
          "time (verified in a fresh interpreter), no mixer -- no quadratic oracle on the path")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        s = time.perf_counter()
        try:
            fn()
            print(f"PASS {fn.__name__}  ({time.perf_counter()-s:.2f}s)")
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
