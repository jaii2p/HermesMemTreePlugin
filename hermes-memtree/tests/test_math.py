"""Exact-math audit of the ORIGINALS (run: python3 tests/test_math.py from mcp/).

Every check re-derives the original formula from first principles and compares it
to what the code actually computes, on the vendored upstream copy in
`mcp/upstream/memtree/`. Nothing here edits the originals; it only measures them.

Covered: geometry, store, watchdog, runtime, persist (the memory core) and
ref_numpy / tree_mixer / sparse_attn / phi_adapter (the model math).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.dirname(HERE) + "/upstream"
sys.path.insert(0, UPSTREAM)

from memtree import ref_numpy as R                                       # noqa: E402
from memtree import persist as P                                         # noqa: E402
from memtree.geometry import (depth_of_rank, fenwick_blocks,             # noqa: E402
                              fenwick_level_matrix, level_capacity, levels_needed)
from memtree.store import (HALF_LIFE, BridgeTable, LogGate, MemoryStore,  # noqa: E402
                           Node, Branch, strengthen, vitality, words_of)
from memtree.watchdog import (BOUNDARY, Watchdog, _snippet,               # noqa: E402
                              normalize_key)
from memtree import runtime as RT                                        # noqa: E402


# --------------------------------------------------------------------- geometry
def test_geometry_depth_is_log2():
    for rank in range(4097):
        assert depth_of_rank(rank) == (rank + 1).bit_length() - 1
        assert depth_of_rank(rank) == math.floor(math.log2(rank + 1))
    for n in range(1, 5000):
        assert levels_needed(n) == n.bit_length() == math.ceil(math.log2(n + 1))
    print("  depth_of_rank/levels_needed == floor/ceil log2 exact on 4097 ranks")


def test_fenwick_blocks_tile_and_count():
    for n in range(1, 4097):
        blocks = fenwick_blocks(n)
        assert sum(1 << j for _, j in blocks) == n, "blocks do not tile [0,n)"
        assert len(blocks) == bin(n).count("1"), "block count != popcount"
        pos = 0
        for start, j in blocks:  # oldest/largest first, contiguous, aligned
            assert start == pos and (start % (1 << j)) == 0
            pos += 1 << j
        assert all(b[1] <= a[1] for a, b in zip(blocks, blocks[1:])), "not oldest-largest first"
    print("  fenwick_blocks tiles [0,n), counts popcount(n), oldest-largest, on n<=4096")


def test_fenwick_level_matrix_matches_definition():
    for T in (1, 2, 3, 7, 16, 33, 64):
        lev = fenwick_level_matrix(T)
        assert lev.shape == (T, T)
        for t in range(T):
            for s in range(T):
                if s > t:
                    assert lev[t, s] == -1
                elif s == t:
                    assert lev[t, s] == 0
                else:
                    want = next(j + 1 for start, j in fenwick_blocks(t)
                                if start <= s < start + (1 << j))
                    assert lev[t, s] == want, (T, t, s)
    print("  fenwick_level_matrix == the block definition for every (t,s), T<=64")


def test_xor_levels_equals_fenwick_rows():
    for Tq, Tk in ((1, 1), (3, 77), (5, 64), (17, 200), (64, 64)):
        assert (R.xor_levels(Tq, Tk) == fenwick_level_matrix(Tk)[Tk - Tq:]).all(), (Tq, Tk)
    print("  ref_numpy.xor_levels == fenwick_level_matrix rows (highbit(t xor s)+1)")


# ------------------------------------------------------------------------ store
def test_words_of_stems_and_stopwords():
    w = words_of("Allergies and the allergic reaction drove the driving drive")
    assert "alle" in w and "driv" in w, w          # 4-letter stem
    assert "the" not in w and "and" not in w        # stopwords
    assert words_of("a i") == set()                 # len>1 filter + stopwords
    assert "4821" in words_of("code 48213")         # digits kept, cut to the 4-char stem
    print("  words_of: 4-letter stems, stopwords dropped, len>1, digits kept")


def test_vitality_half_life_and_strengthen():
    n = Node(key="k", peer="p", value="v")
    n.strength, n.s_clock = 1.0, 0
    assert vitality(n, 0) == 1.0
    assert abs(vitality(n, HALF_LIFE) - 0.5) < 1e-12   # fades by half
    assert abs(vitality(n, 2 * HALF_LIFE) - 0.25) < 1e-12
    strengthen(n, 10, 1.0)
    assert abs(n.strength - (vitality(Node("k", "p", "v"), 0) * 0.5 ** (10 / HALF_LIFE) + 1.0)) < 1e-12
    assert n.s_clock == 10
    print("  vitality: strength*0.5**(dt/60); strengthen adds on top of current vitality")


def test_branch_geometry_and_invariants():
    b = Branch("p")
    for i in range(600):
        b.insert_front(Node(key=f"k{i}", peer="p", value=i))
        b.check_invariants()  # exact 2**d capacity + rank==depth
        if i:
            assert len(b.levels) == levels_needed(i + 1)
    ranks = list(b.nodes_by_recency())
    for rank, node in enumerate(ranks):
        assert b.index[node.key] == depth_of_rank(rank)
    assert (1 << (len(b.levels) - 1)) <= 600 <= (1 << len(b.levels))
    for key in list(b.index)[::7]:
        b.remove(key)
    b.check_invariants()
    b.touch(next(iter(b.index)))
    b.check_invariants()
    print(f"  600 inserts keep 2**d capacity; depth == floor(log2 n); max depth {len(b.levels)-1}")


def test_bridge_table_cap_symmetry_eviction():
    bt = BridgeTable(max_degree=4)
    a = ("p", "a")
    for i in range(6):
        bt.add(a, ("p", f"n{i}"), 1.0 + i)   # weakest first, strongest last
    bt.check_invariants()
    assert len(bt.neighbors(a)) == 4, "degree cap not enforced"
    assert ("p", "n0") not in bt.neighbors(a) and ("p", "n1") not in bt.neighbors(a), "weakest not evicted"
    bt.redirect(("p", "n5"), ("p", "n9"))
    bt.check_invariants()
    assert any(k == ("p", "n9") for k in bt.neighbors(a)), "redirect dropped the edge"
    print("  BridgeTable: symmetric, degree cap 4, weakest evicted, redirect moves edges")


def test_loggate_dedup_update_near_dup_and_index():
    st = MemoryStore()
    g = LogGate(st, sim_threshold=0.95)
    assert g.ingest("p", "k1", "the Marisol protocol is biomedicine") == "new"
    assert g.ingest("p", "k1", "the Marisol protocol is biomedicine") == "refresh"
    assert g.ingest("p", "k1", "the Marisol protocol is neurobiology") == "update"
    node = st.branch("p").get("k1")[0]
    assert node.history == ["the Marisol protocol is biomedicine"], node.history
    assert len(node.mentions) == 3
    emb = np.zeros(8); emb[0] = 1.0
    assert g.ingest("p", "k2", "near twin", emb=emb) == "new"
    twin = emb * 1.0; twin[1] = 0.1
    assert g.ingest("p", "k3", "near twin two", emb=twin) == "near_dup"
    assert st.branch("p").aliases["k3"] == "k2"
    far = np.zeros(8); far[7] = 1.0
    assert g.ingest("p", "k4", "far", emb=far) == "new"
    assert g.ingest("q", "k1", "shared fact") == "new"
    assert st.bridges.neighbors(("p", "k1")).get(("q", "k1")) == 1.0, "cross-peer key bridge missing"
    idx = st.words["p"]
    # postings map stem -> node keys; the stem itself is never a member
    assert "mari" in idx and "prot" in idx, sorted(idx)
    assert idx["mari"] == {"k1"} and idx["prot"] == {"k1"}, (idx["mari"], idx["prot"])
    assert "the" not in idx, "stopword leaked into the word index"
    print("  LogGate: new/refresh/update(+history), cos>=0.95 near_dup+alias, cross-peer bridge, word index")


# --------------------------------------------------------------------- watchdog
def _mk_peer(store, items, peer="p"):
    gate = LogGate(store)
    for k, v in items:
        gate.ingest(peer, k, v)
    return Watchdog(store)


def _idf_scores(dog, peer, query, limit=3, exclude=(), max_df=512):
    """Independent re-derivation of Watchdog.search's formula."""
    b, idx = dog.store.branches[peer], dog.store.words[peer]
    q = words_of(query)
    n = max(len(b), 1)
    usable = [w for w in q if idx.get(w) and len(idx[w]) <= max_df] or [w for w in q if idx.get(w)][:1]
    scores = {}
    for w in usable:
        posting = idx[w]
        idf = math.log(1.0 + n / len(posting))
        for k in list(posting):
            if k not in b.index or k in exclude:
                continue
            node = b.get(k)[0]
            if w not in words_of(k) | words_of(node.value):
                continue
            scores[k] = scores.get(k, 0.0) + idf
    full = {w for w in RT.re.findall(r"[a-z0-9]+", str(query).lower()) if len(w) > 4 and w[:4] in q} \
        if hasattr(RT, "re") else set()
    import re as _re
    full = {w for w in _re.findall(r"[a-z0-9]+", str(query).lower()) if len(w) > 4 and w[:4] in q}
    for k in scores:
        node = b.get(k)[0]
        if full:
            have = set(_re.findall(r"[a-z0-9]+", str(node.value).lower()))
            scores[k] += sum(math.log(1.0 + n / max(len(idx.get(w[:4], ())), 1)) for w in full if w in have)
        if str(node.value).rstrip().endswith("?"):
            scores[k] *= 0.5
        scores[k] *= 0.5 + 0.75 * min(vitality(node, b.clock), 2.0)
    ranked = sorted(scores, key=lambda k: (-scores[k], -b.get(k)[0].touched))
    return [(b.get(k)[0], scores[k]) for k in ranked[:limit]]


def test_watchdog_search_score_formula():
    st = MemoryStore()
    dog = _mk_peer(st, [
        ("k1", "the Marisol protocol uses biomedicine"),
        ("k2", "a protocol handbook for common things"),
        ("k3", "common things and common habits"),
        ("k4", "what is the protocol?"),
        ("k5", "biomedicine studies at the lab"),
    ])
    for query in ("Marisol protocol", "protocol", "common things", "biomedicine lab"):
        got = dog.search("p", query, limit=3)
        want = _idf_scores(dog, "p", query, limit=3)
        assert [n.key for n, _ in got] == [n.key for n, _ in want], (query, [n.key for n, _ in got])
        for (_, a), (_, bb) in zip(got, want):
            assert abs(a - bb) < 1e-12, (query, a, bb)
    print("  search: score == sum of log(1+n/df) (+full-word IDF), ?-penalty x0.5, vitality x(0.5+0.75v)")


def test_watchdog_whispers_gate_and_lookalike_suppression():
    st = MemoryStore()
    dog = _mk_peer(st, [
        ("k1", "Marisol ran the Marisol protocol on Tuesday"),
        ("k2", "Marisa filed the allergy paperwork"),
        ("k3", "the fax machine is in the hallway"),
        ("k4", "Tuesday meetings moved to the lab"),
    ])
    b = st.branches["p"]
    top = dog.search("p", "Marisol protocol", limit=10)
    floor = 0.1 * math.log(1.0 + len(b))
    assert top and top[0][1] >= floor, "best candidate below floor"
    out = dog.whispers("p", "Marisol protocol")
    keys = [w.key for w in out]
    assert "k1" in keys, keys
    assert "k2" not in keys, "stem-only look-alike (Marisa) was not suppressed"
    assert "k3" not in keys, "unrelated memory passed the gate"
    live = b.get("k1")[0]
    assert live.strength > 1.0, "surfacing a whisper did not strengthen the memory"
    print(f"  whispers: floor 0.1*log(1+n)={floor:.4f}, rel 0.35*top; Marisa look-alike dropped; memory strengthened")


def test_snippet_window_bounded_and_covering():
    text = ("filler " * 40) + "the secret code for Harlan is 48213 " + ("filler " * 40)
    q = words_of("secret code Harlan")
    snip = _snippet(text, q, 60)
    assert len(snip) <= 60 + 7, len(snip)          # + "..." markers only
    assert "48213" in snip, snip
    assert all(w[:4] in snip.lower() for w in ("secret", "code", "harlan")), snip
    print("  _snippet: window <= width, cluster placed early so the answer survives the clip")


def test_reference_resolution_confidence_and_promotion():
    st = MemoryStore()
    dog = _mk_peer(st, [(f"k{i}", f"memory number {i}") for i in range(6)])
    b = st.branches["p"]
    # newest is k5 (depth 0, the center); the oldest is k0 (depth 2, outside it)
    assert b.index["k5"] < BOUNDARY and b.index["k0"] >= BOUNDARY, (b.index["k5"], b.index["k0"])
    assert b.index["k4"] == 1  # depth 1 is still center (boundary 2)
    # center (depth < boundary): confirm, refresh, deliver nothing
    d0 = dog.reference("p", "k5")
    assert (d0.found, d0.resolution, d0.value, d0.confidence) == (True, "in_context", None, 1.0)
    assert d0.provenance["matched_by"] == "key"
    top_before = b.index["k4"]
    dog.reference("p", "k4")            # depth 1: refresh pulls it toward the centre
    assert b.index["k4"] <= top_before and b.index["k4"] == 0, "center reference did not refresh"
    # deep: stub by default, full when expanded within budget
    ds = dog.reference("p", "k0")
    assert ds.resolution == "stub" and ds.stub.key == "k0" and ds.stub.summary
    api = b.get("k0")[0]
    before_refs, before_strength = api.refs, api.strength
    df = dog.reference("p", "k0", budget_chars=1000, expand=True)
    assert df.resolution == "full" and df.value == api.value
    assert api.refs == before_refs + 1 and api.strength > before_strength, "reference did not strengthen"
    # budget smaller than the value -> stub, never a partial value
    dsmall = dog.reference("p", "k0", budget_chars=3, expand=True)
    assert dsmall.resolution == "stub" and dsmall.value is None and len(dsmall.stub.summary) <= 3
    # alias (near-dup merged key) at 0.9, search fallback at 0.6
    st2 = MemoryStore(); g = LogGate(st2); dog2 = Watchdog(st2)
    emb = np.ones(4)
    g.ingest("p", "canon", "the twin fact", emb=emb)
    g.ingest("p", "dup", "the twin fact copy", emb=emb * (1 + 0.01))
    assert st2.branch("p").aliases.get("dup") == "canon"
    da = dog2.reference("p", "dup")
    assert (da.confidence, da.provenance["matched_by"]) == (0.9, "alias")
    miss = dog2.reference("p", "nothing like this exists anywhere").found
    assert miss is False
    hit = dog2.reference("p", "twin fact")
    assert hit.found and (hit.confidence, hit.provenance["matched_by"]) == (0.6, "search"), hit
    # promotion needs repeated references (hysteresis)
    st3 = MemoryStore(); dog3 = _mk_peer(st3, [(f"m{i}", f"value {i}") for i in range(6)])
    assert st3.branch("p").index["m0"] >= BOUNDARY
    for _ in range(dog3.promote_votes):
        d = dog3.reference("p", "m0")
    assert d.promoted is True and st3.branch("p").index["m0"] == 0, "no promotion after votes"
    # miss cost == a full descent
    dm = dog3.reference("p", "no such key at all")
    assert dm.found is False and dm.hops == len(st3.branch("p").levels)
    print("  reference: in_context/full/stub/miss, key=1.0 alias=0.9 search=0.6, votes->promotion, refs strengthen")


def test_aside_is_honest_and_does_not_repeat():
    st = MemoryStore()
    dog = _mk_peer(st, [("k1", "a fact about tea"), ("k2", "a fact about coffee")])
    assert dog.aside("p", []) is None, "spoke with nothing to say"
    node = st.branch("p").get("k1")[0]
    node.mentions = [1, 2, 3, 4, 5]
    line = dog.aside("p", [])
    assert line and "keep coming back to" in line, line
    assert dog.aside("p", []) is None, "repeated the same thought twice running"
    print("  aside: silent on a plain tree, reads state only, never repeats itself twice running")


# ---------------------------------------------------------------------- runtime
def test_split_for_window_conserves_and_bounds():
    text = " ".join(f"Sentence number {i} is short." for i in range(60))
    spans, tail = RT.split_for_window(text, tail_words=40, span_words=20)
    assert " ".join(spans + [tail]).split() == text.split(), "words were lost or reordered"
    assert len(tail.split()) <= 40 + 5, len(tail.split())
    for s in spans:
        assert len(s.split()) <= 20, len(s.split())
    assert RT.split_for_window("short message", 40, 20) == ([], "short message")
    print(f"  split_for_window: no word lost, tail<={len(tail.split())}, every span<=20")


def test_runtime_helpers():
    assert RT.whisper_query("a b c d e f g h i j k l m n o p q r", 16) == "c d e f g h i j k l m n o p q r"
    assert RT.whisper_query("word " + RT.FILED_NOTE + " tail words here", 2) == "words here"
    body = f"{RT.MEM_REF}k1{RT.MEM_REF_END}{RT.MEM}[k1] value{RT.MEM_END}the answer"
    assert RT.clean_reply(body) == "the answer", RT.clean_reply(body)
    assert RT.clean_reply("plain text") == "plain text"
    assert RT.window_budgets(1024) == {"tail_words": 204, "span_words": 80, "entry_chars": 307}
    assert RT.window_budgets(100) == {"tail_words": 60, "span_words": 80, "entry_chars": 200}
    st = MemoryStore(); g = LogGate(st)
    shown, qkey, tail = RT.file_user_turn(g, "p", 3, "hello there", 40, 20)
    assert shown == "hello there" and qkey == "turn3:user" and tail == "hello there"
    assert st.branch("p").get("turn3:user")[0].value == "hello there"
    long_text = " ".join(f"Sentence number {i} is short." for i in range(60))
    shown2, qkey2, _ = RT.file_user_turn(g, "p", 4, long_text, 40, 20, move=True)
    assert shown2.startswith(RT.FILED_NOTE) and qkey2 == "turn4:user"
    assert any(k.startswith("turn4:part") for k in st.branch("p").index), "spans not filed"
    print("  whisper_query/clean_reply/window_budgets/file_user_turn keys and move=True filing")


# ---------------------------------------------------------------------- persist
def test_persist_roundtrip_is_exact():
    st = MemoryStore(); g = LogGate(st)
    g.ingest("p", "k1", "the first fact", emb=np.arange(4, dtype=float))
    g.ingest("p", "k2", "the first fact, edited")
    g.ingest("q", "k1", "shared fact")
    dog = Watchdog(st)
    dog.reference("q", "k1")
    st.branches["p"].clock = 7
    strengthen(st.branch("p").get("k2")[0], 7)
    d1 = P.dump(st, {"turn": 9})
    loaded, meta = P.load(json.loads(json.dumps(d1)))          # through a real JSON hop
    d2 = P.dump(loaded, meta)
    assert d1 == d2, "dump/load/dump is not a fixed point"
    n1 = st.branch("p").get("k2")[0]; n2 = loaded.branch("p").get("k2")[0]
    for f in ("value", "strength", "s_clock", "mentions", "history", "aliases", "refs",
              "votes", "touched", "created", "visibility", "key", "peer"):
        assert getattr(n1, f) == getattr(n2, f), f
    assert list(st.branch("p").levels) and len(st.branch("p").levels) == len(loaded.branch("p").levels)
    assert [list(l) for l in st.branch("p").levels] == [list(l) for l in loaded.branch("p").levels]
    assert loaded.bridges.neighbors(("p", "k1")).get(("q", "k1")) == 1.0
    assert "firs" in loaded.words["p"] or "fact" in loaded.words["p"], "word index not rebuilt"
    print("  persist: dump/load/dump identical (incl. level order, bridges, strength, history)")


# ----------------------------------------------------------- model-side math
def test_ref_numpy_sparse_matches_dense():
    rng = np.random.default_rng(0)
    T, d, w, g = 64, 6, 5, 2
    q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
    dense = R.sparse_attention_dense(q, k, v, w, g)
    gather, visits = R.sparse_attention_gather(q, k, v, w, g)
    assert np.allclose(dense, gather, atol=1e-12), np.abs(dense - gather).max()
    idx = R.sparse_indices(T, w, g)
    for t in range(T):
        assert visits[t] == len(idx[t][idx[t] >= 0]) <= w + g
    print(f"  sparse attention: gather == dense (max {np.abs(dense-gather).max():.1e}), visits <= w+g")


def test_ref_numpy_tree_mixer_matches_naive():
    rng = np.random.default_rng(1)
    for T, J in ((1, 1), (7, 3), (16, 4), (37, 6)):
        d = 5
        q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
        lam = np.abs(rng.normal(size=J + 1)) + 0.1
        gam = np.abs(rng.normal(size=max(J - 1, 0))) * 0.9
        naive = R.tree_mixer_naive(q, k, v, lam, gam)
        rec, stats = R.tree_mixer_recurrent(q, k, v, lam, gam)
        assert np.allclose(naive, rec, atol=1e-12), (T, np.abs(naive - rec).max())
        for t in range(T):
            assert stats["reads"][t] == bin(t).count("1") + 1, (t, stats["reads"][t])
            assert stats["reads"][t] <= J + 1
        assert stats["max_states"] <= max(1, math.floor(math.log2(T)))
    print("  tree mixer: recurrent == naive (1e-12); reads[t] == popcount(t)+1 <= log2(T)+1")


def test_torch_matches_numpy_references():
    try:
        import torch
        from memtree.tree_mixer import LogLinearTreeMixer
        from memtree.tree_mixer import phi as phi_t
        from memtree.phi_adapter import tree_rows, xor_levels as t_xor
        from memtree.sparse_attn import sparse_attention as sparse_t
    except Exception as e:  # pragma: no cover
        print(f"  SKIP torch checks ({type(e).__name__})")
        return
    rng = np.random.default_rng(2)
    assert np.allclose(phi_t(torch.tensor([[-3.0, 0.0, 2.0]])).numpy(), R.phi(np.array([[-3.0, 0.0, 2.0]])))
    d, H = 6, 2
    for T, J in ((1, 1), (9, 4), (16, 4), (37, 6)):
        m = LogLinearTreeMixer(H, J).double()
        with torch.no_grad():
            m.lam_raw.normal_(); m.gam_raw.normal_(mean=1.0)
        q, k, v = (torch.tensor(rng.normal(size=(1, H, T, d))) for _ in range(3))
        rec, dense = m(q, k, v).detach(), m.forward_dense(q, k, v)
        assert torch.allclose(rec, dense, atol=1e-12)
        lam, gam = m.lam().detach().numpy(), m.gam().detach().numpy()
        for h in range(H):
            ref = R.tree_mixer_naive(q[0, h].numpy(), k[0, h].numpy(), v[0, h].numpy(), lam[h], gam[h])
            assert np.allclose(rec[0, h].numpy(), ref, atol=1e-10), (T, h)
    # chunked tree branch == the numpy tree formula (needs max_levels >= log2(chunk))
    T, d, H = 100, 6, 2
    q, k, v = (torch.tensor(rng.normal(size=(1, H, T, d))) for _ in range(3))
    m = LogLinearTreeMixer(H, 7).double()
    with torch.no_grad():
        m.lam_raw.normal_(); m.gam_raw.normal_(mean=1.0)
    lam, gam = m.lam().detach().numpy(), m.gam().detach().numpy()
    for Tq in (T, 4, 20):
        with torch.no_grad():
            tr = tree_rows(m, phi_t(q[:, :, -Tq:]), phi_t(k), v).numpy()[0]
        for h in range(H):
            lev = R.xor_levels(Tq, T); wlev = R._level_weights(lam[h], gam[h])
            W = np.where(lev >= 0, wlev[np.maximum(lev, 0)], 0.0)
            a = (R.phi(q.numpy()[0, h, -Tq:]) @ R.phi(k.numpy()[0, h]).T) * W
            ref = (a @ v.numpy()[0, h]) / (a.sum(-1, keepdims=True) + 1e-6)
            assert np.allclose(tr[h], ref, atol=1e-10), (Tq, h, np.abs(tr[h] - ref).max())
    assert (t_xor(5, 77, "cpu").numpy() == R.xor_levels(5, 77)).all()
    # torch sparse attention == numpy gather
    T, d, w, g = 40, 6, 4, 2
    q, k, v = (torch.tensor(rng.normal(size=(1, 1, T, d))) for _ in range(3))
    got = sparse_t(q, k, v, w, g).numpy()[0, 0]
    want, _ = R.sparse_attention_gather(q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy(), w, g)
    assert np.allclose(got, want, atol=1e-10)
    print("  torch: phi, mixer recurrent==dense==numpy, chunked tree_rows==numpy, sparse==gather")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
