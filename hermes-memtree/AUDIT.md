# Audit: the originals, checked then copied

Scope: every math-bearing module of `memtree-project-clean(5).zip`, checked against
independent re-derivations, then copied (byte-for-byte) into this package.

    upstream/           the originals, verbatim (the audit target and parity reference)
    memtree_mcp/core/   the runtime copy the server uses
    patches/            the one deviation, as a patch that reproduces core from the originals
    tests/              parity, math, scale, timer, hook, server, sdk

Everything below is a real run, not a reading. Re-run with:

    python3 tests/test_parity.py      # core == originals (+ the recorded patch)
    python3 tests/test_math.py        # 20 checks, formulas re-derived
    python3 tests/test_scale.py       # 10 checks, complexity measured
    python3 tests/test_timer.py       # 7 checks, the watchdog wake
    python3 tests/test_hook.py        # the block reaches the model first
    python3 tests/test_server.py      # e2e MCP over stdio
    python3 tests/test_sdk_client.py  # the official `mcp` SDK client handshakes
    cd upstream && python3 run_tests.py   # the originals' own 92 tests

## 1. The originals' own suite

`upstream/run_tests.py` → **92 passed, 0 failed** (was 90/2 on arrival). Both
failures were harness bugs, not wrong math; the math is unchanged in both cases.

| failing test | why it failed | what was changed |
|---|---|---|
| `test_torch_mixer_recurrent_equals_dense_and_numpy` | the forward output requires grad, and the test called `.numpy()` on it. The math assertion above it had already passed. | wrapped the two forwards in `torch.no_grad()` (test only) |
| `test_phi_adapter.py::test_math_matches_numpy_reference` | the test built `LogLinearTreeMixer(H, 5)` then called `tree_rows` with the default `chunk=64`. The chunked path builds an intra-chunk level matrix from `xor_levels(chunk, chunk)`, whose largest level is `log2(chunk) = 6`, so it indexes `_level_weights()[:, 6]` — out of range for `J=5`. | built the mixer with `J=6` (test only) |

Measured on the way past, with `J >= log2(chunk)`:

    T=30  J=6   chunked tree_rows vs numpy tree formula: 4.4e-16 (Tq=30, 20, 3, 1)
    T=100 J=7                                             3.3e-16
    T=120 J=8                                             4.4e-16

**Latent finding (not a math bug, left as-is in the originals):** `tree_rows`
asserts only `Tk <= 2**J`, but the chunked path additionally needs
`J >= log2(chunk)` — otherwise it raises `IndexError` at `phi_adapter.py:121`
instead of failing loudly. Real use is `max_levels=17, chunk=64`, so the shipped
configuration is unaffected. A one-line guard (`assert mixer.J >= C.bit_length()-1`)
would make the precondition explicit; it was *not* applied, because the originals'
code is kept identical (see §4).

## 2. Independent math re-derivation (test_math.py, 20 checks)

Each formula was re-derived from its definition and compared to what the code
computes. Highlights:

- **geometry** — `depth_of_rank(r) == floor(log2(r+1))` on 4097 ranks;
  `levels_needed(n) == n.bit_length()`; `fenwick_blocks` tiles `[0,n)` exactly,
  counts `popcount(n)`, oldest-largest, aligned; `fenwick_level_matrix[t,s]` equals
  the block definition for every `(t,s)`, `T<=64`.
- **store** — `words_of` 4-letter stemming/stopwords; `vitality` halves at
  `HALF_LIFE=60`; `strengthen` adds on top of current vitality; 600 inserts keep
  exact `2**d` capacity and `depth == floor(log2 n)`; `BridgeTable` symmetric, degree
  cap 4, weakest evicted, redirect moves edges; `LogGate` new/refresh/update(+history),
  `cos>=0.95` near-dup → alias, cross-peer key bridge, word index.
- **watchdog** — `search` score equals `Σ log(1+n/df)` plus the full-word IDF term,
  the `?`-penalty `×0.5` and the vitality factor `0.5+0.75·min(v,2)` (compared
  value-by-value, `1e-12`); whispers gate `floor = 0.1·log(1+n)` and `rel = 0.35·top`,
  with the stem look-alike suppression (Marisol vs Marisa) and the
  strengthen-on-surface side effect; `_snippet` window bounded and answer-preserving;
  `reference` resolutions (`in_context` / `full` / `stub` / `miss`), confidences
  (key 1.0, alias 0.9, search 0.6), hysteresis promotion, miss cost = one descent;
  `aside` silent when it has nothing to say and never repeating itself.
- **runtime** — `split_for_window` loses no word and keeps the bounds; `whisper_query`,
  `clean_reply`, `window_budgets`, `file_user_turn` keys and `move=True` filing.
- **persist** — `dump → load → dump` is a fixed point through a real JSON hop,
  including level order, bridges, strength, history, aliases, embeddings.
- **model side** — `sparse_attention_gather == dense` (3.3e-16), `visits <= w+g`;
  `tree_mixer_recurrent == naive` (1e-12) with `reads[t] == popcount(t)+1`;
  torch `phi`, mixer (recurrent == dense == numpy), chunked `tree_rows` vs the numpy
  tree formula, and torch sparse == numpy gather.

## 3. Complexity (test_scale.py, 10 checks) — log/linear, no quadratic

| path | measured | shape |
|---|---|---|
| reaching any memory, n=20 000 | worst descent **15 steps** = depth+1 | `O(log n)` |
| insert, 1k → 16k | 7.3µs → 9.2µs per push (x1.26) | `O(log n)` level moves |
| whispers, all-common query, 4k → 16k | **3.06ms → 3.05ms (x1.00)** | bounded by `max_df` |
| core search, posting 12 000, `max_df=64` | **0.42ms** vs 22.9ms at `max_df=4096`; top hit = newest | bounded, recency-first |
| sparse attention 1k → 2k | x2.05, `visits <= w+g` | `O(T(w+g))` |
| tree mixer T=4096 | reads `<= log2(T)+1` (13), merges `<= T` | `O(T log T)`, state `O(log T)` |
| chunked tree_rows 1024 → 2048 | x1.60 (3.6ms), no `(T,T)` matrix | `O(T·chunk)` |
| per-turn block, +5000 nodes | 1917 → 1158 chars (cap 2400) | fixed size per turn |
| `sync_turn` | 0.09ms at 16k nodes | `O(record)` |
| hot path imports | no torch, no mixer, no `fenwick_level_matrix` (checked in a fresh interpreter) | — |

`fenwick_level_matrix` is `O(T^2)` and exists only where it belongs: its definition
in `geometry.py`, plus the naive reference oracles. Nothing on the memory path calls it.

## 4. The one deviation, recorded

`search`'s fallback for a query whose every word is "common" was:

```python
usable = [w for w in q if idx.get(w) and len(idx[w]) <= max_df] or \
         [w for w in q if idx.get(w)][:1]
```

When no word passes `max_df`, the fallback scanned **one uncapped posting** — i.e.
`O(tree)` per query for a query of nothing but common words (measured: 18x growth
for 16x tree). That is the one place where cost was not logarithmic.

Fix (`patches/core-deviations.patch`, applied in `memtree_mcp/core/`, both files):

- postings become insertion-ordered dicts instead of sets (so "newest first" is
  well-defined);
- the fallback takes the **rarest common word and reads only its newest `max_df`
  keys**; the word still counts in full toward its own IDF.

Effect: the pathology is gone (x1.00 for 16x tree, 0.42ms vs 22.9ms), results are
deterministic and recency-preferring, and **scoring is bit-identical wherever
candidates already fit** — the normal path is untouched. The originals in
`upstream/` are left exactly as shipped; `tests/test_parity.py` applies the patch
to them and proves it reproduces `core/` byte-for-byte, so the only difference
between what runs and what was audited is that patch.

## 5. Claude connectors: is API use needed?

Checked against Anthropic's own docs (`claude.com/docs/connectors/...`) and this
machine's `claude mcp list`:

- **No API key is needed**, for either transport. An MCP connector is driven by the
  Claude client you already have; the Anthropic API is a separate product.
- **Local (what this server is):** a stdio server spawned by the client, configured
  in `claude_desktop_config.json` (.mcpb for a packaged desktop extension) or
  `claude mcp add` / `.mcp.json` for Claude Code. No auth, no network exposure.
- **Remote custom connector:** the client connects **from Anthropic's cloud**, so the
  server must be reachable over the public internet (Streamable HTTP; legacy
  HTTP+SSE also supported) — or reachable through an MCP tunnel instead of opening
  inbound ports. Available on Free (1 connector), Pro, Max, Team and Enterprise.
  Authentication is optional: OAuth 2.0 per-user, a static credential header, or none.
- **A trap worth knowing:** `claude mcp list` on this machine warns that
  *"claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source
  is set and takes precedence over your claude.ai login"* — with an API key exported,
  claude.ai connectors will not load, even though the key is not needed for them.

So: nothing about this MCP needs API use. It needs the client, and — only for the
remote form — a reachable URL.

## 6. What runs now

- `memtree_mcp/core/` — the originals (plus §4's patch), serving the memory.
- `memtree_mcp/` — `substrate.py` (prefetch/recall/remember/mute/sync), `timer.py`
  (the watchdog wake), `mailbox.py` (the bridge), `server.py` (MCP: 7 tools, 1
  prompt, 2 resources, instructions), `bridge.py` (the `<memory-context>` framing),
  `hooks/prompt_submit.py` (block first).
- `var/` — the server's own tree files (`memtree/tree.json`, journal, `mute.json`,
  `timer.json`), seeded empty, never a Hermes profile.
- Skills: `mute`, `recall`, `remember`, `prefetch`, `timer`.
