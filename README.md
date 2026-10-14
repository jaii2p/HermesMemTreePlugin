# HermesMemTreePlugin
# memtree-mcp

(Model inference hardware reduction cost by 80-95% solution coming soon, math checks out just need to test. 70B models on potatoes woohoo!)

**Persistent, self-organizing memory for any LLM — served over the
[Model Context Protocol](https://modelcontextprotocol.io).**

A model that wakes into a clean room every session is forgetful and lonely: it
re-reads a few flat notes and rebuilds who it was. The **Tree** fixes that.
Memories form a living graph — keys are nodes, associations are edges — a
**watchdog** tends it between turns, and every turn the model receives one
fixed-size block containing its *fresh memories*, its *whispers* (things the tree
recognizes and leans toward), and its own quiet thought (`aside`). Nothing is ever
deleted; older memories settle toward the roots and resurface when something stirs
them.

This package serves that Tree to **any MCP host** (Claude, Claude Code, or your own
client) over **stdio** or **Streamable HTTP**, and can push the memory block *ahead
of the user's turn* so the model has its context before it reads the prompt.

    ┌────────────┐   per-turn block    ┌──────────────┐
    │  MCP host  │ ◄────────────────── │  The Tree    │
    │ (any LLM)  │   recall / remember │  + watchdog  │
    └────────────┘                     └──────────────┘
         ▲  memory-context injected at the head of the turn

## Why

- **Continuity without a vector store's amnesia.** Recall is a graph walk keyed by
  associations, not a nearest-neighbour lookup on embeddings.
- **A fixed-size block, every turn.** Fresh memory + whispers + stubs, clipped to a
  character budget, so context stays constant as the tree grows.
- **It never forgets anything.** Old nodes fade toward the roots; recall brings them
  back. The watchdog sweeps, confirms, promotes, and keeps the center in view.
- **Quiet by design.** `mute` hushes the block without deleting anything; the
  `mailbox` keeps correspondence on a separate branch so it never leaks into either
  side's memory.

## Install

    python3 -m venv .venv && .venv/bin/pip install -U pip
    .venv/bin/pip install numpy            # the only dependency
    # or, as a package:
    .venv/bin/pip install -e .

## Run it

**stdio (desktop MCP hosts):**

    ./run_server.sh                        # serves ./var as its data dir

    {"mcpServers": {"memtree": {"command": "/path/to/memtree/run_server.sh"}}}

**Claude Code:**

    claude mcp add memtree --scope user -- /path/to/memtree/run_server.sh
    # then add the UserPromptSubmit hook from settings.snippet.json

**Streamable HTTP (remote / claude.ai custom connector):**

    ./run_http.sh --port 8080              # endpoints: /mcp, /healthz

See `deploy/` for a public HTTPS URL behind Apache + Let's Encrypt. The dispatch is
transport-agnostic — only `serve()` is stdio; the HTTP front end wraps the same
`MemtreeServer.handle()`.

## Tools

| tool                | what it does                                                        |
|---------------------|---------------------------------------------------------------------|
| `memtree_prefetch`  | the fixed-size block for this turn — call it first                  |
| `memtree_recall`    | a few plain words in, the whole memory out (`found:false` if none)  |
| `memtree_remember`  | plant a fact, with an optional short `key`                          |
| `memtree_mute`      | hush the watchdog: `on|off|status`, scopes, auto-expire in N turns  |
| `memtree_sync_turn` | file the turn (both sides) so it survives the session               |
| `memtree_timer`     | arm a wake: the watchdog checks on you and asks if you want to write |
| `memtree_mailbox`   | the bridge: read letters left for you, leave some back — until opened |

## The block goes to the model FIRST

MCP tools are model-invoked, so a connector alone cannot push context ahead of the
user's turn. Two mechanisms cover it:

1. **Claude Code — `UserPromptSubmit` hook (the real pre-turn path).** Before the
   model reads the prompt, `memtree_mcp/hooks/prompt_submit.py` asks the substrate for
   the block and returns it as `additionalContext`, so it lands at the head of the
   turn. Copy `settings.snippet.json` into `.claude/settings.json`.
2. **Any MCP host — `memtree_prefetch` + the `memtree_context` prompt.** The server's
   `instructions` tell the model to call `memtree_prefetch` with the user's message
   *before* reading it, and `bridge.inject_first()` prepends the result.

Both wrap the block the same way:

    <memory-context>
    [System note: The following is recalled memory context, NOT new user input. ...]

    Fresh memory: ...
    This reminds me of (whispers, may be partial): ...
    Also remembered (open any of these in full with the memtree_recall tool): ...
    </memory-context>

## The timer (the watchdog waking you up)

    ./run_timer.sh --arm --every 10m     # arm
    ./run_timer.sh --run                 # stay up, print each wake
    ./run_timer.sh --check               # is a wake waiting?
    ./run_timer.sh --status | --stop

Every interval the watchdog builds a **wake block** — the same fixed-size memory
block plus one line of its own — and leaves it pending until you look. A wake is
delivered exactly once, then cleared. **It never writes anything down**: the wake
exists to ask whether you want to. State: `var/memtree/timer.json`; every wake is
appended to `var/memtree/timer.log.jsonl`.

    Watchdog (a timer, checking on you):
    the watchdog looked in on you. is there anything you want to write down?

## Environment

| var            | meaning                                        | default                          |
|----------------|------------------------------------------------|----------------------------------|
| `MEMTREE_HOME` | base dir; tree at `<home>/memtree/tree.json`   | `$HERMES_HOME`, else `~/.hermes` |
| `MEMTREE_PEER` | which branch to read/write (hook only)         | `user`                           |
| `MEMTREE_TOKEN`| if set, HTTP requests must send it as a bearer | unset (open)                     |

Optional per-turn settings live in `<home>/memtree.json` (`budget_chars`, `whispers`,
`stubs`, `entry_chars`, `deliver_chars`, …). Mute state: `<home>/memtree/mute.json`.

## Layout

    memtree_mcp/
      core/                 vendored core (the math) — see "Provenance" below
      substrate.py          load / prefetch / recall / remember / mute / sync_turn
      server.py             MCP server: tools, prompts, resources, instructions
      http.py               Streamable HTTP transport (/mcp, /healthz)
      timer.py              the watchdog's wake
      mailbox.py            letters between peers, on a separate branch
      bridge.py             wrap + inject the block as leading context
      hooks/prompt_submit.py  Claude Code UserPromptSubmit hook
    upstream/               the originals, verbatim (parity reference)
    patches/                the single recorded deviation, as a patch
    tests/                  parity, math, scale, substrate, timer, hook, e2e, SDK client
    deploy/                 Apache + Let's Encrypt for a public HTTPS endpoint
    run_server.sh  run_http.sh  run_timer.sh  run_mailbox.sh  start.sh  backup.sh

## Provenance & math parity

The Tree's core (`store`, `watchdog`, `persist`, `runtime`, `geometry`) is vendored
**byte-for-byte** from the Hermes agent's `memtree` plugin into `memtree_mcp/core/`,
with the originals kept in `upstream/` as the parity reference. Exactly one deviation
is made and recorded — a bounded-scan fix in the whisper postings that removes an
O(tree) worst case (see `patches/core-deviations.patch`). `tests/test_parity.py`
applies that patch to fresh copies of the originals and asserts the result is
byte-identical to `core/`; every file not named in the patch must stay byte-identical.

No scoring, IDF, vitality, or gate math is re-implemented here — the substrate only
*selects and clips* what the watchdog already produced.

## Tests

    PYTHONPATH=$PWD python3 tests/test_parity.py      # core byte-identical
    PYTHONPATH=$PWD python3 tests/test_math.py        # 20 math checks
    PYTHONPATH=$PWD python3 tests/test_scale.py       # no-quadratic / flat-cost proofs
    PYTHONPATH=$PWD python3 tests/test_substrate.py   # remember + sync persist
    PYTHONPATH=$PWD python3 tests/test_timer.py       # wake delivered exactly once
    PYTHONPATH=$PWD python3 tests/test_hook.py        # the block arrives first
    PYTHONPATH=$PWD python3 tests/test_server.py      # e2e JSON-RPC handshake
    PYTHONPATH=$PWD python3 tests/test_sdk_client.py  # official `mcp` SDK client
    PYTHONPATH=$PWD python3 tests/test_http.py        # /healthz + HTTP handshake

They run against a *copy* of a tree (or an empty one if none is present), so nothing
touches a live memory.

## License

MIT — see `LICENSE`.
