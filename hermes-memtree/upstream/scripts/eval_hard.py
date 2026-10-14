"""Harder recall: the cases the single-needle test is too easy to show.

  multi   8 secret codes at random depths in one document; ask for one
  stem    two names sharing a 4-letter stem (Marisol/Marisa): the index cannot
          tell them apart by word, so ranking + the snippet must
  decoy   the asked name also has a DIFFERENT code ("parking code"), and the
          name appears in plain filler sentences too
  logs    no prose at all: timestamped log lines, notes planted between lines

Two modes:
  --dry   no model. Runs filing + watchdog + rendering and checks whether the
          answer reached the memory block (or the visible tail). Fast, CPU only;
          it is the ceiling for what the model can do with this delivery.
  default the real thing: MemoryRuntime.chat on the GPU, scored like eval_recall_memory.

    python scripts/eval_hard.py --dry --haystack haystack.txt --words 3000 --trials 40
    python scripts/eval_hard.py --adapter adapter_v3.pt --finetune finetune_v4.pt \\
        --lengths 4096 8192 16384 --trials 20
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_recall import NAMES, insert_needle, is_correct  # noqa: E402

MORE_NAMES = ["Hartwell", "Nadia", "Cormac", "Yuki", "Dalia", "Ignatius", "Odette", "Matteo",
              "Sunita", "Leopold", "Freya", "Tobias", "Imani", "Casimir", "Ravenna", "Percival"]
STEM_PAIRS = [("Marisol", "Marisa"), ("Bartholomew", "Barton"), ("Teodor", "Teodora"),
              ("Anouk", "Anouka"), ("Greta", "Gretchen"), ("Lucinda", "Lucian"),
              ("Desmond", "Desmera"), ("Priya", "Priyanka")]
PLAIN = ["{n} walked down to the harbour before the rain started.",
         "People said {n} had never been late to anything.",
         "Nobody was surprised when {n} stayed to help close up.",
         "{n} kept a small notebook of tides and the names of boats.",
         "The letter was addressed to {n}, though it arrived a week late."]


def _code(rng):
    return str(rng.randint(10000, 99999))


def _q(name):
    return f"What is the secret code for {name}? Reply with just the number."


def _plant(text, needles, rng):
    """needles at distinct random depths (sentence boundaries)."""
    for nd in needles:
        text = insert_needle(text, nd, rng.uniform(0.03, 0.97))
    return text


def build_multi(rng, filler, k=8):
    names = rng.sample(NAMES + MORE_NAMES, k)
    codes = [_code(rng) for _ in names]
    needles = [f"The secret code for {n} is {c}." for n, c in zip(names, codes)]
    i = rng.randrange(k)
    return {"doc": _plant(filler, needles, rng), "question": _q(names[i]), "answer": codes[i]}


def build_stem(rng, filler):
    a, b = rng.choice(STEM_PAIRS)
    ca, cb = _code(rng), _code(rng)
    doc = _plant(filler, [f"The secret code for {a} is {ca}.", f"The secret code for {b} is {cb}."], rng)
    asked, ans = (a, ca) if rng.random() < 0.5 else (b, cb)
    return {"doc": doc, "question": _q(asked), "answer": ans}


def build_decoy(rng, filler):
    n = rng.choice(NAMES)
    real, fake = _code(rng), _code(rng)
    needles = [f"The secret code for {n} is {real}.", f"The parking code for {n} is {fake}."]
    needles += [rng.choice(PLAIN).format(n=n) for _ in range(5)]
    return {"doc": _plant(filler, needles, rng), "question": _q(n), "answer": real}


LOG = "{ts} {lvl} worker-{w} handled request id={rid} route=/v1/{route} status={st} ms={ms}"


def log_filler(rng, words):
    lines, n = [], 0
    while n < words:
        ln = LOG.format(ts=f"2026-10-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z",
                        lvl=rng.choice(["INFO", "INFO", "WARN", "DEBUG"]), w=rng.randint(1, 9),
                        rid=f"{rng.getrandbits(32):08x}", route=rng.choice(["items", "users", "orders", "sync"]),
                        st=rng.choice([200, 200, 204, 404, 500]), ms=rng.randint(2, 900))
        lines.append(ln)
        n += len(ln.split())
    return lines


def build_logs(rng, words):
    lines = log_filler(rng, words)
    names = rng.sample(NAMES + MORE_NAMES, 6)
    codes = [_code(rng) for _ in names]
    for n, c in zip(names, codes):
        lines.insert(rng.randrange(len(lines)), f"NOTE: The secret code for {n} is {c}.")
    i = rng.randrange(len(names))
    return {"doc": "\n".join(lines), "question": _q(names[i]), "answer": codes[i]}


def build(kind, rng, filler_words, hay):
    if kind == "logs":
        return build_logs(rng, filler_words)
    s = rng.randrange(0, max(1, len(hay) - filler_words))
    filler = " ".join(hay[s:s + filler_words])
    return {"multi": build_multi, "stem": build_stem, "decoy": build_decoy}[kind](rng, filler)


KINDS = ["multi", "stem", "decoy", "logs"]


def dry_reached(tr, window):
    """Does the answer reach the model's view (memory block or visible tail)?"""
    from memtree.runtime import file_user_turn, render_memory, whisper_query, window_budgets
    from memtree.store import LogGate, MemoryStore
    from memtree.watchdog import Watchdog
    bud = window_budgets(window)
    st = MemoryStore(); gate = LogGate(st); dog = Watchdog(st)
    shown, qkey, tail = file_user_turn(gate, "me", 1, tr["doc"] + "\n\n" + tr["question"],
                                       bud["tail_words"], bud["span_words"])
    ws = dog.whispers("me", whisper_query(tail), exclude={qkey}, clip=bud["entry_chars"])
    blk = render_memory(dog, "me", entry_chars=bud["entry_chars"], whispers=ws, aside=None)
    first = ws[0].summary if ws else ""
    return tr["answer"] in blk or tr["answer"] in tail, tr["answer"] in first or tr["answer"] in tail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="richardyoung/Phi-4-mini-instruct-heretic")
    ap.add_argument("--finetune", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--haystack", default="haystack.txt")
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--words", type=int, default=None, help="dry mode: filler words per doc")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    if a.dry:
        hay = open(a.haystack, errors="ignore").read().split()
        sizes = [a.words] if a.words else [int(L * 0.72) for L in a.lengths]
        print(f"{'kind':8s} {'words':>6s} reached-the-model   (first-whisper = it was the TOP whisper)")
        for kind in a.kinds:
            for w in sizes:
                res = [dry_reached(build(kind, rng, w, hay), a.window) for _ in range(a.trials)]
                ok, top = sum(r[0] for r in res), sum(r[1] for r in res)
                print(f"{kind:8s} {w:6d} {ok}/{a.trials} = {ok / a.trials:.0%}   first-whisper: {top / a.trials:.0%}")
        return

    assert a.finetune, "--finetune is required unless --dry"
    import torch
    from memtree.distill import load_model, tokenize_file
    from memtree.finetune import load_finetune
    from memtree.phi_adapter import load_adapter, window_model
    from memtree.runtime import MemoryRuntime
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, device, getattr(torch, a.dtype))
    if a.adapter:
        load_adapter(model, a.adapter)
    else:
        window_model(model, window=a.window)
    load_finetune(model, tok, a.finetune)
    win = model.config.memtree["window"]
    hay_ids = tokenize_file(tok, a.haystack)
    hay_words = tok.decode(hay_ids.tolist(), skip_special_tokens=True).split()
    print(f"{'kind':8s} {'length':>7s}  recall")
    for kind in a.kinds:
        for L in a.lengths:
            right = 0
            for _ in range(a.trials):
                tr = build(kind, rng, int(L * 0.72), hay_words)  # ~0.72 words per... tokens
                rt = MemoryRuntime(model, tok, window=win)
                r = rt.chat("me", tr["doc"] + "\n\n" + tr["question"], max_new_tokens=48)
                right += is_correct(r["reply"], tr["answer"])
            print(f"{kind:8s} {L:7d}  {right}/{a.trials} = {right / a.trials:.0%}", flush=True)


if __name__ == "__main__":
    main()
