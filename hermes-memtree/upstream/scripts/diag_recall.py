"""Where does a failing recall trial actually break?  Same trials as
eval_recall_memory, but for each one it prints the facts and scores the correct
answer's log-probability under controlled variants of the SAME prompt:

  teacher   original Phi attention (full) on the memtree prompt  -> is the prompt itself answerable?
  wide      her attention, memory block globally attendable       -> the shipped runtime
  narrow    her attention, only 4 global tokens                    -> the old behaviour
  notree    wide, tree gate forced to 0 (window + head only)       -> does the tree help or hurt?

Reading it:  teacher high, wide low   -> the prompt is fine, her attention is not reading it
             teacher low  too         -> the prompt/whisper is not carrying the answer
             notree > wide            -> the tree branch is washing the exact digits out

    python scripts/diag_recall.py --adapter adapter_v3.pt --finetune finetune_v4.pt --length 4096 --trials 6
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_recall import is_correct, make_trial  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="richardyoung/Phi-4-mini-instruct-heretic")
    ap.add_argument("--finetune", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--haystack", default="haystack.txt")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import contextlib
    import torch
    from memtree.distill import load_model, tokenize_file
    from memtree.finetune import load_finetune
    from memtree.phi_adapter import (converted_layers, global_prefix, load_adapter,
                                     teacher_mode, window_model)
    from memtree.runtime import MemoryRuntime, file_user_turn
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, device, getattr(torch, a.dtype))
    if a.adapter:
        load_adapter(model, a.adapter)
    else:
        window_model(model, window=a.window)
    load_finetune(model, tok, a.finetune)
    win = model.config.memtree["window"]

    hay = tokenize_file(tok, a.haystack)
    rng = random.Random(a.seed)

    @contextlib.contextmanager
    def tree_off():
        gs = [(l, l.gate.data.clone()) for l in converted_layers(model) if getattr(l, "use_tree", False)]
        try:
            for l, _ in gs:
                l.gate.data.fill_(0.0)
            yield
        finally:
            for l, g in gs:
                l.gate.data.copy_(g)

    @torch.no_grad()
    def answer_lp(prompt_ids, answer, head, mode):
        ans = tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids = torch.cat([prompt_ids, ans], dim=1)
        ctx = {"teacher": lambda: teacher_mode(model),
               "wide": lambda: global_prefix(model, head),
               "narrow": lambda: contextlib.nullcontext(),
               "notree": lambda: contextlib.ExitStack()}[mode]()
        with ctx:
            if mode == "notree":
                with tree_off(), global_prefix(model, head):
                    logits = model(ids).logits
            else:
                logits = model(ids).logits
        lp = torch.log_softmax(logits[0, prompt_ids.shape[1] - 1:-1].float(), -1)
        return lp.gather(1, ans[0][:, None]).sum().item()

    print(f"window={win}  length={a.length}  (answer log-prob; 0 = certain, -20 = hopeless)\n")
    print(f"{'depth':>5} {'in_win':>6} {'whisper':>7} {'head':>5} {'teacher':>8} {'wide':>8} {'narrow':>8} {'notree':>8}  reply")
    for d in a.depths:
        for _ in range(a.trials):
            n = a.length - 40
            s = rng.randrange(0, len(hay) - n)
            tr = make_trial(rng, tok.decode(hay[s:s + n].tolist(), skip_special_tokens=True), d)
            text = tr["doc"] + "\n\n" + tr["question"]
            rt = MemoryRuntime(model, tok, window=win)
            rt.turn += 1
            shown, qkey, tail = file_user_turn(rt.gate, "me", rt.turn, text,
                                               rt.budget["tail_words"], rt.budget["span_words"])
            ids = rt._prompt_ids("me", shown, qkey, tail)
            head = rt._head
            full = tok.decode(ids[0], skip_special_tokens=False)
            sys_part = full.split("<|user|>")[0]
            whisper_has = tr["answer"].replace(",", "").replace(" ", "") in sys_part.replace(",", "").replace(" ", "")
            pos = len(tok(full.split(tr["answer"])[0], add_special_tokens=False)["input_ids"]) if tr["answer"] in full.split("<|user|>")[-1] else -1
            in_win = pos >= 0 and (ids.shape[1] - pos) <= win
            lps = {m: answer_lp(ids, tr["answer"], head, m) for m in ("teacher", "wide", "narrow", "notree")}
            rt2 = MemoryRuntime(model, tok, window=win)
            reply = rt2.chat("me", text, max_new_tokens=24)["reply"].replace("\n", " ")[:40]
            ok = "OK " if is_correct(reply, tr["answer"]) else "-- "
            print(f"{d:>5.1f} {str(in_win):>6} {str(whisper_has):>7} {head:>5} "
                  f"{lps['teacher']:>8.1f} {lps['wide']:>8.1f} {lps['narrow']:>8.1f} {lps['notree']:>8.1f}  {ok}{reply}", flush=True)


if __name__ == "__main__":
    main()
