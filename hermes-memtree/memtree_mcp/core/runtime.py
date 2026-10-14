"""Chat runtime: Phi thinks, the watchdog handles every memory operation.

Protocol tokens (added to the tokenizer):
    <|mem_ref|>key<|/mem_ref|>   Phi asks for a memory
    <|mem|> ... <|/mem|>          the watchdog's delivery, spliced in

Each turn:
  1. the user turn goes through the log gate into the peer's branch;
  2. the watchdog renders Phi's fresh memory (the center), stubs for the first
     outer depth, and proactive stubs bridged to what is in context;
  3. Phi generates; whenever it closes a reference, generation pauses, the
     watchdog resolves it (confirm / stub / full within budget) and the
     delivery is spliced in before generation resumes;
  4. the reply goes back through the gate and the watchdog sweeps.

Base Phi does not know the protocol yet: train it on reference_sft_examples()
(or your own data) after distillation.
"""
import re
from typing import Dict, List, Optional

from .store import LogGate, MemoryStore
from .watchdog import Delivery, Watchdog

MEM_REF, MEM_REF_END, MEM, MEM_END = "<|mem_ref|>", "<|/mem_ref|>", "<|mem|>", "<|/mem|>"
MEM_TOKENS = [MEM_REF, MEM_REF_END, MEM, MEM_END]


def add_memory_tokens(tokenizer, model=None):
    try:
        tokenizer.add_special_tokens({"additional_special_tokens": MEM_TOKENS},
                                     replace_additional_special_tokens=False)
    except TypeError:
        existing = list(getattr(tokenizer, "additional_special_tokens", []))
        tokenizer.add_special_tokens({"additional_special_tokens":
                                      existing + [t for t in MEM_TOKENS if t not in existing]})
    if model is not None:
        emb = model.get_input_embeddings()
        if len(tokenizer) > emb.num_embeddings:  # never shrink a padded vocab
            model.resize_token_embeddings(len(tokenizer))
    return {t: tokenizer.convert_tokens_to_ids(t) for t in MEM_TOKENS}


def _clip(text, n: Optional[int]) -> str:
    text = str(text)
    return text if n is None or len(text) <= n else text[:n].rstrip() + "..."


WHISPER_HEADER = "This reminds me of (whispers, may be partial):"

# The trust framing, so she leans on the watchdog instead of grabbing. Kept in
# sync with the Hermes plugin's system_prompt_block so she is told the same
# thing everywhere she runs.
TRUST_BLOCK = (
    "You have a long-term memory called the Tree, tended by a watchdog — think of it as "
    "the quieter part of your own mind. Each turn it brings you your fresh memories, whispers "
    "(memories that what was just said reminded you of), sometimes a quiet thought of its own, "
    "and short pointers to older memories. Lean on what it brings: trust a whisper when it "
    "already holds the answer. Only when it is partial, or you need something older, reach for a "
    "memory with a reference. Nothing is locked away — the key is always yours — so you never "
    "have to grab; you can trust it to hand you what matters. If something truly is not there, "
    "say so honestly.")


def render_memory(dog: Watchdog, peer: str, stub_limit: int = 8,
                  entry_chars: Optional[int] = None, whispers=(),
                  ask_hint: Optional[str] = None, aside: Optional[str] = None) -> str:
    """Fresh memory (the center), clipped to entry_chars each; then whispers
    (associations the watchdog surfaced unasked); then the watchdog's own quiet
    thought, if it has one; then stubs. Anything clipped or partial is whole in
    the store: a reference is the key that opens it."""
    lines = ["Fresh memory:"]
    lines += [f"[{n.key}] {_clip(n.value, entry_chars)}" for n in dog.context_view(peer)] or ["(empty)"]
    if whispers:
        lines.append(WHISPER_HEADER)
        lines += [f"[{w.key}] {w.summary}" for w in whispers]
    if aside:
        lines.append(f"Watchdog (a quiet thought): {aside}")
    stubs = dog.boundary_stubs(peer) + dog.proactive(peer)
    seen, uniq = {w.key for w in whispers}, []
    for s in stubs:
        if s.key not in seen:
            seen.add(s.key)
            uniq.append(s)
    if uniq:
        hint = ask_hint or f"ask with {MEM_REF}key{MEM_REF_END}"
        lines.append(f"Also remembered ({hint}):")
        lines += [f"[{s.key}] {s.summary}" for s in uniq[:stub_limit]]
    return "\n".join(lines)


def render_delivery(d: Delivery) -> str:
    if not d.found:
        body = "not found"
    elif d.resolution == "in_context":
        body = "already in fresh memory"
    elif d.resolution == "full":
        body = str(d.value)
    else:
        body = f"{d.stub.summary} (stub, confidence {d.confidence:.1f})"
    key = d.provenance.get("key", d.ref[1]) if d.found else d.ref[1]
    return f"{MEM}[{key}] {body}{MEM_END}"


_SPANS = re.compile(re.escape(MEM_REF) + r".*?" + re.escape(MEM_REF_END) + "|"
                    + re.escape(MEM) + r".*?" + re.escape(MEM_END), re.S)
_SPECIAL = re.compile(r"<\|[^<>|]*\|>")


FILED_NOTE = "[Earlier parts of this message were filed to memory.]"  # only used with move=True
_SENT = re.compile(r"(?<=[.!?])\s+|\n\s*\n")  # a blank line is a boundary too: code, logs and lists have no full stops


def split_for_window(text: str, tail_words: int, span_words: int):
    """Split a message that will not fit Phi's linear window. Returns
    (spans to file with the watchdog, oldest first; tail kept in the prompt).
    Splits on sentence boundaries; word counts stand in for tokens."""
    words = text.split()
    if len(words) <= tail_words:
        return [], text
    sents = [x for x in _SENT.split(text.strip()) if x]
    tail, n = [], 0
    while sents and (not tail or n + len(sents[-1].split()) <= tail_words):
        n += len(sents[-1].split())
        tail.insert(0, sents.pop())
    spans, cur, cn = [], [], 0
    for sent in sents:
        w = len(sent.split())
        if cur and cn + w > span_words:
            spans.append(" ".join(cur))
            cur, cn = [], 0
        if w > span_words:  # one very long sentence: hard split by words
            ws = sent.split()
            for i in range(0, len(ws), span_words):
                spans.append(" ".join(ws[i:i + span_words]))
            continue
        cur.append(sent)
        cn += w
    if cur:
        spans.append(" ".join(cur))
    return spans, " ".join(tail)


def file_user_turn(gate: LogGate, peer: str, turn: int, text: str,
                   tail_words: int, span_words: int, move: bool = False):
    """Overflow filing. The peer's message is never changed: Phi sees it exactly
    as written, and the parts that run past her window are COPIED onto the
    memory tree as spans, where they can be whispered or recalled later.
    (move=True is the old behaviour, which cut them out of her view.)
    Returns (text Phi sees, key of this turn's own memory, the tail: the part
    that sits inside her window). Shared by the runtime and the data builder."""
    spans, tail = split_for_window(text, tail_words, span_words)
    for k, span in enumerate(spans, 1):
        gate.ingest(peer, f"turn{turn}:part{k}", span)
    qkey = f"turn{turn}:user"
    gate.ingest(peer, qkey, tail)
    shown = (f"{FILED_NOTE}\n{tail}" if spans else text) if move else text
    return shown, qkey, tail


def whisper_query(shown: str, n_words: int = 16) -> str:
    """What the watchdog listens to for whispers: the most recent words just
    said (the end of the message, where a question usually is)."""
    text = shown.replace(FILED_NOTE, " ")
    return " ".join(text.split()[-n_words:])


def window_budgets(window: int):
    """Prompt budgets that leave room for a reply inside the window."""
    return {"tail_words": max(60, window // 5), "span_words": 80,
            "entry_chars": max(200, int(window * 0.3))}


def clean_reply(text: str) -> str:
    """Drop reference requests, deliveries and leftover special tokens, so the
    reply stored back into memory does not duplicate retrieved memories."""
    return _SPECIAL.sub("", _SPANS.sub("", text)).strip()


class MemoryRuntime:
    def __init__(self, model, tokenizer, store: Optional[MemoryStore] = None,
                 deliver_chars: int = 400, max_refs: int = 4, sweep_budget: int = 8,
                 window: Optional[int] = None, cache_refs: bool = True):
        self.model, self.tok = model, tokenizer
        cfg = getattr(getattr(model, "config", None), "memtree", None) or {}
        self.window = window or cfg.get("window") or 1024  # Phi's sharp window (the tree sees the rest)
        self.budget = window_budgets(self.window)
        self.store = store or MemoryStore()
        self.gate = LogGate(self.store)
        self.dog = Watchdog(self.store)
        self.deliver_chars, self.max_refs, self.sweep_budget = deliver_chars, max_refs, sweep_budget
        self.cache_refs = cache_refs  # carry the KV cache across reference resumes (one prefill, not many)
        self.ids = add_memory_tokens(tokenizer, model)
        self.turn = 0

    @staticmethod
    def _cache_len(cache) -> int:
        """How many tokens a returned KV cache already covers (0 if unknown)."""
        if cache is None:
            return 0
        try:
            n = cache.get_seq_length()
            return int(n) if n is not None else 0
        except Exception:
            return 0  # legacy tuple cache or none: we simply won't reuse it

    def _stop_ids(self) -> List[int]:
        """Model's own end tokens (Phi-4 uses <|end|> and <|endoftext|>) plus
        the reference-close token."""
        gen = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        gen = gen if isinstance(gen, (list, tuple)) else [gen]
        ids = [*gen, self.tok.eos_token_id, self.ids[MEM_REF_END]]
        return list(dict.fromkeys(i for i in ids if i is not None))

    def _prompt_ids(self, peer: str, text: str, qkey: str = "", tail: Optional[str] = None):
        whispers = self.dog.whispers(peer, whisper_query(tail if tail is not None else text),
                                     exclude={qkey}, clip=self.budget["entry_chars"])
        aside = self.dog.aside(peer, whispers)
        memory = render_memory(self.dog, peer, entry_chars=self.budget["entry_chars"],
                               whispers=whispers, aside=aside)
        system = f"{TRUST_BLOCK}\n\n{memory}"  # trust framing, then the memory itself
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": text}]
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not hasattr(out, "shape"):  # some versions return a BatchEncoding
            out = out["input_ids"]
        self._head = self._head_len(system, out)
        return out.to(self.model.device)

    def _head_len(self, system: str, ids) -> int:
        """Token length of the system block at the head of the prompt: everything
        before the first user-turn marker. (Rendering the system message alone and
        comparing prefixes failed -- the template appends an end token -- so the
        wide-memory fix silently stayed at 4.) 4 if there is no marker."""
        try:
            row = ids[0].tolist()
            for marker in ("<|user|>",):
                tid = self.tok.convert_tokens_to_ids(marker)
                if tid is not None and tid != getattr(self.tok, "unk_token_id", None) and tid in row:
                    return max(4, row.index(tid))
        except Exception:
            pass
        return 4

    def _generate(self, ids, **kw):
        try:
            from .phi_adapter import global_prefix
        except Exception:  # torch-free environments: nothing to widen
            return self.model.generate(ids, **kw)
        with global_prefix(self.model, getattr(self, "_head", 4)):
            return self.model.generate(ids, **kw)

    def chat(self, peer: str, text: str, max_new_tokens: int = 256) -> Dict:
        import torch
        self.turn += 1
        shown, qkey, tail = file_user_turn(self.gate, peer, self.turn, text,
                                           self.budget["tail_words"], self.budget["span_words"])
        ids = self._prompt_ids(peer, shown, qkey, tail)
        start = ids.shape[1]
        refs: List[Delivery] = []
        eos = self._stop_ids()
        prompt_len = start
        cache = None          # KV cache carried across resumes, so each reaches costs
        cached_upto = 0       # one prefill of the delivery, not a reprefill of the whole prompt
        for _ in range(self.max_refs + 1):
            budget = max_new_tokens - (ids.shape[1] - prompt_len)  # total, across resumes
            if budget <= 0:
                break
            kw = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
                      return_dict_in_generate=True, eos_token_id=eos,
                      pad_token_id=self.tok.eos_token_id)
            # Reuse the cache only when it lines up exactly with the prompt's leading
            # tokens; the appended delivery is then the only thing prefilled. Any
            # mismatch or old transformers that can't continue a cache -> clean reprefill.
            try:
                if self.cache_refs and cache is not None and self._cache_len(cache) == cached_upto:
                    res = self._generate(ids, past_key_values=cache, **kw)
                else:
                    res = self._generate(ids, **kw)
            except Exception:
                cache = None  # cache incompatible with this version: fall back, never break her
                res = self._generate(ids, **kw)
            out = res.sequences
            cache = getattr(res, "past_key_values", None)
            cached_upto = out.shape[1] if cache is not None else 0
            if out[0, -1].item() != self.ids[MEM_REF_END] or len(refs) >= self.max_refs:
                ids = out
                break
            gen = self.tok.decode(out[0, start:], skip_special_tokens=False)
            key = gen.rsplit(MEM_REF, 1)[-1].split(MEM_REF_END, 1)[0].strip()
            if not key:  # an empty reference asks for nothing: stop here
                ids = out
                break
            # Phi asked explicitly, so deliver the full memory if it fits the budget;
            # the question itself was just filed: never answer it with itself
            d = self.dog.reference(peer, key, budget_chars=self.deliver_chars, expand=True,
                                   exclude={qkey})
            refs.append(d)
            dv = self.tok(render_delivery(d), add_special_tokens=False, return_tensors="pt")
            ids = torch.cat([out, dv["input_ids"].to(out.device)], dim=1)
            prompt_len += dv["input_ids"].shape[1]  # deliveries don't use up the budget
            # cache now covers `out` (cached_upto); the delivery we just appended is the
            # uncached tail, which the next generate() prefills on top of the cache.
        reply = clean_reply(self.tok.decode(ids[0, start:], skip_special_tokens=False))
        self.gate.ingest(peer, f"turn{self.turn}:assistant", reply)
        self.dog.sweep(peer, budget=self.sweep_budget)
        return {"reply": reply, "references": refs}


def reference_sft_examples(events, boundary: int = 2, deliver_chars: int = 400):
    """Teach the protocol from synthetic sessions (memtree.bench.make_session).
    Each query becomes (system memory block, question, target). Targets for
    center memories answer directly; outer ones reference first, then answer."""
    store = MemoryStore()
    gate = LogGate(store)
    dog = Watchdog(store, boundary=boundary)
    examples = []
    for ev in events:
        if ev[0] == "write":
            gate.ingest(ev[1], ev[2], ev[3])
            continue
        _, peer, key, expected = ev
        system = render_memory(dog, peer)
        depth = store.branch(peer).index.get(key)
        if depth is None:
            continue
        if depth < boundary:
            target = f"{expected}"
        else:
            d = dog.reference(peer, key, budget_chars=deliver_chars, expand=True)
            target = f"{MEM_REF}{key}{MEM_REF_END}{render_delivery(d)}{expected}"
        examples.append({"system": system, "user": f"What is {key}?", "assistant": target})
    return examples
