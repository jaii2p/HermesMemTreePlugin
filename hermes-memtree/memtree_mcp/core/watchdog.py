"""Watchdog (the subconscious): caretaker of the memory tree.

Memories live on the tree and move freely: they fade when left alone, grow
stronger each time they are recalled or met again, drift toward the center
when hot and sink when faint, and surface on their own as whispers when
something said resembles them. They are never deleted. The watchdog does not
own them; he tends the chaos: dedup, organization, cleanup, and delivery when
Phi uses a reference as the key.


The main model (consciousness) never grabs. The center of every branch
(depths [0, boundary)) is kept in the main model's context by the watchdog via
context_view(); every reference the main model makes, fresh or deep, goes to
the watchdog. The watchdog:
  * confirms references to center memories cheaply (they are already in
    context: no delivery) and refreshes them so hot memories stay central;
  * hands the main model compact stubs for cold outer memories and full values
    for hot ones, always within a size budget;
  * attaches a confidence and provenance to every delivery;
  * promotes a node toward depth 0 only after repeated references (hysteresis);
  * sweeps cold regions in the background so unreferenced memories still get
    checked, deduplicated and repaired;
  * creates bridges from co-reference and walks them with a visited set and a
    hop budget (lateral links form cycles).
"""
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .store import _WORD, MemoryStore, Node, Ref, strengthen, vitality, words_of

BOUNDARY = 2  # depth 0..1 = center, kept in the main model's context


@dataclass
class Stub:
    key: str
    peer: str
    summary: str
    handle: Ref
    depth: int
    confidence: float = 1.0


@dataclass
class Delivery:
    found: bool
    ref: Ref
    depth: int = -1
    resolution: str = "miss"  # "in_context" | "full" | "stub" | "miss"
    value: Any = None
    stub: Optional[Stub] = None
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    hops: int = 0  # descent steps (depth + 1); misses cost a full descent
    promoted: bool = False


def _snippet(text: str, qwords: set, width: int) -> str:
    """The width-char window of text that covers the most distinct query words
    (ties: earliest), with the cluster placed early so what follows the cue --
    usually the answer -- survives. Never just 'around the first match'."""
    if len(text) <= width:
        return text
    hits = [(m.start(), m.group().lower()[:4]) for m in re.finditer(r"[A-Za-z0-9]+", text)
            if m.group().lower()[:4] in qwords]
    if not hits:
        return text[:width].rstrip() + "..."
    best, best_n, j = hits[0][0], 0, 0
    for i, (pos, _) in enumerate(hits):  # two pointers: O(len(hits))
        while hits[j][0] < pos - width // 2:
            j += 1
        n = len({w for _, w in hits[j:i + 1]})
        if n > best_n:
            best_n, best = n, hits[j][0]
    start = max(0, best - width // 6)
    end = min(len(text), start + width)
    start = max(0, end - width)
    return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()


def _phrase(text: str, n: int = 6) -> str:
    """A short, human handle for a memory: its first few words."""
    w = str(text).split()
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


class Watchdog:
    def __init__(self, store: MemoryStore, boundary: int = BOUNDARY,
                 promote_votes: int = 3, hot_refs: int = 3, stub_chars: int = 24,
                 max_hops: int = 2, allow_cross_peer: bool = True,
                 voice_mentions: int = 4, voice_bridge: float = 2.0):
        self.store = store
        self.boundary = boundary
        self.promote_votes = promote_votes
        self.hot_refs = hot_refs
        self.stub_chars = stub_chars
        self.max_hops = max_hops
        self.allow_cross_peer = allow_cross_peer
        self.voice_mentions = voice_mentions  # times seen before the voice calls it recurring
        self.voice_bridge = voice_bridge      # bridge strength before the voice calls it a connection
        self.steps = 0  # total descent steps spent, for cost accounting
        self._cycle: Dict[str, Tuple[List[str], int]] = {}
        self._norm: Dict[str, Dict[str, str]] = {}
        self._said: Dict[str, str] = {}  # last aside per peer, so it never repeats itself twice running

    # ------------------------------------------------------------ main's view
    def _stub(self, node: Node, depth: int, confidence: float = 1.0) -> Stub:
        s = str(node.value)
        summary = s if len(s) <= self.stub_chars else s[: self.stub_chars] + "..."
        return Stub(node.key, node.peer, summary, (node.peer, node.key), depth, confidence)

    def context_view(self, peer: str) -> List[Node]:
        """The center: what the watchdog keeps in the main model's context.
        At most 2**boundary - 1 nodes, newest first, full values."""
        b = self.store.branches.get(peer)
        if b is None:
            return []
        return [n for d in range(min(self.boundary, len(b.levels))) for n in b.depth_nodes(d)]

    def boundary_stubs(self, peer: str) -> List[Stub]:
        """Stubs for the first outer depth: at most 2**boundary of them."""
        b = self.store.branches.get(peer)
        if b is None:
            return []
        return [self._stub(n, self.boundary) for n in b.depth_nodes(self.boundary)]

    def proactive(self, peer: str, limit: int = 4) -> List[Stub]:
        """Outer memories bridged to what is in context right now: surfaced
        without being asked, so the main model learns they exist."""
        b = self.store.branches.get(peer)
        if b is None:
            return []
        center = {(peer, n.key) for n in self.context_view(peer)}
        out, seen = [], set()
        for ref in sorted(center):
            for nb, _, _ in self.related(*ref, max_hops=1):
                if nb in center or nb in seen:
                    continue
                seen.add(nb)
                node, depth = self.store.lookup(*nb)
                if depth >= self.boundary or nb[0] != peer:
                    out.append(self._stub(node, depth, confidence=0.5))
                if len(out) >= limit:
                    return out
        return out

    # ----------------------------------------------------------------- search
    def search(self, peer: str, query: str, limit: int = 3, exclude=(),
               max_df: int = 512) -> List[Tuple[Node, float]]:
        """Find memories by content when no exact key matches. Uses the word
        index built at ingest, so cost scales with how many memories share the
        query's words, not with the size of memory. Rare words count more (IDF);
        ties go to the more recent memory."""
        b = self.store.branches.get(peer)
        idx = self.store.words.get(peer)
        if b is None or not idx:
            return []
        q = words_of(query)
        n = max(len(b), 1)
        scores: Dict[str, float] = {}
        # The scan is bounded by max_df in every case. A word shared by more
        # memories than that is read only as far as max_df, newest first, so a
        # query made entirely of common words cannot walk the whole posting (it
        # used to, which was O(tree) per query); the word still counts in full
        # toward its own IDF, so scoring is unchanged wherever candidates fit.
        usable = [w for w in q if idx.get(w) and len(idx[w]) <= max_df]
        bounded = False
        if not usable:
            common = [w for w in q if idx.get(w)]
            if common:
                usable = [min(common, key=lambda w: (len(idx[w]), w))]
                bounded = True
        for w in usable:
            posting = idx[w]
            idf = math.log(1.0 + n / len(posting))
            keys = list(posting)
            for k in (keys[-max_df:] if bounded else keys):
                if k not in b.index:  # merged or removed since indexed
                    posting.pop(k, None)
                    continue
                if k in exclude:
                    continue
                node = b.get(k)[0]
                if w not in words_of(k) | words_of(node.value):  # value changed since
                    continue
                scores[k] = scores.get(k, 0.0) + idf
        # The index cuts words to 4 letters so allergy/allergic match, which also makes
        # Marisol and Marisa the same word. Among memories the stems already matched, a
        # whole-word match breaks the tie toward what was actually said.
        full = {w for w in _WORD.findall(str(query).lower()) if len(w) > 4 and w[:4] in q}
        for k in scores:
            node = b.get(k)[0]
            if full:
                have = set(_WORD.findall(str(node.value).lower()))
                scores[k] += sum(1.0 * math.log(1.0 + n / max(len(idx.get(w[:4], ())), 1))
                                 for w in full if w in have)
            if str(node.value).rstrip().endswith("?"):  # a remembered question rarely holds the answer
                scores[k] *= 0.5
            # living memories: strong ones come forward, faint ones still answer a strong match
            scores[k] *= 0.5 + 0.75 * min(vitality(node, b.clock), 2.0)
        ranked = sorted(scores, key=lambda k: (-scores[k], -b.get(k)[0].touched))
        return [(b.get(k)[0], scores[k]) for k in ranked[:limit]]

    # ---------------------------------------------------------------- whispers
    def whispers(self, peer: str, text: str, limit: int = 3, exclude=(), clip: Optional[int] = None,
                 width: int = 220, min_score: Optional[float] = None,
                 rel: float = 0.35) -> List[Stub]:
        """Unasked-for associations: memories outside Phi's fresh memory that
        share distinctive words with what was just said, wherever they sit in
        the tree. Each is a snippet around the matching words, so the relevant
        part shows even when the memory is long. Lower confidence than a
        reference: a whisper can be only loosely related."""
        # Center memories are already in front of her -- unless the prompt clips them
        # (`clip` chars): then the part that matters may be cut off, so they stay
        # eligible and the query-centred snippet delivers what the clip dropped.
        center = {n.key for n in self.context_view(peer)
                  if clip is None or len(str(n.value)) <= clip}
        found = self.search(peer, text, limit=limit + len(center) + len(exclude),
                            exclude=set(exclude) | center)
        q = words_of(text)
        out = []
        b = self.store.branches.get(peer)
        if b is None:
            return out
        # The gate scales with the corpus instead of a fixed 1.0: IDF is relative to
        # tree size, so an absolute bar starves common words in a small tree and
        # lets anything through in a huge one. A match must clear a floor of a tenth
        # of log(1+n) (one word seen in ~1 in 10 memories) and be within `rel` of
        # this turn's best, so the best candidates always surface and noise doesn't.
        floor = min_score if min_score is not None else 0.1 * math.log(1.0 + max(len(b), 1))
        top = found[0][1] if found else 0.0
        keep = [(n, sc) for n, sc in found if sc >= floor and sc >= rel * top]
        # A look-alike is worse than silence: when the question names something exactly
        # (Marisol) and some candidates contain that exact word, candidates that only
        # share its 4-letter stem (Marisa) are dropped, so she is never handed two
        # near-identical facts to choose between.
        raw = {n.key: set(_WORD.findall(str(n.value).lower())) for n, _ in keep}
        for w in {w for w in _WORD.findall(str(text).lower()) if len(w) > 4 and w[:4] in q}:
            has = [n.key for n, _ in keep if w in raw[n.key]]
            alike = {n.key for n, _ in keep if w not in raw[n.key] and any(t[:4] == w[:4] for t in raw[n.key])}
            if has and alike:  # someone has her exact word; the stem-only look-alikes go
                keep = [(n, sc) for n, sc in keep if n.key not in alike]
        for node, score in keep[:limit]:
            depth = b.index[node.key]
            strengthen(node, b.clock, 0.25)  # surfacing a memory keeps it a little more alive
            out.append(Stub(node.key, node.peer, _snippet(str(node.value), q, width),
                            (peer, node.key), depth, confidence=0.5))
        return out

    # ------------------------------------------------------------- inner voice
    def aside(self, peer: str, whispers=()) -> Optional[str]:
        """A quiet thought from the subconscious, in the watchdog's own voice.

        Never invented: every line is read straight from the tree's state, so it
        is always honestly true. It only looks at what is already surfacing this
        turn (the center plus the whispers just computed), so it stays O(what is
        in hand), never a scan. Most turns it stays quiet and returns None; it
        speaks only when the tree itself shows something worth a nudge:

          * two things in play that keep arriving together (a bridge has grown);
          * something she keeps coming back to (met many times);
          * a memory whose value she has since changed (it has history).

        This is how trust gets built instead of demanded: the watchdog is a
        companion who notices, not a vault that answers.
        """
        b = self.store.branches.get(peer)
        if b is None:
            return None
        cand = list(self.context_view(peer))
        for w in whispers:
            hit = b.get(w.key)
            if hit is not None:
                cand.append(hit[0])
        if not cand:
            return None
        in_play = {(peer, n.key): n for n in cand}
        # 1) a connection forming between two things in play right now
        for (_, ka), na in in_play.items():
            for nb, s in self.store.bridges.neighbors((peer, ka)).items():
                if nb in in_play and s >= self.voice_bridge and nb[1] > ka:
                    other = in_play[nb]
                    return self._voice(peer, f"{_phrase(na.value)} and {_phrase(other.value)} "
                                             f"keep arriving together.")
        # 2) something she keeps returning to
        hot = max(cand, key=lambda n: len(n.mentions))
        if len(hot.mentions) >= self.voice_mentions:
            return self._voice(peer, f"you keep coming back to {_phrase(hot.value)} — it seems to matter.")
        # 3) a view that has changed
        for n in cand:
            if n.history:
                return self._voice(peer, f"you see {_phrase(n.value)} differently than you used to.")
        return None

    def _voice(self, peer: str, line: str) -> Optional[str]:
        """Don't say the same quiet thought twice in a row; let it breathe."""
        if self._said.get(peer) == line:
            return None
        self._said[peer] = line
        return line

    # -------------------------------------------------------------- reference
    def reference(self, peer: str, key: str, budget_chars: Optional[int] = None,
                  expand: bool = False, co_refs: Tuple[Ref, ...] = (),
                  exclude=()) -> Delivery:
        ref = (peer, key)
        branch = self.store.branches.get(peer)
        if branch is None:
            return Delivery(False, ref)
        canon = branch.aliases.get(key, key)
        confidence = 1.0 if canon == key else 0.9
        matched_by = "key" if canon == key else "alias"
        hit = branch.get(canon)
        if hit is None:
            found = self.search(peer, key, limit=1, exclude=exclude)
            if not found:
                hops = len(branch.levels)
                self.steps += hops
                return Delivery(False, ref, hops=hops)
            canon, confidence, matched_by = found[0][0].key, 0.6, "search"
            hit = branch.get(canon)
        node, depth = hit
        hops = depth + 1
        self.steps += hops
        node.refs += 1
        strengthen(node, branch.clock)  # recalled: more alive
        for other in co_refs:
            self._co_reference((peer, canon), other)
        if depth < self.boundary:
            # Already in the main model's context: confirm, refresh, deliver nothing.
            if depth > 0:
                branch.touch(canon)
            return Delivery(True, ref, depth, "in_context", None, None, confidence,
                            {"mentions": len(node.mentions), "superseded": len(node.history),
                             "aliases": list(node.aliases), "visibility": node.visibility,
                             "key": canon, "matched_by": matched_by},
                            hops, False)
        node.votes += 1
        s = str(node.value)
        fits = budget_chars is None or len(s) <= budget_chars
        hot = node.refs >= self.hot_refs or expand
        if hot and fits:
            resolution, value, stub = "full", node.value, None
        else:
            resolution, value = "stub", None
            stub = self._stub(node, depth, confidence)
            if budget_chars is not None:
                stub.summary = stub.summary[:budget_chars]
        promoted = False
        if node.votes >= self.promote_votes:
            branch.touch(canon)  # hysteresis: only after repeated agreement
            node.votes = 0
            promoted = True
        return Delivery(True, ref, depth, resolution, value, stub, confidence,
                        {"mentions": len(node.mentions), "superseded": len(node.history),
                         "aliases": list(node.aliases), "visibility": node.visibility,
                         "key": canon, "matched_by": matched_by},
                        hops, promoted)

    def _co_reference(self, a: Ref, b: Ref) -> None:
        if not self._visible(a, b):
            return
        self.store.bridges.add(a, b, 1.0)

    def _visible(self, a: Ref, b: Ref) -> bool:
        if a[0] == b[0] or self.allow_cross_peer:
            return True
        nb = self.store.lookup(*b)
        return nb is not None and nb[0].visibility == "shared"

    # ------------------------------------------------------------ bridge walk
    def related(self, peer: str, key: str, max_hops: Optional[int] = None,
                limit: int = 16) -> List[Tuple[Ref, int, float]]:
        """BFS over bridges. Visited set + hop budget make cycles harmless.
        Returns [(ref, hops, edge_strength)] for live, visible nodes."""
        max_hops = self.max_hops if max_hops is None else max_hops
        start = (peer, key)
        seen = {start}
        frontier = [start]
        out: List[Tuple[Ref, int, float]] = []
        for hop in range(1, max_hops + 1):
            nxt = []
            for cur in frontier:
                for nb, s in sorted(self.store.bridges.neighbors(cur).items(),
                                    key=lambda kv: -kv[1]):
                    if nb in seen:
                        continue
                    seen.add(nb)
                    if self.store.lookup(*nb) is None or not self._visible(start, nb):
                        continue
                    out.append((nb, hop, s))
                    nxt.append(nb)
                    if len(out) >= limit:
                        return out
            frontier = nxt
        return out

    # ------------------------------------------------------------------ sweep
    def sweep(self, peer: str, budget: int = 8) -> Dict[str, int]:
        """Check up to `budget` deep nodes. Cycles through every deep node, so
        cold regions are eventually visited. Amortized O(1) per node checked."""
        stats = {"checked": 0, "merged": 0, "repaired": 0, "decayed": 0}
        branch = self.store.branches.get(peer)
        if branch is None:
            return stats
        keys, pos = self._cycle.get(peer, ([], 0))
        norm = self._norm.setdefault(peer, {})
        wraps = 0
        while stats["checked"] < budget:
            if pos >= len(keys):
                wraps += 1
                if wraps > 1:  # one full pass already done this call
                    break
                keys, pos = list(branch.index), 0
                if not keys:
                    break
            key = keys[pos]
            pos += 1
            depth = branch.index.get(key)
            if depth is None or depth < self.boundary:
                continue  # gone, or main's zone
            stats["checked"] += 1
            node = branch.levels[depth].get(key)
            if node is None:  # index points at the wrong depth: repair
                for d, lvl in enumerate(branch.levels):
                    if key in lvl:
                        branch.index[key] = d
                        stats["repaired"] += 1
                        break
                continue
            if node.votes > 0:
                node.votes -= 1
                stats["decayed"] += 1
            nk = normalize_key(key)
            canon_key = norm.get(nk)
            if canon_key is None or canon_key not in branch.index:
                norm[nk] = key
            elif canon_key != key:
                self._merge(branch, canon_key, key)
                stats["merged"] += 1
        self._cycle[peer] = (keys, pos)
        return stats

    def _merge(self, branch, canon_key: str, dup_key: str) -> None:
        canon = branch.get(canon_key)[0]
        dup = branch.get(dup_key)[0]
        canon.mentions.extend(dup.mentions)
        canon.aliases.extend([dup_key] + dup.aliases)
        if dup.value != canon.value:
            if dup.touched > canon.touched:
                canon.history.append(canon.value)
                canon.value = dup.value
            else:
                canon.history.append(dup.value)
        canon.history.extend(dup.history)
        canon.refs += dup.refs
        branch.aliases[dup_key] = canon_key
        for a in dup.aliases:
            branch.aliases[a] = canon_key
        branch.remove(dup_key)
        self.store.bridges.redirect((branch.peer, dup_key), (branch.peer, canon_key))
        self.store.key_peers[dup_key].discard(branch.peer)
