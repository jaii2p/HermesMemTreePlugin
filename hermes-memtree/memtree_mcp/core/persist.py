"""Save and load the whole memory tree, so memories survive between sessions.

Everything is kept: every node with its value, strength, history, aliases and
mentions, the exact order on every level (so depth = age is preserved), the
bridges between memories, and the turn counter. The word index is rebuilt on
load. Writes are atomic (temp file + rename), so a crash mid-save never
corrupts the tree.
"""
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from .store import Branch, MemoryStore, Node

VERSION = 1


def _node_to_dict(n: Node) -> Dict[str, Any]:
    d = asdict(n)
    d["emb"] = None if n.emb is None else np.asarray(n.emb, dtype=float).tolist()
    return d


def _node_from_dict(d: Dict[str, Any]) -> Node:
    d = dict(d)
    if d.get("emb") is not None:
        d["emb"] = np.asarray(d["emb"], dtype=float)
    fields = Node.__dataclass_fields__
    return Node(**{k: v for k, v in d.items() if k in fields})


def dump(store: MemoryStore, meta: Dict[str, Any] = None) -> Dict[str, Any]:
    branches = {}
    for peer, b in store.branches.items():
        branches[peer] = {
            "clock": b.clock,
            "levels": [[_node_to_dict(n) for n in level.values()] for level in b.levels],
            "aliases": dict(b.aliases),
        }
    bridges = []
    for a, nb in store.bridges.adj.items():
        for b_, s in nb.items():
            if a < b_:
                bridges.append([a[0], a[1], b_[0], b_[1], s])
    return {"version": VERSION, "branches": branches, "bridges": bridges,
            "key_peers": {k: sorted(v) for k, v in store.key_peers.items() if v},
            "max_bridge_degree": store.bridges.max_degree, "meta": meta or {}}


def load(data: Dict[str, Any]) -> Tuple[MemoryStore, Dict[str, Any]]:
    store = MemoryStore(max_bridge_degree=data.get("max_bridge_degree", 4))
    for peer, bd in data.get("branches", {}).items():
        b = Branch(peer)
        b.clock = bd.get("clock", 0)
        b.levels = []
        for d, level in enumerate(bd.get("levels", [])):
            od = OrderedDict()
            for nd in level:
                n = _node_from_dict(nd)
                od[n.key] = n
                b.index[n.key] = d
            b.levels.append(od)
        if not b.levels:
            b.levels = [OrderedDict()]
        b.aliases = dict(bd.get("aliases", {}))
        store.branches[peer] = b
        for n in b.nodes_by_recency():
            store.index_words(peer, n.key, n.value)
    for ap, ak, bp, bk, s in data.get("bridges", []):
        a, b_ = (ap, ak), (bp, bk)
        store.bridges.adj.setdefault(a, {})[b_] = s
        store.bridges.adj.setdefault(b_, {})[a] = s
    for k, peers in data.get("key_peers", {}).items():
        store.key_peers[k] = set(peers)
    return store, data.get("meta", {})


def save_file(store: MemoryStore, path: str, meta: Dict[str, Any] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".tree-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dump(store, meta), f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_file(path: str) -> Tuple[MemoryStore, Dict[str, Any]]:
    if not os.path.exists(path):
        return MemoryStore(), {}
    with open(path, encoding="utf-8") as f:
        return load(json.load(f))
