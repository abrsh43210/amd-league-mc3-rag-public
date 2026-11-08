"""Small dependency-free BM25 over text units, tuned for part numbers and ids."""

from __future__ import annotations

import math
import re
from collections import Counter

from .extract import Doc, Unit

_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*#?")
STOP = set(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why with does do did into than then "
    "there their they any all can could should would will".split()
)
# Identifier-looking tokens: ORR-1847, E7731, 4.3.2, B14, THERM_ALERT#
ID_RE = re.compile(r"\b(?:[A-Z]{1,6}[-_]?\d[\w.-]*|\d+(?:\.\d+){1,3}|[A-Z]+_[A-Z_#]+#?)\b")


def tokens(text: str) -> list[str]:
    out = []
    for m in _WORD.finditer(text):
        w = m.group(0).lower()
        if w in STOP:
            continue
        out.append(w)
        if any(c in w for c in "-_./#"):
            out.extend(p for p in re.split(r"[-_./#]", w) if p and p not in STOP)
    return out


class Index:
    def __init__(self, docs: list[Doc]):
        self.docs = {d.path: d for d in docs}
        self.units: list[Unit] = [u for d in docs for u in d.units]
        self.toks = [tokens(u.path.replace("/", " ") + " " + u.text) for u in self.units]
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.n = max(1, len(self.units))
        self.avg = sum(len(t) for t in self.toks) / self.n if self.toks else 1.0
        self.tf = [Counter(t) for t in self.toks]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 8, k1: float = 1.4, b: float = 0.6) -> list[tuple[float, int]]:
        q = tokens(query)
        if not q:
            return []
        qc = Counter(q)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i]) or 1
            s = 0.0
            for term, qn in qc.items():
                f = tf.get(term)
                if not f:
                    continue
                s += qn * self.idf(term) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:k]

    def units_containing(self, needle: str, limit: int = 4) -> list[int]:
        n = needle.lower()
        hits = [i for i, u in enumerate(self.units) if n in u.text.lower()]
        return hits[:limit]


def identifiers(text: str) -> list[str]:
    seen, out = set(), []
    for m in ID_RE.finditer(text):
        v = m.group(0).rstrip(".-")
        if len(v) >= 3 and v not in seen:
            seen.add(v)
            out.append(v)
    return out
