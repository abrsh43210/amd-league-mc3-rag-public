"""Question -> (answer, citations, confidence). Model-agnostic: takes any `llm`
with .chat(system, text, images) -> str, so it can be tested without a GPU."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .extract import Doc
from .search import Index, identifiers

MAX_CONTEXT_CHARS = 14_000
MAX_IMAGES = 2

SYSTEM = """You answer questions about a company's internal document corpus using ONLY the files shown to you.
The products and companies are fictional: never use outside knowledge, never guess.

Rules:
1. "answer" is the bare value only: a number, part number, code, version, date, quarter or name. No sentences. Keep qualifiers that are part of the value (write "Q3 FY27", not "Q3"). Units are optional.
2. If a file is marked WITHDRAWN or SUPERSEDED, do not use it when a current revision exists; use the current one.
3. Some questions need two files chained together (for example: a log names a ticket id, and a ticket database gives the fix version for that ticket). Follow such links through the files.
4. "files" lists ONLY the files without which the answer would be impossible. Usually one file; two when you had to chain. Never list a file just because it is related.
5. A comment that mentions an old or previous value is not the answer; the current definition is.
6. If the shown files do not contain the answer, return found=false with an empty answer and no files. Do not answer from memory.

Reply with one JSON object and nothing else:
{"found": true|false, "answer": "<value>", "files": ["F1", ...], "evidence": "<short verbatim quote>"}"""

TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written, line by line, including "
    "labels, table cells, part numbers, pin names and numbers. If it is a diagram, pinout, "
    "chart or table, then also list each label together with the value or position it maps "
    "to, one per line, as 'label: value'. Plain text only."
)


def _norm(s: str) -> str:
    s = s.lower().replace("°", "").replace("degrees", "")
    return re.sub(r"[\s,]", "", s)


def _contains(haystack: str, answer: str) -> bool:
    a = _norm(answer)
    if not a:
        return False
    h = _norm(haystack)
    if a in h:
        return True
    # "94 C" style answers: accept the numeric core.
    core = re.sub(r"(c|f|s|sec|seconds|w|v|gib|gb|mb|mhz|ghz|%)$", "", a)
    return bool(core) and len(core) >= 1 and core in h


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            pass
    return {}


def gather(index: Index, query: str) -> list[int]:
    """Unit ids for the prompt: BM25 hits, then one hop through identifiers."""
    hits = [i for _, i in index.search(query, k=8)]
    chosen = list(dict.fromkeys(hits))
    qlow = query.lower()
    hop_ids = []
    for i in hits[:4]:
        for ident in identifiers(index.units[i].text):
            if ident.lower() not in qlow and ident not in hop_ids:
                hop_ids.append(ident)
    for ident in hop_ids[:6]:
        for j in index.units_containing(ident, limit=3):
            if j not in chosen:
                chosen.append(j)
    return chosen


def build_context(index: Index, unit_ids: list[int]) -> tuple[str, dict[str, str], list[str]]:
    """Group units by file, assign F-ids, cap the size. Returns (text, fid->path, images)."""
    order: list[str] = []
    per_file: dict[str, list[int]] = {}
    for i in unit_ids:
        p = index.units[i].path
        if p not in per_file:
            per_file[p] = []
            order.append(p)
        per_file[p].append(i)

    blocks, fids, images, used = [], {}, [], 0
    for n, path in enumerate(order, start=1):
        doc: Doc = index.docs[path]
        fid = f"F{n}"
        header = f"=== [{fid}] {path} ({doc.kind})"
        if doc.flags:
            header += " [" + "; ".join(doc.flags) + "]"
        body_parts = []
        for i in per_file[path]:
            u = index.units[i]
            body_parts.append((f"({u.loc}) " if u.loc else "") + u.text)
        body = "\n".join(body_parts)
        room = MAX_CONTEXT_CHARS - used
        if room < 300:
            break
        body = body[:room]
        used += len(body) + len(header)
        blocks.append(header + "\n" + body)
        fids[fid] = path
        if doc.kind == "image" and len(images) < MAX_IMAGES:
            images.append(path)
    return "\n\n".join(blocks), fids, images


def answer(index: Index, llm, corpus: Path, query: str) -> tuple[str, list[str], float]:
    unit_ids = gather(index, query)
    if not unit_ids:
        return "", [], 0.0
    context, fids, images = build_context(index, unit_ids)
    img_note = ""
    if images:
        tags = [f for f, p in fids.items() if p in images]
        img_note = f"\nThe images for {', '.join(tags)} are attached, in that order; read them directly.\n"
    user = f"FILES:\n{context}\n{img_note}\nQUESTION: {query}\n\nJSON:"
    raw = llm.chat(SYSTEM, user, [str(corpus / p) for p in images])
    out = _extract_json(raw)
    if not out or not out.get("found"):
        return "", [], 0.0
    ans = str(out.get("answer", "")).strip().strip('"').strip()
    if not ans or ans.lower() in {"unknown", "none", "n/a", "not found", "i don't know"}:
        return "", [], 0.0

    files = []
    for f in out.get("files") or []:
        f = str(f).strip().strip("[]")
        path = fids.get(f) or (f if f in index.docs else None)
        if path and path not in files:
            files.append(path)

    def text_of(p: str) -> str:
        return index.docs[p].full_text

    # The answer must be visible in what we cite (images: trust the reading).
    supported = [p for p in files if index.docs[p].kind == "image" or _contains(text_of(p), ans)]
    if not supported:
        # Model cited the wrong file: look for the answer in the shown files.
        holders = [p for p in fids.values() if _contains(text_of(p), ans)]
        current = [p for p in holders if not index.docs[p].flags]
        holders = current or holders
        if len(holders) == 1:
            files = holders + [p for p in files if p != holders[0]][:1] if len(files) > 1 else holders
        else:
            return "", [], 0.0

    # Drop a withdrawn file when a current cited file already holds the answer.
    current_holders = [p for p in files if not index.docs[p].flags and
                       (index.docs[p].kind == "image" or _contains(text_of(p), ans))]
    if current_holders:
        files = [p for p in files if not index.docs[p].flags]

    if not files:
        return "", [], 0.0
    conf = 0.9 if len(files) == 1 else 0.75
    return ans, files[:3], conf
