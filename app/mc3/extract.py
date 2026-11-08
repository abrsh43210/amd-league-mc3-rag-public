"""Turn a mixed-type corpus into searchable text units.

Every file is handled inside its own try/except: one unreadable, encrypted or
unknown file must never stop the walk. Nothing here prints to stdout.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TEXT_EXT = {
    ".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml", ".ini", ".cfg",
    ".conf", ".toml", ".xml", ".html", ".htm", ".js", ".ts", ".java", ".c",
    ".h", ".cpp", ".go", ".rs", ".sh", ".sql", ".rst", ".tex", ".csv_", ".env",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_EXT = {".csv", ".tsv"}

SUPERSEDED_RE = re.compile(
    r"\b(withdrawn|superseded|obsolete|deprecated|do not use|do not design to)\b", re.I
)


@dataclass
class Unit:
    path: str          # corpus-relative, forward slashes
    kind: str          # pdf, docx, xlsx, csv, text, image
    text: str          # what we search and show to the model
    loc: str = ""      # page / sheet / row range, for the prompt only


@dataclass
class Doc:
    path: str
    kind: str
    flags: list[str] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    full_text: str = ""


@dataclass
class Skipped:
    path: str
    reason: str


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def walk(root: Path) -> list[Path]:
    """Every regular file under root, sorted, never raising."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                if p.is_file() and not p.is_symlink():
                    out.append(p)
            except OSError:
                continue
    return out


def _chunk_lines(text: str, max_chars: int = 1800, overlap_lines: int = 3) -> list[tuple[str, str]]:
    lines = text.splitlines()
    if len(text) <= max_chars:
        return [(text, "")]
    chunks, cur, start, size = [], [], 0, 0
    for i, line in enumerate(lines):
        cur.append(line)
        size += len(line) + 1
        if size >= max_chars:
            chunks.append(("\n".join(cur), f"lines {start + 1}-{i + 1}"))
            keep = cur[-overlap_lines:] if overlap_lines else []
            start = i + 1 - len(keep)
            cur, size = list(keep), sum(len(x) + 1 for x in keep)
    if cur and (not chunks or len(cur) > overlap_lines):
        chunks.append(("\n".join(cur), f"lines {start + 1}-{len(lines)}"))
    return chunks


def _decode(raw: bytes) -> str | None:
    """Text if the bytes look like text, else None (unknown binary)."""
    if not raw:
        return ""
    if b"\x00" in raw[:4096]:
        return None
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            s = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        sample = s[:4096]
        printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in sample)
        if sample and printable / len(sample) < 0.9:
            return None
        return s
    return None


def _table_rows(rows: list[list[str]], label: str) -> list[tuple[str, str]]:
    """Each row rendered with its header, so a row is self-describing."""
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header = rows[0]
    units = []
    if len(rows) == 1:
        return [(" | ".join(header), label)]
    for i, r in enumerate(rows[1:], start=2):
        pairs = []
        for j, v in enumerate(r):
            if not v:
                continue
            h = header[j] if j < len(header) and header[j] else f"col{j + 1}"
            pairs.append(f"{h}: {v}")
        units.append(("; ".join(pairs), f"{label} row {i}".strip()))
    return units


def _pdf(p: Path) -> tuple[list[tuple[str, str]], str | None]:
    import pypdf

    reader = pypdf.PdfReader(str(p))
    if reader.is_encrypted:
        # Never try to open it, not even with an empty password: anything
        # inside is out of bounds by the rules.
        return [], "encrypted pdf"
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            out.append((t, f"page {i}"))
    if not out:
        return [], "pdf without extractable text"
    return out, None


def _docx(p: Path) -> list[tuple[str, str]]:
    import docx

    d = docx.Document(str(p))
    parts = [para.text for para in d.paragraphs if para.text.strip()]
    for ti, table in enumerate(d.tables, start=1):
        rows = [[c.text for c in r.cells] for r in table.rows]
        for text, loc in _table_rows(rows, f"table {ti}"):
            parts.append(text)
    return [("\n".join(parts), "")] if parts else []


def _xlsx(p: Path) -> list[tuple[str, str]]:
    import openpyxl

    wb = openpyxl.load_workbook(str(p), data_only=True, read_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        out.extend(_table_rows(rows, f"sheet '{ws.title}'"))
    return out


def _csv(p: Path, raw: bytes) -> list[tuple[str, str]]:
    text = _decode(raw)
    if text is None:
        return []
    delim = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return _table_rows(rows, "")


def extract(root: Path) -> tuple[list[Doc], list[Skipped], list[str]]:
    """Returns (docs, skipped, image_paths). Images get text later, from the VLM."""
    docs: list[Doc] = []
    skipped: list[Skipped] = []
    images: list[str] = []
    for p in walk(root):
        rel = _rel(root, p)
        ext = p.suffix.lower()
        try:
            if ext in IMAGE_EXT:
                # Prove it is readable now; the model reads it later.
                with open(p, "rb") as fh:
                    fh.read(16)
                images.append(rel)
                continue
            raw = None
            if ext not in {".pdf", ".docx", ".xlsx", ".xlsm"}:
                with open(p, "rb") as fh:
                    raw = fh.read(20_000_000)
            if ext == ".pdf":
                pieces, why = _pdf(p)
                kind = "pdf"
                if why:
                    skipped.append(Skipped(rel, why))
                    continue
            elif ext == ".docx":
                pieces, kind = _docx(p), "docx"
            elif ext in {".xlsx", ".xlsm"}:
                pieces, kind = _xlsx(p), "xlsx"
            elif ext in TABLE_EXT:
                pieces, kind = _csv(p, raw), "csv"
            else:
                text = _decode(raw)
                if text is None:
                    skipped.append(Skipped(rel, "binary / unknown format"))
                    continue
                pieces, kind = _chunk_lines(text), "text"
            pieces = [(t, loc) for t, loc in pieces if t and t.strip()]
            if not pieces:
                skipped.append(Skipped(rel, "no text"))
                continue
            doc = make_doc(rel, kind, pieces)
            docs.append(doc)
        except Exception as e:  # unreadable, corrupt, permission denied...
            skipped.append(Skipped(rel, f"{type(e).__name__}: {e}"[:200]))
    return docs, skipped, images


def make_doc(rel: str, kind: str, pieces: list[tuple[str, str]]) -> Doc:
    full = "\n".join(t for t, _ in pieces)
    flags = []
    head = rel + "\n" + full[:600]
    if SUPERSEDED_RE.search(head):
        flags.append("marked WITHDRAWN/SUPERSEDED - prefer the current revision")
    doc = Doc(path=rel, kind=kind, flags=flags, full_text=full)
    # Small documents stay whole: whole-file context beats fragments.
    if kind in ("pdf", "docx", "text") and len(full) <= 4000:
        doc.units = [Unit(rel, kind, full, "")]
    elif kind in ("pdf", "docx"):
        for t, loc in pieces:
            for sub, subloc in _chunk_lines(t):
                doc.units.append(Unit(rel, kind, sub, " ".join(x for x in (loc, subloc) if x)))
    else:
        doc.units = [Unit(rel, kind, t, loc) for t, loc in pieces]
    return doc


def to_json(docs: list[Doc]) -> list[dict]:
    return [asdict(d) for d in docs]


def from_json(items: list[dict]) -> list[Doc]:
    out = []
    for d in items:
        units = [Unit(**u) for u in d.pop("units")]
        out.append(Doc(units=units, **d))
    return out
