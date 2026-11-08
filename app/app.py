#!/usr/bin/env python3
"""AMD AI League - Match 3 (RAG over a mixed-filetype corpus).

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

--index parses every file (pdf, docx, xlsx, csv, text, code, logs), starts a
resident model server that keeps a vision-language model on the AMD GPU,
reads every image with that model, and persists the index. Each --query then
only connects to the server over a unix socket, so the model is loaded once,
not ten times.

Nothing is printed to stdout. The output file is always written, whole, even
when something fails (as a refusal), so the harness never sees a malformed file.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Extra packages ship in /app/vendor; appended so the base image's own torch/numpy win.
sys.path.append("/app/vendor")  # VENDOR

from mc3 import paths  # noqa: E402
from mc3.extract import Doc, Unit, extract, to_json  # noqa: E402

STARTUP_DEADLINE_S = 540      # leave headroom inside the 600 s startup budget
QUERY_DEADLINE_S = 26         # inside the 30 s per-question budget
SERVER = Path(__file__).resolve().parent / "mc3" / "server.py"


def _log(msg: str) -> None:
    try:
        paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(paths.INDEX_DIR / "app.log", "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg.rstrip() + "\n")
    except OSError:
        pass


def _request(req: dict, timeout: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(paths.SOCKET))
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    finally:
        s.close()


def _server_alive() -> bool:
    try:
        return bool(_request({"op": "ping"}, timeout=3).get("ok"))
    except (OSError, ValueError):
        return False


def _start_server() -> None:
    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(paths.INDEX_DIR / "server.stderr", "ab"),
        start_new_session=True,
        close_fds=True,
        cwd=str(SERVER.parent.parent),
    )


def _wait_for_server(deadline: float) -> bool:
    while time.time() < deadline:
        if _server_alive():
            return True
        time.sleep(1.0)
    return False


def index(corpus: Path) -> None:
    t0 = time.time()
    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not _server_alive():
        _start_server()                      # model loads while we parse
    docs, skipped, images = extract(corpus)
    _log(f"parsed {len(docs)} docs, {len(images)} images, skipped {len(skipped)}")

    if not _wait_for_server(t0 + STARTUP_DEADLINE_S - 60):
        _log("model server did not come up")
    for rel in images:
        if time.time() > t0 + STARTUP_DEADLINE_S:
            skipped.append({"path": rel, "reason": "out of startup time"})
            continue
        try:
            resp = _request({"op": "transcribe", "path": str(corpus / rel)}, timeout=120)
            text = (resp.get("text") or "").strip() if resp.get("ok") else ""
        except (OSError, ValueError) as e:
            text = ""
            _log(f"transcribe {rel}: {e}")
        # Even with no text the image stays findable by its file name.
        body = text or "(image; no text read)"
        docs.append(Doc(path=rel, kind="image", full_text=body,
                        units=[Unit(rel, "image", body, "image transcription")]))

    paths.INDEX_FILE.write_text(
        json.dumps({"corpus": str(corpus), "docs": to_json(docs)}, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.SKIPPED_FILE.write_text(
        json.dumps([s if isinstance(s, dict) else s.__dict__ for s in skipped], indent=1),
        encoding="utf-8",
    )
    try:
        _request({"op": "load_index"}, timeout=60)
    except (OSError, ValueError) as e:
        _log(f"load_index: {e}")
    _log(f"index done in {time.time() - t0:.1f}s")


def query(corpus: Path, text: str) -> tuple[str, list[str], float]:
    t0 = time.time()
    if not _server_alive():
        # Should not happen after --index; recover rather than refuse outright.
        _start_server()
        if not _wait_for_server(t0 + QUERY_DEADLINE_S - 6):
            return "", [], 0.0
    remaining = max(3.0, QUERY_DEADLINE_S - (time.time() - t0))
    resp = _request({"op": "answer", "query": text}, timeout=remaining)
    if not resp.get("ok"):
        _log(f"answer error: {resp.get('error')}")
        return "", [], 0.0
    return str(resp.get("answer", "")), [str(c) for c in resp.get("citations", [])], float(resp.get("confidence", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        try:
            index(args.index)
        except Exception as e:
            _log(f"index failed: {type(e).__name__}: {e}")
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    try:
        text, citations, confidence = query(args.corpus, args.query)
    except Exception as e:
        _log(f"query failed: {type(e).__name__}: {e}")
        text, citations, confidence = "", [], 0.0
    if not text:
        citations = []

    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.OUTPUT_DIR / (args.query_id + "_output.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"answer": text, "citations": list(citations), "confidence": confidence},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, out)  # atomic: the harness never reads half a file
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
