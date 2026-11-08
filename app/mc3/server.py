"""Resident model server. Started once by --index, it keeps the model on the GPU
and answers each question over a unix socket, so the per-question process only
has to connect, ask and write a file.

Protocol: one JSON object per line in, one JSON object per line out.
  {"op": "ping"}                      -> {"ok": true}
  {"op": "transcribe", "path": "..."} -> {"ok": true, "text": "..."}
  {"op": "load_index"}                -> {"ok": true, "units": N}
  {"op": "answer", "query": "..."}    -> {"ok": true, "answer": "...", "citations": [...], "confidence": x}
"""

from __future__ import annotations

import json
import os
import socket
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.append("/app/vendor")  # VENDOR: extra packages, base image wins

from mc3 import paths  # noqa: E402
from mc3.extract import from_json  # noqa: E402
from mc3.pipeline import TRANSCRIBE_PROMPT, answer  # noqa: E402
from mc3.search import Index  # noqa: E402


def log(msg: str) -> None:
    with open(paths.SERVER_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")


class State:
    def __init__(self, llm):
        self.llm = llm
        self.index: Index | None = None
        self.corpus = Path("/app/corpus")

    def load_index(self) -> int:
        data = json.loads(paths.INDEX_FILE.read_text(encoding="utf-8"))
        self.corpus = Path(data["corpus"])
        self.index = Index(from_json(data["docs"]))
        return len(self.index.units)

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True}
        if op == "transcribe":
            text = self.llm.chat("", TRANSCRIBE_PROMPT, [req["path"]], max_new_tokens=700)
            return {"ok": True, "text": text}
        if op == "load_index":
            return {"ok": True, "units": self.load_index()}
        if op == "answer":
            if self.index is None:
                self.load_index()
            a, c, conf = answer(self.index, self.llm, self.corpus, req["query"])
            return {"ok": True, "answer": a, "citations": c, "confidence": conf}
        return {"ok": False, "error": f"unknown op {op!r}"}


def serve(llm) -> None:
    sock_path = str(paths.SOCKET)
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    state = State(llm)
    if paths.INDEX_FILE.exists():
        try:
            state.load_index()
        except Exception:
            log(traceback.format_exc())
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    log("ready")
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf.decode("utf-8") or "{}")
                try:
                    resp = state.handle(req)
                except Exception as e:
                    log(traceback.format_exc())
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except Exception:
                log(traceback.format_exc())


def main() -> None:
    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MC3_FAKE_LLM"):
        from mc3.fake_llm import FakeLLM
        llm = FakeLLM()
    else:
        from mc3.model import VLM
        llm = VLM()
    serve(llm)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
