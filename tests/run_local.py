#!/usr/bin/env python3
"""Run the real app end to end on the sample corpus with either the fake LLM
(no GPU, checks parsing + retrieval + validation) or the real model.

    MC3_FAKE_LLM=1 python3 tests/run_local.py          # anywhere
    python3 tests/run_local.py                         # on an AMD GPU box with weights
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("MC3_TEST_CORPUS", ROOT / "tests" / "mc3-corpus"))
QS = json.load(open(ROOT / "tests" / "sample-questions.json"))["queries"]

work = Path(tempfile.mkdtemp(prefix="mc3-"))
env = dict(os.environ)
env.setdefault("MC3_INDEX_DIR", str(work / "index"))
env.setdefault("MC2_OUTPUT_DIR", str(work / "output"))
env.setdefault("MC3_SOCKET", str(work / "model.sock"))
if env.get("MC3_FAKE_LLM"):
    env.setdefault("MC3_FAKE_ORACLE", str(ROOT / "tests" / "sample-questions.json"))
    env.setdefault("MC3_FAKE_TRANSCRIPTS", json.dumps({
        "backplane_pinout.png": "TQ-40 backplane pinout\nB13: GND\nTHERM_ALERT#: B14\nB15: PWR_GOOD",
        "asset_label.jpg": "ORRERY SYSTEMS\nTQ-40\nBOARD REV-C2\nS/N 0042-7781",
    }))

app = ROOT / "app" / "app.py"
t = time.time()
r = subprocess.run([sys.executable, str(app), "--index", str(CORPUS)], env=env, capture_output=True, text=True)
print(f"index: rc={r.returncode} {time.time() - t:.1f}s stdout={r.stdout!r}")

def norm(s):
    return s.lower().replace("°", "").replace(" ", "")

score = 0
for q in QS:
    qid = f"query_{q['n']:02d}"
    t = time.time()
    r = subprocess.run([sys.executable, str(app), "--corpus", str(CORPUS), "--query-id", qid, "--query", q["query"]],
                       env=env, capture_output=True, text=True)
    dt = time.time() - t
    out = json.load(open(Path(env["MC2_OUTPUT_DIR"]) / f"{qid}_output.json"))
    exp = [q["expected_answer"]] + q.get("answer_aliases", [])
    a_ok = norm(out["answer"]) in {norm(x) for x in exp} if q["expected_answer"] else out["answer"] == ""
    c_ok = set(out["citations"]) == set(q["expected_citations"])
    score += 20 * (a_ok and c_ok)
    print(f"{qid} {'PASS' if a_ok and c_ok else 'FAIL'} {dt:4.1f}s stdout={r.stdout!r} -> {out}")
print(f"score {score}/200   work dir {work}")
print("skipped:", (Path(env["MC3_INDEX_DIR"]) / "skipped.json").read_text())
subprocess.run(["pkill", "-f", "mc3/server.py"])
