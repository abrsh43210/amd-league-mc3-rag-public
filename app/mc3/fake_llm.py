"""Test double for the VLM, used only in local tests (MC3_FAKE_LLM=1).

It answers from the sample questions file, but only cites a file if that file
actually made it into the prompt, so it measures retrieval coverage and the
validation logic without a GPU. It is never shipped as the answering path.
"""

from __future__ import annotations

import json
import os
import re


class FakeLLM:
    def __init__(self):
        path = os.environ.get("MC3_FAKE_ORACLE", "")
        self.qs = json.load(open(path))["queries"] if path else []
        self.transcripts = json.loads(os.environ.get("MC3_FAKE_TRANSCRIPTS", "{}"))

    def chat(self, system, text, images=None, max_new_tokens=160):
        if not system:  # transcription
            name = os.path.basename((images or [""])[0])
            return self.transcripts.get(name, "")
        q = text.rsplit("QUESTION:", 1)[-1].split("\n\nJSON:")[0].strip()
        fids = dict(re.findall(r"=== \[(F\d+)\] (\S+)", text))
        rev = {v: k for k, v in fids.items()}
        for item in self.qs:
            if item["query"] == q:
                if not item["expected_answer"]:
                    return json.dumps({"found": False, "answer": "", "files": []})
                cites = [rev[c] for c in item["expected_citations"] if c in rev]
                if len(cites) != len(item["expected_citations"]):
                    return json.dumps({"found": False, "answer": "", "files": [],
                                       "missing": [c for c in item["expected_citations"] if c not in rev]})
                return json.dumps({"found": True, "answer": item["expected_answer"], "files": cites})
        return json.dumps({"found": False})
