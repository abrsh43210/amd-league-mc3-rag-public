import os
from pathlib import Path

INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
INDEX_FILE = INDEX_DIR / "index.json"
SKIPPED_FILE = INDEX_DIR / "skipped.json"
SERVER_LOG = INDEX_DIR / "server.log"
SOCKET = Path(os.environ.get("MC3_SOCKET", "/tmp/mc3-model.sock"))
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
