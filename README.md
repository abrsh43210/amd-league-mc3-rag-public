# AMD AI League – Match 3: RAG over a mixed-filetype corpus

A Docker submission for the AMD AI League (lablab.ai) Mini-Challenge 3. It answers questions about a corpus of
PDFs, Word files, spreadsheets, CSVs, logs, source code and images, and returns the exact source files.

## How it works

1. **`--index`** (once, before the graded clock)
   - starts a resident model server that loads a vision-language model (Qwen3-VL-8B-Instruct, bf16) onto the AMD GPU via ROCm
   - while it loads, parses every file defensively: one unreadable, encrypted or unknown file never stops the walk
   - reads every image with the VLM (labels, pinouts, tables) so images become searchable
   - persists the index to `/app/index`
2. **`--query`** (once per question, a new process each time)
   - connects to the resident server over a unix socket, so the model is never reloaded
   - BM25 retrieval tuned for identifiers (part numbers, ticket ids, versions), plus one automatic hop through
     identifiers found in the top hits (log → ticket id → bug database)
   - the VLM answers from the shown files only, as strict JSON; images are attached directly
   - the answer is then **verified**: it must literally appear in a cited file (or the file is an image),
     withdrawn/superseded documents are dropped when a current one holds the answer, and anything unsupported
     becomes a refusal (empty answer, no citations)
   - writes `/app/output/<query-id>_output.json` atomically; nothing is printed to stdout

## Test locally

```bash
MC3_FAKE_LLM=1 python3 tests/run_local.py    # no GPU: parsing + retrieval + validation
python3 tests/run_local.py                   # on an AMD GPU with the weights at /models/vlm
python3 tests/selfcheck.py <image> tests/mc3-corpus
```

## Build

GitHub Actions (`.github/workflows/build-image.yml`) builds the image on the mandated ROCm base and pushes it
to GHCR. Weights are baked in at build time; the container needs no network at evaluation.

Built by Abe (Abrsh23) for the AMD AI League. MIT licensed.
