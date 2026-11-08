# Mandated base image: checked by layer identity at evaluation. Do not change.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# Weights ship INSIDE the image: there is no network at evaluation time.
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python3 -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/vlm', token=os.environ.get('HF_TOKEN') or None, \
allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py'])" \
 && rm -rf /root/.cache

ENV MC3_MODEL_DIR=/models/vlm \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index && chmod -R a+rwX /app/index /app/output

# No service, no port, no entrypoint script: the harness execs into this.
CMD ["sleep", "infinity"]
