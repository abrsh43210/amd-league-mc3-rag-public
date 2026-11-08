"""The vision-language model, placed explicitly on the AMD GPU (ROCm exposes it
through torch.cuda). Loaded once, by the resident server only."""

from __future__ import annotations

import os

MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/vlm")
MAX_IMAGE_SIDE = 1600


class VLM:
    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible to torch (ROCm)")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def _load_image(self, path: str):
        from PIL import Image

        im = Image.open(path)
        im = im.convert("RGB")
        w, h = im.size
        scale = MAX_IMAGE_SIDE / max(w, h)
        if scale < 1:
            im = im.resize((int(w * scale), int(h * scale)))
        return im

    def chat(self, system: str, text: str, images: list[str] | None = None, max_new_tokens: int = 160) -> str:
        content = []
        for p in images or []:
            try:
                content.append({"type": "image", "image": self._load_image(p)})
            except Exception:
                continue
        content.append({"type": "text", "text": text})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
