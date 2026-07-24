"""
vlm/backends.py — VLM backend implementations.

VLMBackend   : abstract base
APIBackend   : OpenAI-compatible cloud API — model hot-swappable at runtime
LocalBackend : local fine-tuned LLaVA + LoRA
RemoteBackend: remote GPU server
"""

import io
import base64
import threading
from abc import ABC, abstractmethod

from PIL import Image as PILImage
from logger import log


# ── Available API models shown in the dashboard dropdown ────────
API_MODELS = {
    # ── OpenAI ───────────────────────────────────────────────
    "gpt-5-nano":                 "GPT-5 Nano          — FREE, fast vision",
    "gpt-5-mini":                 "GPT-5 Mini          — OpenAI latest small",
    "gpt-5":                      "GPT-5               — OpenAI flagship",
    "gpt-4.1":                    "GPT-4.1             — Latest GPT-4",
    "gpt-4o":                     "GPT-4o              — Vision flagship",
    "gpt-4o-mini":                "GPT-4o Mini         — Fast & cheap",
    # ── Claude (Anthropic) ───────────────────────────────────
    "claude-sonnet-4-6":          "Claude Sonnet 4.6   — Balanced (default)",
    "claude-opus-4-6":            "Claude Opus 4.6     — Most capable",
    "claude-haiku-4-5-20251001":  "Claude Haiku 4.5    — Fastest/cheapest",
    "claude-4.7-opus":            "Claude 4.7 Opus     — Latest Opus",
    # ── Gemini (Google) ──────────────────────────────────────
    "gemini-2.5-pro":             "Gemini 2.5 Pro      — Google flagship",
    "gemini-2.5-flash":           "Gemini 2.5 Flash    — Fast Gemini",
    "gemini-3-flash":             "Gemini 3 Flash      — Latest Flash",
    "gemini-3.1-pro":             "Gemini 3.1 Pro      — Latest Pro",
    # ── Meta ─────────────────────────────────────────────────
    "llama-4-maverick":           "Llama 4 Maverick    — Meta open model",
    # ── X-AI ─────────────────────────────────────────────────
    "grok-4":                     "Grok 4              — xAI flagship",
    "grok-4.1-fast":              "Grok 4.1 Fast       — Fast Grok",
    # ── Korean models (via MindLogic gateway) ────────────────
    "sait-3-pro":                 "SAIT 3 Pro          — MindLogic KR",
    "k-exaone":                   "K-EXAONE            — LG AI KR",
    "solar-pro-3":                "Solar Pro 3         — Upstage KR",
    # ── Other ────────────────────────────────────────────────
    "sonar-pro":                  "Sonar Pro           — Perplexity",
}


# ════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ════════════════════════════════════════════════════════════
class VLMBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def info(self) -> dict: ...

    @abstractmethod
    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str: ...

    def img_to_b64(self, img: PILImage.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════════════
# BACKEND 1 — Cloud API  (model hot-swappable)
# ════════════════════════════════════════════════════════════
class APIBackend(VLMBackend):
    """OpenAI-compatible cloud API. Model can be changed at runtime."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self._lock   = threading.Lock()
        self.model   = model
        self.client  = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        log(f"API backend ready: {model} @ {base_url}", "VLM")

    def switch_model(self, model: str) -> str:
        """Hot-swap the model string — no reconnection needed."""
        with self._lock:
            old          = self.model
            self.model   = model
        log(f"API model switched: {old} → {model}", "VLM")
        return f"API model → {model}"

    @property
    def name(self) -> str:
        return "api"

    @property
    def info(self) -> dict:
        return {
            "backend": "api",
            "model":   self.model,
            "gpu":     False,
            "status":  "ready",
        }

    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str:
        b64 = self.img_to_b64(img)
        with self._lock:
            model = self.model
        try:
            resp = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"API error: {e}"


# ════════════════════════════════════════════════════════════
# BACKEND 2 — Local fine-tuned LLaVA
# ════════════════════════════════════════════════════════════
class LocalBackend(VLMBackend):
    """Local fine-tuned LLaVA-1.5 + LoRA. Loads in background thread."""

    def __init__(self, lora_path: str, base_model: str):
        self._model     = None
        self._processor = None
        self._device    = None
        self._ready     = False
        self._loading   = False
        self._error     = None
        self.lora_path  = lora_path
        self.base_model = base_model
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        self._loading = True
        log(f"Loading local model: {self.base_model}", "VLM")
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from peft import PeftModel
            self._device    = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self.base_model)
            self._processor.tokenizer.padding_side = "right"
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            base  = LlavaForConditionalGeneration.from_pretrained(
                self.base_model, torch_dtype=dtype,
                device_map={"": 0} if self._device == "cuda" else "cpu",
                low_cpu_mem_usage=True,
            )
            self._model   = PeftModel.from_pretrained(base, self.lora_path)
            self._model.eval()
            self._ready   = True
            self._loading = False
            log("✅ Local LLaVA ready!", "VLM")
        except Exception as e:
            self._error   = str(e)
            self._loading = False
            log(f"❌ Local model failed: {e}", "VLM")

    @property
    def name(self) -> str:
        return "local"

    @property
    def info(self) -> dict:
        return {
            "backend": "local",
            "model":   self.base_model,
            "status":  ("loading" if self._loading
                        else f"error: {self._error}" if self._error
                        else "ready" if self._ready else "not_started"),
        }

    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str:
        if self._loading: return "⏳ Local model loading..."
        if not self._ready: return f"❌ Local model not ready: {self._error}"
        import torch
        try:
            img_r = img.convert("RGB").resize((336, 336), PILImage.LANCZOS)
            text  = f"USER: <image>\n{prompt} ASSISTANT:"
            enc   = self._processor(text=text, images=img_r,
                                    return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model.generate(**enc, max_new_tokens=max_tokens,
                                           do_sample=False, temperature=1.0)
            return self._processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        except Exception as e:
            return f"Local inference error: {e}"


# ════════════════════════════════════════════════════════════
# BACKEND 3 — Remote GPU server
# ════════════════════════════════════════════════════════════
class RemoteBackend(VLMBackend):
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.url}/health", timeout=3)
            log(f"Remote backend ready: {self.url}", "VLM")
        except Exception:
            log(f"Remote backend unreachable: {self.url}", "VLM")

    @property
    def name(self) -> str: return "remote"

    @property
    def info(self) -> dict:
        return {"backend": "remote", "url": self.url, "gpu": "server"}

    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str:
        import json, urllib.request
        body = json.dumps({"prompt": prompt, "image": self.img_to_b64(img),
                           "max_tokens": max_tokens}).encode()
        req  = urllib.request.Request(f"{self.url}/infer", data=body,
                                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["response"]
        except Exception as e:
            return f"Remote error: {e}"