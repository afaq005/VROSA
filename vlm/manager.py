"""vlm/manager.py — Runtime VLM backend registry with experiment logging."""

from __future__ import annotations

import os
import threading
import time

from PIL import Image as PILImage

from config import args
from experiment_logger import record_event, record_latency
from logger import log
from state import state
from vlm.backends import APIBackend, LocalBackend, RemoteBackend, VLMBackend


class VLMManager:
    def __init__(self):
        self._backends: dict[str, VLMBackend] = {}
        self._active = "api"
        self._lock = threading.Lock()

    def register(self, backend: VLMBackend) -> None:
        self._backends[backend.name] = backend
        log(f"Registered VLM backend: {backend.name}", "VLM")

    def switch(self, mode: str) -> str:
        if mode not in self._backends:
            return f"Unknown backend '{mode}'. Available: {list(self._backends)}"
        with self._lock:
            self._active = mode
        record_event("VLM_BACKEND_SWITCH", backend=mode)
        log(f"VLM switched → {mode}", "VLM")
        return f"Switched to {mode} backend"

    def switch_model(self, model: str) -> str:
        backend = self._backends.get(self._active)
        if hasattr(backend, "switch_model"):
            result = backend.switch_model(model)
            record_event("VLM_MODEL_SWITCH", backend=self._active, model=model)
            return result
        return f"Active backend '{self._active}' does not support model switching"

    @property
    def active_model(self) -> str:
        backend = self._backends.get(self._active)
        return getattr(backend, "model", getattr(backend, "base_model", ""))

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def active(self) -> VLMBackend:
        return self._backends[self._active]

    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str:
        backend = self.active
        state["vlm_query_count"] = int(state.get("vlm_query_count", 0)) + 1
        qid = state["vlm_query_count"]
        model = self.active_model
        record_event("VLM_REQUEST", query_index=qid, backend=self._active, model=model,
                     prompt_preview=prompt[:160])
        t0 = time.perf_counter_ns()
        success = True
        try:
            response = backend.ask(prompt, img, max_tokens)
            low = (response or "").lower()
            if (not response or "api error" in low or "remote error" in low or
                    "inference error" in low or "not ready" in low or "timeout" in low):
                success = False
            return response
        except Exception as e:
            success = False
            response = f"VLM error: {e}"
            return response
        finally:
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            record_latency("vlm_backend", dt_ms, success=success,
                           backend=self._active, model=model, query_index=qid)
            record_event("VLM_RESPONSE", query_index=qid, backend=self._active,
                         model=model, success=success,
                         response_preview=(response[:160] if 'response' in locals() and response else ""))

    def status(self) -> dict:
        return {
            "active": self._active,
            "backends": {k: v.info for k, v in self._backends.items()},
        }


vlm_mgr = VLMManager()
vlm_mgr.register(APIBackend(api_key=args.api_key, base_url=args.api_base, model=args.api_model))

if args.vlm == "local" or (args.local_model and os.path.exists(args.local_model)):
    vlm_mgr.register(LocalBackend(lora_path=args.local_model, base_model=args.local_base))

if args.remote_url:
    vlm_mgr.register(RemoteBackend(args.remote_url))

if args.vlm in vlm_mgr._backends:
    vlm_mgr.switch(args.vlm)
elif args.vlm != "api":
    log(f"Requested VLM backend '{args.vlm}' unavailable; using API", "WARN")

log(f"Active VLM backend: {vlm_mgr.active_name} ({vlm_mgr.active_model})", "VLM")
