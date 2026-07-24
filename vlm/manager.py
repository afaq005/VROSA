"""
vlm/manager.py — VLMManager handles hot-swapping of VLM backends at runtime.
Also initializes the vlm_mgr singleton based on CLI args.
"""

import os
import threading

from PIL import Image as PILImage

from config import args
from logger import log
from vlm.backends import VLMBackend, APIBackend, LocalBackend, RemoteBackend


# ════════════════════════════════════════════════════════════
# VLM MANAGER
# ════════════════════════════════════════════════════════════
class VLMManager:
    """Registry of VLM backends with hot-swap support."""

    def __init__(self):
        self._backends: dict[str, VLMBackend] = {}
        self._active:   str = "api"
        self._lock = threading.Lock()

    def register(self, backend: VLMBackend) -> None:
        self._backends[backend.name] = backend
        log(f"Registered VLM backend: {backend.name}", "VLM")

    def switch(self, mode: str) -> str:
        if mode not in self._backends:
            return f"Unknown backend '{mode}'. Available: {list(self._backends)}"
        with self._lock:
            self._active = mode
        log(f"VLM switched → {mode}", "VLM")
        return f"Switched to {mode} backend"

    def switch_model(self, model: str) -> str:
        """Change the model string on the active API backend (no reconnect needed)."""
        backend = self._backends.get(self._active)
        if hasattr(backend, "switch_model"):
            return backend.switch_model(model)
        return f"Active backend '{self._active}' does not support model switching"

    @property
    def active_model(self) -> str:
        """Return the model string of the active backend if available."""
        backend = self._backends.get(self._active)
        return getattr(backend, "model", "")

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def active(self) -> VLMBackend:
        return self._backends[self._active]

    def ask(self, prompt: str, img: PILImage.Image, max_tokens: int = 400) -> str:
        return self.active.ask(prompt, img, max_tokens)

    def status(self) -> dict:
        return {
            "active":   self._active,
            "backends": {k: v.info for k, v in self._backends.items()},
        }


# vlm_mgr = VLMManager()

# vlm_mgr.register(APIBackend(
#     api_key=args.api_key,
#     base_url=args.api_base,
#     model=args.api_model,
# ))

# log(f"Hardware mode: API backend only — {args.api_model}", "VLM")
# log(f"Active VLM backend: {vlm_mgr.active_name}", "VLM")
# ════════════════════════════════════════════════════════════
# SINGLETON INITIALIZATION
# ════════════════════════════════════════════════════════════
vlm_mgr = VLMManager()

# Always register the cloud API backend
vlm_mgr.register(APIBackend(
    api_key=args.api_key,
    base_url=args.api_base,
    model=args.api_model,
))

# Register local backend only if path exists or --vlm local was requested
if args.vlm != "remote" and (args.vlm == "local" or os.path.exists(args.local_model)):
    vlm_mgr.register(LocalBackend(
        lora_path=args.local_model,
        base_model=args.local_base,
    ))
    if args.vlm == "local":
        vlm_mgr.switch("local")
        log("Starting in LOCAL model mode", "VLM")
    else:
        log("Local model found — available but not active", "VLM")
elif args.vlm == "remote":
    log("Remote mode — skipping local model load (saves VRAM)", "VLM")
else:
    log("Local model path not found — API only mode", "VLM")

# Register remote backend if a URL was provided
if args.remote_url:
    vlm_mgr.register(RemoteBackend(args.remote_url))
    if args.vlm == "remote":
        vlm_mgr.switch("remote")
        log("Starting in REMOTE model mode", "VLM")

log(f"Active VLM backend: {vlm_mgr.active_name}", "VLM")