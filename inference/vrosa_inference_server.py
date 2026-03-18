"""
V-ROSA Inference Server
=======================
Serves the fine-tuned LLaVA-1.5 + LoRA model on a remote GPU (e.g. A100).
The main vrosa_live.py connects to this via --vlm remote.

Usage (on the A100 server):
    python inference/vrosa_inference_server.py \
        --model /path/to/vrosa_vlm_v4 \
        --base  llava-hf/llava-1.5-7b-hf \
        --port  7860

Then on robot machine:
    python vrosa/vrosa_live.py --vlm remote --remote-url http://SERVER_IP:7860
"""

import argparse, base64, io, time, logging
from flask import Flask, request, jsonify
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./weights/vrosa_vlm_v4",
                    help="Path to fine-tuned LoRA adapter")
parser.add_argument("--base",  default="llava-hf/llava-1.5-7b-hf",
                    help="Base model identifier")
parser.add_argument("--port",  type=int, default=7860)
args = parser.parse_args()

app     = Flask(__name__)
_model     = None
_processor = None
_device    = None

def load_model():
    global _model, _processor, _device
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SERVER] Loading on {_device}...")

    _processor = AutoProcessor.from_pretrained(args.base)
    _processor.tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if _device == "cuda" else torch.float32
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": 0} if _device == "cuda" else "cpu",
        low_cpu_mem_usage=True,
    )
    _model = PeftModel.from_pretrained(base, args.model)
    _model.eval()
    print("[SERVER] ✅ Model ready!")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": args.model, "device": _device})


@app.route("/infer", methods=["POST"])
def infer():
    import torch
    data       = request.get_json()
    prompt     = data.get("prompt", "Describe the scene.")
    image_b64  = data.get("image", "")
    max_tokens = data.get("max_tokens", 400)

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(
        (336, 336), Image.LANCZOS
    )

    text = f"USER: <image>\n{prompt} ASSISTANT:"
    enc  = _processor(
        text=text, images=img,
        return_tensors="pt",
        truncation=True, max_length=512,
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    latency = time.time() - t0

    full   = _processor.decode(out[0], skip_special_tokens=True)
    answer = full.split("ASSISTANT:")[-1].strip()

    return jsonify({"response": answer, "latency_s": round(latency, 2)})


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    load_model()
    print(f"[SERVER] Listening on port {args.port}")
    print(f"[SERVER] POST /infer  {{prompt, image_base64, max_tokens}}")
    print(f"[SERVER] GET  /health")
    app.run(host="0.0.0.0", port=args.port, threaded=False)
