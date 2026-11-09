import os
import json
import time
import argparse

import numpy as np
import torch
import transformers

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


MODEL_ID = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Indoor-Small-hf"
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--root",
    required=True,
)

args = parser.parse_args()

ROOT = os.path.abspath(
    args.root
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 70)
print("VROSA BATCH DA-V2 INFERENCE")
print("=" * 70)

print("Root   :", ROOT)
print("Model  :", MODEL_ID)
print("Device :", device)


processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_ID
    )
)


model = (
    AutoModelForDepthEstimation
    .from_pretrained(
        MODEL_ID
    )
    .to(device)
    .eval()
)


activation_name = (
    model.head.activation2
    .__class__
    .__name__
)


head_max_depth = getattr(
    model.head,
    "max_depth",
    None,
)


if (
    getattr(
        model.config,
        "depth_estimation_type",
        None,
    )
    != "metric"
):

    raise RuntimeError(
        "Model is not metric depth."
    )


if activation_name != "Sigmoid":

    raise RuntimeError(
        "Invalid metric model head."
    )


if head_max_depth is None:

    raise RuntimeError(
        "Metric max_depth missing."
    )


print(
    "Transformers:",
    transformers.__version__,
)

print(
    "Head:",
    activation_name,
)

print(
    "Max depth:",
    head_max_depth,
)


sample_dirs = []


for name in sorted(
    os.listdir(ROOT)
):

    path = os.path.join(
        ROOT,
        name,
    )

    if not os.path.isdir(
        path
    ):
        continue

    rgb_path = os.path.join(
        path,
        "frame_000001_rgb.png",
    )

    if os.path.exists(
        rgb_path
    ):

        sample_dirs.append(
            path
        )


print()
print(
    "Samples found:",
    len(sample_dirs),
)


if len(sample_dirs) == 0:

    raise RuntimeError(
        "No RGB samples found."
    )


latencies = []


for index, sample_dir in enumerate(
    sample_dirs,
    start=1,
):

    sample_name = os.path.basename(
        sample_dir
    )


    rgb_path = os.path.join(
        sample_dir,
        "frame_000001_rgb.png",
    )


    out_path = os.path.join(
        sample_dir,
        "frame_000001_da_raw.npy",
    )


    metadata_path = os.path.join(
        sample_dir,
        "frame_000001_da_metadata.json",
    )


    image = (
        Image.open(
            rgb_path
        )
        .convert(
            "RGB"
        )
    )


    original_w = image.width
    original_h = image.height


    inputs = processor(
        images=image,
        return_tensors="pt",
    )


    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    if device == "cuda":
        torch.cuda.synchronize()


    t0 = time.perf_counter()


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    if device == "cuda":
        torch.cuda.synchronize()


    latency_ms = (
        time.perf_counter()
        - t0
    ) * 1000.0


    pred = (
        outputs.predicted_depth
        .detach()
        .float()
    )


    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(
            original_h,
            original_w,
        ),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)[0]


    depth = (
        pred.cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    np.save(
        out_path,
        depth,
    )


    metadata = {

        "sample_name":
            sample_name,

        "model":
            MODEL_ID,

        "transformers_version":
            transformers.__version__,

        "depth_estimation_type":
            "metric",

        "head_activation":
            activation_name,

        "head_max_depth":
            float(
                head_max_depth
            ),

        "resize_mode":
            "bicubic",

        "depth_scale":
            1.0,

        "depth_bias":
            0.0,

        "inference_latency_ms":
            latency_ms,

        "finite_pixels":
            int(
                np.isfinite(
                    depth
                ).sum()
            ),

        "positive_pixels":
            int(
                (
                    depth > 0
                ).sum()
            ),
    }


    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    latencies.append(
        latency_ms
    )


    print(
        f"[{index:02d}/{len(sample_dirs):02d}] "
        f"{sample_name:30s} "
        f"{latency_ms:7.2f} ms"
    )


print()
print("=" * 70)

print(
    "Mean latency:",
    f"{np.mean(latencies):.2f} ms",
)

print(
    "Std latency :",
    f"{np.std(latencies):.2f} ms",
)

print(
    "✅ Batch inference complete"
)
