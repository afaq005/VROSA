import os
import json
import time

import numpy as np
import torch
import transformers
from PIL import Image
import argparse
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


# ============================================================
# PATHS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Run DA-V2 metric depth "
        "on one validation sample"
    )
)

parser.add_argument(
    "--sample-dir",
    required=True,
    help=(
        "Directory containing "
        "frame_000001_rgb.png"
    ),
)

args = parser.parse_args()

ROOT = os.path.abspath(
    args.sample_dir
)

RGB_PATH = os.path.join(
    ROOT,
    "frame_000001_rgb.png",
)

OUT_PATH = os.path.join(
    ROOT,
    "frame_000001_da_raw.npy",
)

META_PATH = os.path.join(
    ROOT,
    "frame_000001_da_metadata.json",
)


MODEL_ID = (
    "depth-anything/"
    "Depth-Anything-V2-"
    "Metric-Indoor-Small-hf"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n==========================================")
print("VROSA DA-V2 ONE-FRAME VALIDATION")
print("==========================================")

print("Model :", MODEL_ID)
print("Device:", device)


# ============================================================
# LOAD IMAGE
# ============================================================

img = Image.open(
    RGB_PATH
).convert("RGB")

img_np = np.asarray(img)

print()
print("RGB image:")
print(" size :", img.size)
print(" min  :", int(img_np.min()))
print(" max  :", int(img_np.max()))
print(" mean :", float(img_np.mean()))
print(" std  :", float(img_np.std()))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading processor...")

processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_ID
    )
)

print("Loading DA-V2 model...")

model = (
    AutoModelForDepthEstimation
    .from_pretrained(
        MODEL_ID
    )
    .to(device)
    .eval()
)


print("\nModel configuration:")

print(
    "depth_estimation_type:",
    getattr(
        model.config,
        "depth_estimation_type",
        None,
    )
)

print(
    "max_depth:",
    getattr(
        model.config,
        "max_depth",
        None,
    )
)


# ============================================================
# VERIFY METRIC HEAD
# ============================================================

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

print(
    "transformers_version:",
    transformers.__version__,
)

print(
    "head_activation:",
    activation_name,
)

print(
    "head_max_depth:",
    head_max_depth,
)


# Fail immediately if the metric checkpoint is loaded incorrectly.
if (
    getattr(
        model.config,
        "depth_estimation_type",
        None,
    )
    != "metric"
):
    raise RuntimeError(
        "DA-V2 checkpoint is not loaded "
        "as metric depth."
    )

if activation_name != "Sigmoid":
    raise RuntimeError(
        "Incorrect DA-V2 metric head: "
        f"expected Sigmoid, got {activation_name}"
    )

if head_max_depth is None:
    raise RuntimeError(
        "Metric DA-V2 head has no max_depth."
    )

print(
    "✅ Metric DA-V2 head verified"
)



# ============================================================
# PREPROCESS
# ============================================================

inputs = processor(
    images=img,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print(
    "Input tensor:",
    inputs["pixel_values"].shape,
)


# ============================================================
# INFERENCE
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()

with torch.no_grad():

    outputs = model(
        **inputs
    )

if device.type == "cuda":
    torch.cuda.synchronize()

latency_ms = (
    time.perf_counter()
    - t0
) * 1000.0


print(
    f"\nInference latency: "
    f"{latency_ms:.2f} ms"
)


# ============================================================
# INSPECT RAW OUTPUT
# ============================================================

raw_pred = (
    outputs.predicted_depth
)

print(
    "Raw predicted tensor:",
    raw_pred.shape,
)

print(
    "Raw finite:",
    int(
        torch.isfinite(
            raw_pred
        ).sum().item()
    ),
    "/",
    raw_pred.numel(),
)


# ============================================================
# DEPTH OUTPUT
# ============================================================

# For Depth Anything V2, outputs.predicted_depth contains
# the predicted metric depth at the model output resolution.
#
# The current Transformers DPTImageProcessor does not provide
# post_process_depth_estimation(), so we use the model output
# directly and explicitly resize it to the original RGB image
# resolution.

pred_model = (
    outputs.predicted_depth
    .detach()
    .float()
)

# predicted_depth shape:
# [batch, height, width]
pred_model = pred_model[0]

# Resize depth to the original RGB resolution.
#
# IMPORTANT:
# Interpolation is only spatial resizing. It does not change
# the metric depth values.
pred = torch.nn.functional.interpolate(
    pred_model.unsqueeze(0).unsqueeze(0),
    size=(
        img.height,
        img.width,
    ),
    mode="bicubic",
    align_corners=False,
)[0, 0]

pred = (
    pred
    .cpu()
    .numpy()
    .astype(np.float32)
)


# ============================================================
# DIAGNOSTICS
# ============================================================

finite = np.isfinite(pred)

positive = (
    finite
    & (pred > 0.0)
)

print("\n==========================================")
print("DA-V2 DEPTH OUTPUT")
print("==========================================")

print(
    "Shape:",
    pred.shape,
)

print(
    "Finite pixels:",
    int(finite.sum()),
    "/",
    pred.size,
)

print(
    "Positive pixels:",
    int(positive.sum()),
    "/",
    pred.size,
)


if positive.any():

    p = pred[positive]

    print(
        "Min    :",
        float(p.min()),
    )

    print(
        "P01    :",
        float(
            np.percentile(
                p,
                1,
            )
        ),
    )

    print(
        "P05    :",
        float(
            np.percentile(
                p,
                5,
            )
        ),
    )

    print(
        "P25    :",
        float(
            np.percentile(
                p,
                25,
            )
        ),
    )

    print(
        "Median :",
        float(
            np.median(p)
        ),
    )

    print(
        "P75    :",
        float(
            np.percentile(
                p,
                75,
            )
        ),
    )

    print(
        "P95    :",
        float(
            np.percentile(
                p,
                95,
            )
        ),
    )

    print(
        "Max    :",
        float(p.max()),
    )

    print(
        "Mean   :",
        float(p.mean()),
    )


    # --------------------------------------------------------
    # Center 41x41 region
    # --------------------------------------------------------

    cy = pred.shape[0] // 2
    cx = pred.shape[1] // 2

    half = 20

    patch = pred[
        cy-half:cy+half+1,
        cx-half:cx+half+1,
    ]

    patch_mask = (
        np.isfinite(patch)
        & (patch > 0.0)
    )

    if patch_mask.any():

        center_median = float(
            np.median(
                patch[
                    patch_mask
                ]
            )
        )

    else:

        center_median = None

    print(
        "Center 41x41 median:",
        center_median,
    )

else:

    center_median = None

    print(
        "❌ DA-V2 produced no "
        "positive finite values"
    )


# ============================================================
# # SAVE RAW, UNCALIBRATED METRIC DEPTH
# ============================================================

np.save(
    OUT_PATH,
    pred,
)


metadata = {
    "model":
        MODEL_ID,
    "transformers_version":
        transformers.__version__,

    "head_activation":
        activation_name,

    "head_max_depth":
        head_max_depth,

    "resize_mode":
        "bicubic",

    "raw_metric_output":
        True,

    "depth_estimation_type":
        getattr(
            model.config,
            "depth_estimation_type",
            None,
        ),

    "max_depth":
        getattr(
            model.config,
            "max_depth",
            None,
        ),

    "prediction_shape":
        list(pred.shape),

    "finite_pixels":
        int(finite.sum()),

    "positive_pixels":
        int(positive.sum()),

    "latency_ms":
        float(latency_ms),

    # IMPORTANT:
    # no calibration applied
    "depth_scale":
        1.0,

    "depth_bias":
        0.0,

    "center_41x41_da_median_m":
        center_median,
}


if positive.any():

    p = pred[positive]

    metadata.update({
        "da_min_m":
            float(p.min()),

        "da_p05_m":
            float(
                np.percentile(
                    p,
                    5,
                )
            ),

        "da_p25_m":
            float(
                np.percentile(
                    p,
                    25,
                )
            ),

        "da_median_m":
            float(
                np.median(p)
            ),

        "da_p75_m":
            float(
                np.percentile(
                    p,
                    75,
                )
            ),

        "da_p95_m":
            float(
                np.percentile(
                    p,
                    95,
                )
            ),

        "da_mean_m":
            float(p.mean()),

        "da_max_m":
            float(p.max()),
    })


with open(
    META_PATH,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\n==========================================")
print("SAVED")
print("==========================================")

print(OUT_PATH)
print(META_PATH)

print(
    "\n✅ DA-V2 one-frame inference complete"
)