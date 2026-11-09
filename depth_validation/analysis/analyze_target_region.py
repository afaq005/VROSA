import os
import csv
from collections import deque

import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    "/home/caim/final_vrosa_revised/"
    "depth_validation/data"
)

RESULT_ROOT = (
    "/home/caim/final_vrosa_revised/"
    "depth_validation/results"
)

os.makedirs(
    RESULT_ROOT,
    exist_ok=True,
)


# ============================================================
# DATASETS
# ============================================================

SAMPLES = [
    {
        "name": "range_1m",
        "condition": "plain",
        "distance": 1.0,
    },
    {
        "name": "range_2m",
        "condition": "plain",
        "distance": 2.0,
    },
    {
        "name": "range_3m",
        "condition": "plain",
        "distance": 3.0,
    },
    {
        "name": "range_5m",
        "condition": "plain",
        "distance": 5.0,
    },
    {
        "name": "range_8m",
        "condition": "plain",
        "distance": 8.0,
    },
    {
        "name": "range_10m",
        "condition": "plain",
        "distance": 10.0,
    },
    {
        "name": "appearance_checker_3m",
        "condition": "checker",
        "distance": 3.0,
    },
    {
        "name": "appearance_box_3m",
        "condition": "3d_box",
        "distance": 3.0,
    },
]


# ============================================================
# TARGET GT TOLERANCE
#
# Our controlled target front surface is exactly at the
# nominal distance.
#
# +/- 0.06 m also captures the very shallow decorative
# strips on the 3-D box.
# ============================================================

GT_TOLERANCE_M = 0.06


# ============================================================
# FIND CONNECTED TARGET COMPONENT
# ============================================================

def find_target_bbox(
    gt,
    nominal_distance,
):

    h, w = gt.shape

    cy = h // 2
    cx = w // 2

    # --------------------------------------------------------
    # Candidate pixels at the known target distance.
    # --------------------------------------------------------

    candidate = (
        np.isfinite(gt)
        & (
            np.abs(
                gt - nominal_distance
            )
            <= GT_TOLERANCE_M
        )
    )

    count = int(
        candidate.sum()
    )

    if count == 0:

        raise RuntimeError(
            "No pixels found near "
            f"{nominal_distance} m"
        )

    # --------------------------------------------------------
    # Find candidate pixel nearest image center.
    #
    # Normally the exact center is already on the target,
    # but this handles checker seams etc.
    # --------------------------------------------------------

    if candidate[cy, cx]:

        seed_y = cy
        seed_x = cx

    else:

        ys, xs = np.where(
            candidate
        )

        dist2 = (
            (ys - cy) ** 2
            + (xs - cx) ** 2
        )

        idx = int(
            np.argmin(dist2)
        )

        seed_y = int(
            ys[idx]
        )

        seed_x = int(
            xs[idx]
        )

    # --------------------------------------------------------
    # Flood fill the connected target surface.
    # --------------------------------------------------------

    visited = np.zeros(
        candidate.shape,
        dtype=bool,
    )

    q = deque()

    q.append(
        (seed_y, seed_x)
    )

    visited[
        seed_y,
        seed_x
    ] = True


    min_x = seed_x
    max_x = seed_x

    min_y = seed_y
    max_y = seed_y

    component_pixels = 0


    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]


    while q:

        y, x = q.popleft()

        component_pixels += 1

        min_x = min(
            min_x,
            x,
        )

        max_x = max(
            max_x,
            x,
        )

        min_y = min(
            min_y,
            y,
        )

        max_y = max(
            max_y,
            y,
        )


        for dy, dx in neighbors:

            ny = y + dy
            nx = x + dx

            if (
                ny < 0
                or ny >= h
                or nx < 0
                or nx >= w
            ):
                continue

            if visited[
                ny,
                nx
            ]:
                continue

            if not candidate[
                ny,
                nx
            ]:
                continue

            visited[
                ny,
                nx
            ] = True

            q.append(
                (ny, nx)
            )


    return {
        "x1": min_x,
        "y1": min_y,
        "x2": max_x,
        "y2": max_y,

        "width":
            max_x - min_x + 1,

        "height":
            max_y - min_y + 1,

        "pixels":
            component_pixels,
    }


# ============================================================
# CENTRAL 60% OF BBOX
# ============================================================

def central_60_bbox(
    bbox,
):

    x1 = bbox["x1"]
    y1 = bbox["y1"]

    x2 = bbox["x2"]
    y2 = bbox["y2"]

    width = (
        x2 - x1 + 1
    )

    height = (
        y2 - y1 + 1
    )

    # Remove 20% from each side.
    margin_x = int(
        round(
            width * 0.20
        )
    )

    margin_y = int(
        round(
            height * 0.20
        )
    )


    cx1 = (
        x1 + margin_x
    )

    cx2 = (
        x2 - margin_x
    )

    cy1 = (
        y1 + margin_y
    )

    cy2 = (
        y2 - margin_y
    )


    return (
        cx1,
        cy1,
        cx2,
        cy2,
    )


# ============================================================
# ANALYZE ONE SAMPLE
# ============================================================

def analyze_sample(
    sample,
):

    folder = sample["name"]

    nominal = float(
        sample["distance"]
    )


    sample_dir = os.path.join(
        DATA_ROOT,
        folder,
    )


    gt_path = os.path.join(
        sample_dir,
        "frame_000001_gt_z.npy",
    )


    da_path = os.path.join(
        sample_dir,
        "frame_000001_da_raw.npy",
    )


    gt = np.load(
        gt_path
    ).astype(
        np.float32
    )


    da = np.load(
        da_path
    ).astype(
        np.float32
    )


    if gt.shape != da.shape:

        raise RuntimeError(
            f"Shape mismatch: {folder}"
        )


    # ========================================================
    # FIND TARGET
    # ========================================================

    bbox = find_target_bbox(
        gt,
        nominal,
    )


    (
        x1,
        y1,
        x2,
        y2,
    ) = central_60_bbox(
        bbox
    )


    # Python slicing requires +1
    gt_crop = gt[
        y1:y2+1,
        x1:x2+1,
    ]


    da_crop = da[
        y1:y2+1,
        x1:x2+1,
    ]


    valid = (
        np.isfinite(gt_crop)
        & np.isfinite(da_crop)
        & (gt_crop > 0.01)
        & (da_crop > 0.01)
    )


    if not valid.any():

        raise RuntimeError(
            f"No valid central target "
            f"pixels: {folder}"
        )


    gt_values = gt_crop[
        valid
    ]


    da_values = da_crop[
        valid
    ]


    # ========================================================
    # GT
    # ========================================================

    gt_median = float(
        np.median(
            gt_values
        )
    )


    # ========================================================
    # DA PERCENTILES
    # ========================================================

    p10 = float(
        np.percentile(
            da_values,
            10,
        )
    )

    p20 = float(
        np.percentile(
            da_values,
            20,
        )
    )

    p25 = float(
        np.percentile(
            da_values,
            25,
        )
    )

    p30 = float(
        np.percentile(
            da_values,
            30,
        )
    )

    p40 = float(
        np.percentile(
            da_values,
            40,
        )
    )

    p45 = float(
        np.percentile(
            da_values,
            45,
        )
    )

    p50 = float(
        np.percentile(
            da_values,
            50,
        )
    )


    # ========================================================
    # CURRENT VROSA ADAPTIVE RULE
    #
    # < 2 m  -> p10
    # 2–5 m  -> p25
    # >= 5 m -> p45
    #
    # We use GT distance here ONLY for controlled validation
    # of the intended percentile ranges.
    # Runtime VROSA does not have GT.
    # ========================================================

    if nominal < 2.0:

        adaptive_p = 10
        adaptive_da = p10

    elif nominal < 5.0:

        adaptive_p = 25
        adaptive_da = p25

    else:

        adaptive_p = 45
        adaptive_da = p45


    signed_error = (
        adaptive_da
        - gt_median
    )


    abs_error = abs(
        signed_error
    )


    rel_error = (
        abs_error
        / gt_median
        * 100.0
    )


    return {

        "sample":
            folder,

        "condition":
            sample["condition"],

        "nominal_distance_m":
            nominal,

        "target_bbox_width_px":
            bbox["width"],

        "target_bbox_height_px":
            bbox["height"],

        "target_component_pixels":
            bbox["pixels"],

        "central60_width_px":
            x2 - x1 + 1,

        "central60_height_px":
            y2 - y1 + 1,

        "central60_valid_pixels":
            int(valid.sum()),

        "gt_median_m":
            gt_median,

        "da_p10_m":
            p10,

        "da_p20_m":
            p20,

        "da_p25_m":
            p25,

        "da_p30_m":
            p30,

        "da_p40_m":
            p40,

        "da_p45_m":
            p45,

        "da_p50_m":
            p50,

        "adaptive_percentile":
            adaptive_p,

        "adaptive_da_m":
            adaptive_da,

        "adaptive_signed_error_m":
            signed_error,

        "adaptive_abs_error_m":
            abs_error,

        "adaptive_relative_error_percent":
            rel_error,
    }


# ============================================================
# RUN
# ============================================================

rows = []


print()
print("=" * 110)
print(
    "VROSA TARGET-REGION / PERCENTILE VALIDATION"
)
print("=" * 110)


for sample in SAMPLES:

    result = analyze_sample(
        sample
    )

    rows.append(
        result
    )


    print()
    print(
        f"{result['sample']}"
    )

    print(
        f"  GT          : "
        f"{result['gt_median_m']:.3f} m"
    )

    print(
        f"  bbox        : "
        f"{result['target_bbox_width_px']} x "
        f"{result['target_bbox_height_px']} px"
    )

    print(
        f"  central 60% : "
        f"{result['central60_width_px']} x "
        f"{result['central60_height_px']} px"
    )

    print(
        f"  p10         : "
        f"{result['da_p10_m']:.3f} m"
    )

    print(
        f"  p25         : "
        f"{result['da_p25_m']:.3f} m"
    )

    print(
        f"  p45         : "
        f"{result['da_p45_m']:.3f} m"
    )

    print(
        f"  median      : "
        f"{result['da_p50_m']:.3f} m"
    )

    print(
        f"  adaptive    : p"
        f"{result['adaptive_percentile']} "
        f"= "
        f"{result['adaptive_da_m']:.3f} m"
    )

    print(
        f"  error       : "
        f"{result['adaptive_signed_error_m']:+.3f} m "
        f"("
        f"{result['adaptive_relative_error_percent']:.2f}%"
        f")"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULT_ROOT,
    "target_region_percentile_analysis.csv",
)


with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()
print("=" * 110)

print(
    "Saved:",
    csv_path,
)

print("=" * 110)
