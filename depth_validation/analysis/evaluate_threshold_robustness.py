import os
import csv
import json
from collections import deque

import numpy as np


ROOT = (
    "/home/caim/final_vrosa_revised/"
    "depth_validation/data/"
    "threshold_robustness"
)

RESULT_ROOT = (
    "/home/caim/final_vrosa_revised/"
    "depth_validation/results"
)

CALIBRATION_PATH = os.path.join(
    RESULT_ROOT,
    "short_range_calibration.json",
)

ARRIVAL_THRESHOLD_M = 1.2

GT_TOLERANCE_M = 0.06


with open(
    CALIBRATION_PATH,
    "r",
) as f:

    calibration = json.load(
        f
    )


CAL_A = float(
    calibration["a"]
)

CAL_B = float(
    calibration["b"]
)


# ============================================================
# TARGET BBOX
# ============================================================

def find_target_bbox(
    gt,
    nominal,
):

    h, w = gt.shape

    cy = h // 2
    cx = w // 2


    candidate = (
        np.isfinite(gt)
        & (
            np.abs(
                gt - nominal
            )
            <= GT_TOLERANCE_M
        )
    )


    if not candidate.any():

        raise RuntimeError(
            "No target GT pixels."
        )


    if candidate[
        cy,
        cx
    ]:

        sy = cy
        sx = cx

    else:

        ys, xs = np.where(
            candidate
        )

        dist2 = (
            (ys - cy) ** 2
            + (xs - cx) ** 2
        )

        idx = int(
            np.argmin(
                dist2
            )
        )

        sy = int(
            ys[idx]
        )

        sx = int(
            xs[idx]
        )


    visited = np.zeros(
        candidate.shape,
        dtype=bool,
    )


    queue = deque([
        (
            sy,
            sx,
        )
    ])


    visited[
        sy,
        sx
    ] = True


    min_x = max_x = sx
    min_y = max_y = sy


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


    while queue:

        y, x = queue.popleft()

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


            queue.append(
                (
                    ny,
                    nx,
                )
            )


    return (
        min_x,
        min_y,
        max_x,
        max_y,
    )


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    rows,
):

    tp = tn = fp = fn = 0


    for r in rows:

        truth = r[
            "true_arrival"
        ]

        pred = r[
            "predicted_arrival"
        ]


        if truth and pred:
            tp += 1

        elif (
            (not truth)
            and (not pred)
        ):
            tn += 1

        elif (
            (not truth)
            and pred
        ):
            fp += 1

        else:
            fn += 1


    total = (
        tp + tn + fp + fn
    )


    accuracy = (
        (tp + tn)
        / total
        if total
        else 0.0
    )


    precision = (
        tp
        / (tp + fp)
        if (
            tp + fp
        )
        else 0.0
    )


    recall = (
        tp
        / (tp + fn)
        if (
            tp + fn
        )
        else 0.0
    )


    return {
        "n":
            total,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,
    }


# ============================================================
# LOAD SAMPLES
# ============================================================

rows = []


for sample_name in sorted(
    os.listdir(ROOT)
):

    sample_dir = os.path.join(
        ROOT,
        sample_name,
    )


    if not os.path.isdir(
        sample_dir
    ):
        continue


    meta_path = os.path.join(
        sample_dir,
        "frame_000001_metadata.json",
    )


    gt_path = os.path.join(
        sample_dir,
        "frame_000001_gt_z.npy",
    )


    da_path = os.path.join(
        sample_dir,
        "frame_000001_da_raw.npy",
    )


    if not (
        os.path.exists(meta_path)
        and os.path.exists(gt_path)
        and os.path.exists(da_path)
    ):
        continue


    with open(
        meta_path,
        "r",
    ) as f:

        meta = json.load(
            f
        )


    nominal = float(
        meta[
            "nominal_target_distance_m"
        ]
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


    x1, y1, x2, y2 = (
        find_target_bbox(
            gt,
            nominal,
        )
    )


    width = (
        x2 - x1 + 1
    )

    height = (
        y2 - y1 + 1
    )


    mx = int(
        round(
            width * 0.20
        )
    )

    my = int(
        round(
            height * 0.20
        )
    )


    cx1 = x1 + mx
    cx2 = x2 - mx

    cy1 = y1 + my
    cy2 = y2 - my


    gt_crop = gt[
        cy1:cy2+1,
        cx1:cx2+1,
    ]


    da_crop = da[
        cy1:cy2+1,
        cx1:cx2+1,
    ]


    valid = (
        np.isfinite(
            gt_crop
        )
        & np.isfinite(
            da_crop
        )
        & (gt_crop > 0.01)
        & (da_crop > 0.01)
    )


    gt_values = gt_crop[
        valid
    ]


    da_values = da_crop[
        valid
    ]


    gt_distance = float(
        np.median(
            gt_values
        )
    )


    # All threshold-test distances are < 2 m,
    # therefore current VROSA estimator = p10.
    raw_distance = float(
        np.percentile(
            da_values,
            10,
        )
    )


    calibrated = (
        CAL_A * raw_distance
        + CAL_B
    )


    error = (
        calibrated
        - gt_distance
    )


    true_arrival = (
        nominal
        <= ARRIVAL_THRESHOLD_M
    )


    predicted_arrival = (
        calibrated
        <= ARRIVAL_THRESHOLD_M
    )


    rows.append({

        "sample_name":
            sample_name,

        "distance_m":
            nominal,

        "gt_distance_m":
            gt_distance,

        "lighting":
            meta[
                "lighting_condition"
            ],

        "lighting_multiplier":
            meta[
                "lighting_multiplier"
            ],

        "offset":
            meta[
                "offset_condition"
            ],

        "offset_x_m":
            meta[
                "target_offset_x_m"
            ],

        "rgb_mean":
            meta[
                "rgb_mean"
            ],

        "raw_p10_m":
            raw_distance,

        "calibrated_distance_m":
            calibrated,

        "signed_error_m":
            error,

        "absolute_error_m":
            abs(
                error
            ),

        "true_arrival":
            true_arrival,

        "predicted_arrival":
            predicted_arrival,

        "decision_correct":
            (
                true_arrival
                == predicted_arrival
            ),

        "decision_margin_m":
            calibrated
            - ARRIVAL_THRESHOLD_M,
    })


# ============================================================
# OVERALL DEPTH METRICS
# ============================================================

errors = np.array(
    [
        r[
            "signed_error_m"
        ]
        for r in rows
    ],
    dtype=np.float64,
)


gts = np.array(
    [
        r[
            "gt_distance_m"
        ]
        for r in rows
    ],
    dtype=np.float64,
)


mae = float(
    np.mean(
        np.abs(
            errors
        )
    )
)


rmse = float(
    np.sqrt(
        np.mean(
            errors ** 2
        )
    )
)


bias = float(
    np.mean(
        errors
    )
)


mape = float(
    np.mean(
        np.abs(
            errors
        )
        / gts
    )
    * 100.0
)


overall_cls = (
    classification_metrics(
        rows
    )
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 78)
print("VROSA THRESHOLD ROBUSTNESS — FINAL")
print("=" * 78)

print(
    "Samples:",
    len(rows),
)

print(
    f"Calibrated MAE  : {mae:.4f} m"
)

print(
    f"Calibrated RMSE : {rmse:.4f} m"
)

print(
    f"Calibrated Bias : {bias:+.4f} m"
)

print(
    f"Calibrated MAPE : {mape:.2f}%"
)

print()
print(
    "TP:",
    overall_cls["tp"],
)

print(
    "TN:",
    overall_cls["tn"],
)

print(
    "False arrivals:",
    overall_cls["fp"],
)

print(
    "Missed arrivals:",
    overall_cls["fn"],
)

print(
    f"Accuracy : "
    f"{overall_cls['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{overall_cls['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{overall_cls['recall'] * 100:.2f}%"
)


# ============================================================
# BY LIGHTING
# ============================================================

print()
print("=" * 78)
print("BY LIGHTING")
print("=" * 78)


lighting_summary = {}


for lighting in [
    "dim",
    "normal",
    "bright",
]:

    subset = [
        r
        for r in rows
        if (
            r["lighting"]
            == lighting
        )
    ]


    cls = classification_metrics(
        subset
    )


    sub_errors = np.array([
        r["signed_error_m"]
        for r in subset
    ])


    sub_mae = float(
        np.mean(
            np.abs(
                sub_errors
            )
        )
    )


    rgb_mean = float(
        np.mean([
            r["rgb_mean"]
            for r in subset
        ])
    )


    lighting_summary[
        lighting
    ] = {
        **cls,
        "mae_m":
            sub_mae,
        "mean_rgb":
            rgb_mean,
    }


    print()
    print(
        lighting.upper()
    )

    print(
        f"  mean RGB      : "
        f"{rgb_mean:.2f}"
    )

    print(
        f"  MAE           : "
        f"{sub_mae:.4f} m"
    )

    print(
        f"  false arrivals: "
        f"{cls['fp']}"
    )

    print(
        f"  missed arrival: "
        f"{cls['fn']}"
    )

    print(
        f"  accuracy      : "
        f"{cls['accuracy'] * 100:.2f}%"
    )


# ============================================================
# CRITICAL / WRONG SAMPLES
# ============================================================

wrong = [
    r
    for r in rows
    if not r[
        "decision_correct"
    ]
]


closest = sorted(
    rows,
    key=lambda r:
        abs(
            r[
                "decision_margin_m"
            ]
        ),
)[:5]


print()
print("=" * 78)
print("CLOSEST SAMPLES TO 1.2 m DECISION BOUNDARY")
print("=" * 78)


for r in closest:

    print(
        f"{r['sample_name']:32s} "
        f"GT={r['gt_distance_m']:.3f} "
        f"CAL={r['calibrated_distance_m']:.3f} "
        f"margin={r['decision_margin_m']:+.4f}"
    )


if wrong:

    print()
    print("WRONG DECISIONS:")

    for r in wrong:

        print(
            r["sample_name"],
            "GT=",
            r["gt_distance_m"],
            "CAL=",
            r["calibrated_distance_m"],
        )

else:

    print()
    print(
        "✅ No incorrect arrival decisions."
    )


# ============================================================
# SAVE
# ============================================================

csv_path = os.path.join(
    RESULT_ROOT,
    "threshold_robustness_samples.csv",
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


summary = {

    "n_samples":
        len(rows),

    "calibration": {
        "a":
            CAL_A,

        "b":
            CAL_B,
    },

    "arrival_threshold_m":
        ARRIVAL_THRESHOLD_M,

    "mae_m":
        mae,

    "rmse_m":
        rmse,

    "bias_m":
        bias,

    "mape_percent":
        mape,

    "classification":
        overall_cls,

    "lighting":
        lighting_summary,
}


json_path = os.path.join(
    RESULT_ROOT,
    "threshold_robustness_summary.json",
)


with open(
    json_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "Saved:",
    csv_path,
)

print(
    "Saved:",
    json_path,
)
