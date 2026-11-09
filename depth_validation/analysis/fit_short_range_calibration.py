import numpy as np


# ============================================================
# CALIBRATION DATA
#
# These are the CURRENT VROSA adaptive-estimator outputs:
#
# 1 m -> p10
# 2 m -> p25
# 3 m -> p25
# 5 m -> p45
# ============================================================

gt = np.array(
    [
        1.0,
        2.0,
        3.0,
        5.0,
    ],
    dtype=np.float64,
)


raw = np.array(
    [
        1.918,
        3.773,
        5.286,
        9.242,
    ],
    dtype=np.float64,
)


# ============================================================
# FIT:
#
# GT = a * RAW + b
# ============================================================

a, b = np.polyfit(
    raw,
    gt,
    1,
)


pred = (
    a * raw
    + b
)


error = (
    pred - gt
)


mae = float(
    np.mean(
        np.abs(error)
    )
)


rmse = float(
    np.sqrt(
        np.mean(
            error ** 2
        )
    )
)


bias = float(
    np.mean(error)
)


print()
print("=" * 65)
print("VROSA SHORT-RANGE DEPTH CALIBRATION")
print("=" * 65)

print(
    f"a = {a:.8f}"
)

print(
    f"b = {b:.8f}"
)

print()
print(
    "Calibration equation:"
)

print(
    f"d_cal = "
    f"{a:.6f} * d_raw "
    f"{b:+.6f}"
)


print()
print(
    f"{'GT':>8}"
    f"{'RAW':>10}"
    f"{'CAL':>10}"
    f"{'ERROR':>10}"
)

print("-" * 40)


for g, r, p in zip(
    gt,
    raw,
    pred,
):

    print(
        f"{g:8.3f}"
        f"{r:10.3f}"
        f"{p:10.3f}"
        f"{p-g:+10.3f}"
    )


print()
print(
    f"Calibration-set MAE  : "
    f"{mae:.4f} m"
)

print(
    f"Calibration-set RMSE : "
    f"{rmse:.4f} m"
)

print(
    f"Calibration-set Bias : "
    f"{bias:+.4f} m"
)


# ============================================================
# APPEARANCE SANITY CHECK AT 3 M
# ============================================================

appearance = {
    "plain":
        5.286,

    "checker":
        5.881,

    "synthetic_box":
        5.612,

    "real_box":
        5.7822,
}


print()
print("=" * 65)
print("3 m APPEARANCE SANITY CHECK")
print("=" * 65)


for name, r in appearance.items():

    calibrated = (
        a * r
        + b
    )

    error = (
        calibrated - 3.0
    )

    print(
        f"{name:15s} "
        f"raw={r:6.3f} m   "
        f"cal={calibrated:6.3f} m   "
        f"error={error:+6.3f} m"
    )
