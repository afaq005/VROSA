import os
import csv
import json
import asyncio
import numpy as np

from PIL import Image

import omni.usd
import omni.kit.app
import omni.replicator.core as rep

from pxr import (
    UsdGeom,
    Gf,
)


# ============================================================
# EXPERIMENT DESIGN
# ============================================================

DISTANCES_M = [
    1.1,
    1.2,
    1.3,
    1.4,
]

LIGHTING = {
    "dim": 0.25,
    "normal": 1.00,
    "bright": 2.00,
}

OFFSETS_M = {
    "left": -0.10,
    "center": 0.00,
    "right": 0.10,
}


# ============================================================
# FIXED SETTINGS
# ============================================================

CAMERA_PRIM = (
    "/Nova_Carter_ROS/"
    "chassis_link/front_hawk/right/camera_right"
)

TARGET_PATH = (
    CAMERA_PRIM
    + "/VROSA_DepthValidationTarget"
)

TARGET_WIDTH_M = 0.50
TARGET_HEIGHT_M = 0.50
TARGET_THICKNESS_M = 0.02

WIDTH = 1920
HEIGHT = 1200

DATA_ROOT = (
    "/home/caim/final_vrosa_revised/"
    "depth_validation/data/"
    "threshold_robustness"
)

MANIFEST_PATH = os.path.join(
    DATA_ROOT,
    "manifest.csv",
)

LIGHT_TYPES = {
    "RectLight",
    "SphereLight",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "CylinderLight",
}


# ============================================================
# HELPERS
# ============================================================

def distance_tag(value):

    return (
        f"{value:.1f}"
        .replace(".", "p")
    )


async def wait_updates(n):

    for _ in range(n):

        await (
            omni.kit.app
            .get_app()
            .next_update_async()
        )


# ============================================================
# MAIN
# ============================================================

async def run_experiment():

    print()
    print("=" * 70)
    print("VROSA AUTOMATED THRESHOLD ROBUSTNESS CAPTURE")
    print("=" * 70)

    os.makedirs(
        DATA_ROOT,
        exist_ok=True,
    )

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    meters_per_unit = (
        UsdGeom.GetStageMetersPerUnit(
            stage
        )
    )

    print(
        "Meters per unit:",
        meters_per_unit,
    )


    camera = stage.GetPrimAtPath(
        CAMERA_PRIM
    )

    if not camera.IsValid():

        raise RuntimeError(
            "Camera not found: "
            + CAMERA_PRIM
        )


    # ========================================================
    # DISCOVER LIGHTS AND SAVE ORIGINAL INTENSITIES
    # ========================================================

    lights = []

    for prim in stage.Traverse():

        if (
            prim.GetTypeName()
            not in LIGHT_TYPES
        ):
            continue

        attr = prim.GetAttribute(
            "inputs:intensity"
        )

        if not attr.IsValid():
            continue

        value = attr.Get()

        if value is None:
            continue

        lights.append({
            "path": str(
                prim.GetPath()
            ),
            "attr": attr,
            "original": float(value),
        })


    print(
        "Controllable lights found:",
        len(lights),
    )

    if len(lights) == 0:

        raise RuntimeError(
            "No controllable USD lights found."
        )


    # ========================================================
    # REMOVE OLD TARGET
    # ========================================================

    if stage.GetPrimAtPath(
        TARGET_PATH
    ).IsValid():

        stage.RemovePrim(
            TARGET_PATH
        )

        await wait_updates(
            5
        )


    # ========================================================
    # CREATE TARGET ONCE
    # ========================================================

    cube = UsdGeom.Cube.Define(
        stage,
        TARGET_PATH,
    )

    cube.CreateSizeAttr(
        1.0
    )


    xform = UsdGeom.Xformable(
        cube.GetPrim()
    )

    xform.ClearXformOpOrder()


    translate_op = (
        xform.AddTranslateOp()
    )


    scale_op = (
        xform.AddScaleOp()
    )


    def u(value_m):

        return (
            value_m
            / meters_per_unit
        )


    thickness_u = u(
        TARGET_THICKNESS_M
    )


    scale_op.Set(
        Gf.Vec3f(
            u(TARGET_WIDTH_M),
            u(TARGET_HEIGHT_M),
            thickness_u,
        )
    )


    cube.CreateDisplayColorAttr(
        [
            Gf.Vec3f(
                0.8,
                0.15,
                0.15,
            )
        ]
    )


    # ========================================================
    # RENDER PRODUCT + ANNOTATORS ONCE
    # ========================================================

    render_product = (
        rep.create.render_product(
            CAMERA_PRIM,
            resolution=(
                WIDTH,
                HEIGHT,
            ),
        )
    )


    rgb_annot = (
        rep.AnnotatorRegistry
        .get_annotator(
            "rgb"
        )
    )


    z_annot = (
        rep.AnnotatorRegistry
        .get_annotator(
            "distance_to_image_plane"
        )
    )


    radial_annot = (
        rep.AnnotatorRegistry
        .get_annotator(
            "distance_to_camera"
        )
    )


    rgb_annot.attach(
        [render_product]
    )

    z_annot.attach(
        [render_product]
    )

    radial_annot.attach(
        [render_product]
    )


    rep.orchestrator.set_capture_on_play(
        False
    )


    await wait_updates(
        30
    )


    rows = []


    # ========================================================
    # RUN 3 LIGHTING x 4 DISTANCE x 3 OFFSET = 36
    # ========================================================

    try:

        sample_index = 0

        for light_name, light_factor in LIGHTING.items():

            # ------------------------------------------------
            # APPLY LIGHTING FROM ORIGINAL BASELINE
            # ------------------------------------------------

            print()
            print(
                "Setting lighting:",
                light_name,
                "factor=",
                light_factor,
            )

            for item in lights:

                item["attr"].Set(
                    item["original"]
                    * light_factor
                )


            await wait_updates(
                25
            )


            for distance_m in DISTANCES_M:

                for offset_name, offset_m in OFFSETS_M.items():

                    sample_index += 1


                    sample_name = (
                        "thr_"
                        f"d{distance_tag(distance_m)}_"
                        f"{light_name}_"
                        f"{offset_name}"
                    )


                    sample_dir = os.path.join(
                        DATA_ROOT,
                        sample_name,
                    )


                    os.makedirs(
                        sample_dir,
                        exist_ok=True,
                    )


                    print()
                    print(
                        f"[{sample_index:02d}/36]",
                        sample_name,
                    )


                    # -----------------------------------------
                    # POSITION TARGET
                    #
                    # Camera local -Z = forward.
                    # Front face stays exactly DISTANCE_M.
                    # -----------------------------------------

                    translate_op.Set(
                        Gf.Vec3d(
                            u(offset_m),
                            0.0,
                            -(
                                u(distance_m)
                                + thickness_u / 2.0
                            ),
                        )
                    )


                    await wait_updates(
                        12
                    )


                    # -----------------------------------------
                    # SYNCHRONIZED CAPTURE
                    # -----------------------------------------

                    await rep.orchestrator.step_async(
                        rt_subframes=8,
                        delta_time=0.0,
                        pause_timeline=False,
                    )


                    await wait_updates(
                        5
                    )


                    # -----------------------------------------
                    # RGB
                    # -----------------------------------------

                    rgb = np.asarray(
                        rgb_annot.get_data()
                    )


                    if (
                        rgb.ndim == 3
                        and rgb.shape[2] >= 3
                    ):

                        rgb = rgb[
                            :,
                            :,
                            :3
                        ]


                    rgb = np.clip(
                        rgb,
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )


                    # -----------------------------------------
                    # GT
                    # -----------------------------------------

                    gt_z = np.asarray(
                        z_annot.get_data(),
                        dtype=np.float32,
                    ).squeeze()


                    gt_radial = np.asarray(
                        radial_annot.get_data(),
                        dtype=np.float32,
                    ).squeeze()


                    # -----------------------------------------
                    # BASIC GT SANITY CHECK
                    #
                    # +/-0.1 m lateral shift still leaves the
                    # center inside this 0.5 m panel.
                    # -----------------------------------------

                    h, w = gt_z.shape

                    cy = h // 2
                    cx = w // 2

                    patch = gt_z[
                        cy-10:cy+11,
                        cx-10:cx+11,
                    ]


                    valid = (
                        np.isfinite(patch)
                        & (patch > 0.01)
                    )


                    if not valid.any():

                        raise RuntimeError(
                            "No valid center GT for "
                            + sample_name
                        )


                    center_gt = float(
                        np.median(
                            patch[
                                valid
                            ]
                        )
                    )


                    placement_error = abs(
                        center_gt
                        - distance_m
                    )


                    if placement_error > 0.05:

                        raise RuntimeError(
                            f"{sample_name}: "
                            f"expected {distance_m:.3f} m, "
                            f"center GT={center_gt:.3f} m"
                        )


                    # -----------------------------------------
                    # SAVE
                    # -----------------------------------------

                    rgb_path = os.path.join(
                        sample_dir,
                        "frame_000001_rgb.png",
                    )

                    gt_path = os.path.join(
                        sample_dir,
                        "frame_000001_gt_z.npy",
                    )

                    radial_path = os.path.join(
                        sample_dir,
                        "frame_000001_gt_radial.npy",
                    )

                    metadata_path = os.path.join(
                        sample_dir,
                        "frame_000001_metadata.json",
                    )


                    Image.fromarray(
                        rgb
                    ).save(
                        rgb_path
                    )


                    np.save(
                        gt_path,
                        gt_z.astype(
                            np.float32
                        ),
                    )


                    np.save(
                        radial_path,
                        gt_radial.astype(
                            np.float32
                        ),
                    )


                    rgb_mean = float(
                        rgb.mean()
                    )

                    rgb_std = float(
                        rgb.std()
                    )


                    metadata = {

                        "sample_name":
                            sample_name,

                        "experiment":
                            "threshold_robustness",

                        "nominal_target_distance_m":
                            distance_m,

                        "target_offset_x_m":
                            offset_m,

                        "offset_condition":
                            offset_name,

                        "lighting_condition":
                            light_name,

                        "lighting_multiplier":
                            light_factor,

                        "controlled_light_count":
                            len(lights),

                        "center_gt_m":
                            center_gt,

                        "gt_placement_error_m":
                            placement_error,

                        "rgb_mean":
                            rgb_mean,

                        "rgb_std":
                            rgb_std,

                        "target_width_m":
                            TARGET_WIDTH_M,

                        "target_height_m":
                            TARGET_HEIGHT_M,

                        "camera_prim":
                            CAMERA_PRIM,

                        "primary_ground_truth":
                            "distance_to_image_plane",

                        "resolution": [
                            WIDTH,
                            HEIGHT,
                        ],
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


                    rows.append({
                        "sample_name":
                            sample_name,

                        "distance_m":
                            distance_m,

                        "lighting":
                            light_name,

                        "lighting_multiplier":
                            light_factor,

                        "offset":
                            offset_name,

                        "offset_x_m":
                            offset_m,

                        "center_gt_m":
                            center_gt,

                        "rgb_mean":
                            rgb_mean,
                    })


                    print(
                        " GT:",
                        f"{center_gt:.3f} m",
                        "| RGB mean:",
                        f"{rgb_mean:.2f}",
                    )


    finally:

        # ====================================================
        # RESTORE ORIGINAL LIGHTING
        # ====================================================

        print()
        print(
            "Restoring original light intensities..."
        )

        for item in lights:

            item["attr"].Set(
                item["original"]
            )


        await wait_updates(
            10
        )


        rgb_annot.detach(
            [render_product]
        )

        z_annot.detach(
            [render_product]
        )

        radial_annot.detach(
            [render_product]
        )


    # ========================================================
    # MANIFEST
    # ========================================================

    with open(
        MANIFEST_PATH,
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
    print("=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)

    print(
        "Samples:",
        len(rows),
    )

    print(
        "Manifest:",
        MANIFEST_PATH,
    )

    print(
        "✅ 36-sample robustness dataset complete"
    )


asyncio.ensure_future(
    run_experiment()
)
