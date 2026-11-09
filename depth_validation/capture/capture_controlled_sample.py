import os
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
# ONLY CHANGE THESE TWO VALUES FOR EACH TEST
# ============================================================

DISTANCE_M = 1.6
SAMPLE_NAME = "threshold_1pm"


# ============================================================
# FIXED EXPERIMENT SETTINGS
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
    "depth_validation/data"
)

OUT_DIR = os.path.join(
    DATA_ROOT,
    SAMPLE_NAME,
)


async def run_sample():

    print()
    print("=" * 60)
    print("VROSA CONTROLLED DEPTH VALIDATION")
    print("=" * 60)

    print("Sample       :", SAMPLE_NAME)
    print("Distance     :", DISTANCE_M, "m")
    print("Camera       :", CAMERA_PRIM)
    print("Target size  :",
          TARGET_WIDTH_M,
          "x",
          TARGET_HEIGHT_M,
          "m")

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # GET STAGE
    # ========================================================

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
        "Meters/unit  :",
        meters_per_unit,
    )

    camera_prim = stage.GetPrimAtPath(
        CAMERA_PRIM
    )

    if not camera_prim.IsValid():
        raise RuntimeError(
            "Camera does not exist: "
            + CAMERA_PRIM
        )


    # ========================================================
    # REMOVE PREVIOUS VALIDATION TARGET
    # ========================================================

    old_target = stage.GetPrimAtPath(
        TARGET_PATH
    )

    if old_target.IsValid():

        print(
            "Removing old target..."
        )

        stage.RemovePrim(
            TARGET_PATH
        )

        for _ in range(3):
            await (
                omni.kit.app
                .get_app()
                .next_update_async()
            )


    # ========================================================
    # CONVERT METERS TO STAGE UNITS
    # ========================================================

    def meters_to_units(value):

        return (
            value
            / meters_per_unit
        )


    distance_u = meters_to_units(
        DISTANCE_M
    )

    width_u = meters_to_units(
        TARGET_WIDTH_M
    )

    height_u = meters_to_units(
        TARGET_HEIGHT_M
    )

    thickness_u = meters_to_units(
        TARGET_THICKNESS_M
    )


    # ========================================================
    # CREATE CONTROLLED TARGET
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


    # USD camera looks along local -Z.
    #
    # Cube CENTER is placed slightly farther than requested
    # distance so its FRONT FACE lies exactly at DISTANCE_M.
    translate_op = (
        xform.AddTranslateOp()
    )

    translate_op.Set(
        Gf.Vec3d(
            0.0,
            0.0,
            -(
                distance_u
                + thickness_u / 2.0
            ),
        )
    )


    scale_op = (
        xform.AddScaleOp()
    )

    scale_op.Set(
        Gf.Vec3f(
            width_u,
            height_u,
            thickness_u,
        )
    )


    # Simple visible target color.
    cube.CreateDisplayColorAttr(
        [
            Gf.Vec3f(
                0.8,
                0.15,
                0.15,
            )
        ]
    )


    print()
    print(
        "✅ Target created at",
        DISTANCE_M,
        "m"
    )


    # ========================================================
    # WAIT FOR SCENE TO UPDATE
    # ========================================================

    print(
        "Waiting for scene/render update..."
    )

    for _ in range(30):

        await (
            omni.kit.app
            .get_app()
            .next_update_async()
        )


    # ========================================================
    # CREATE RENDER PRODUCT
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


    # ========================================================
    # ANNOTATORS
    # ========================================================

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


    # ========================================================
    # ANNOTATOR WARMUP
    # ========================================================

    print(
        "Warming annotators..."
    )

    for _ in range(30):

        await (
            omni.kit.app
            .get_app()
            .next_update_async()
        )


    # ========================================================
    # SYNCHRONIZED CAPTURE
    # ========================================================

    print(
        "Capturing RGB + GT..."
    )

    await rep.orchestrator.step_async(
        rt_subframes=8,
        delta_time=0.0,
        pause_timeline=False,
    )


    for _ in range(5):

        await (
            omni.kit.app
            .get_app()
            .next_update_async()
        )


    # ========================================================
    # GET RGB
    # ========================================================

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


    # ========================================================
    # GET GT DEPTH
    # ========================================================

    gt_z = np.asarray(
        z_annot.get_data(),
        dtype=np.float32,
    ).squeeze()


    gt_radial = np.asarray(
        radial_annot.get_data(),
        dtype=np.float32,
    ).squeeze()


    # ========================================================
    # VALIDATION
    # ========================================================

    valid_z = (
        np.isfinite(gt_z)
        & (gt_z > 0.01)
        & (gt_z < 1000.0)
    )


    valid_radial = (
        np.isfinite(gt_radial)
        & (gt_radial > 0.01)
        & (gt_radial < 1000.0)
    )


    # ========================================================
    # CENTER TARGET REGION
    # ========================================================

    h, w = gt_z.shape

    cy = h // 2
    cx = w // 2

    half = 10

    center_gt = gt_z[
        cy-half:cy+half+1,
        cx-half:cx+half+1,
    ]


    center_mask = (
        np.isfinite(center_gt)
        & (center_gt > 0.01)
    )


    if center_mask.any():

        center_gt_median = float(
            np.median(
                center_gt[
                    center_mask
                ]
            )
        )

    else:

        center_gt_median = None


    print()
    print("=" * 60)
    print("GT CHECK")
    print("=" * 60)

    print(
        "GT valid pixels:",
        int(valid_z.sum()),
        "/",
        gt_z.size,
    )

    print(
        "Expected target:",
        DISTANCE_M,
        "m",
    )

    print(
        "Center 21x21 GT:",
        center_gt_median,
        "m",
    )


    # ========================================================
    # IMPORTANT AUTOMATIC SANITY CHECK
    # ========================================================

    if center_gt_median is None:

        raise RuntimeError(
            "No valid GT depth "
            "at target center."
        )


    gt_error = abs(
        center_gt_median
        - DISTANCE_M
    )


    print(
        "GT placement error:",
        gt_error,
        "m",
    )


    if gt_error > 0.05:

        raise RuntimeError(
            "Controlled target placement "
            "failed. Expected "
            f"{DISTANCE_M:.3f} m but "
            f"GT center is "
            f"{center_gt_median:.3f} m."
        )


    print(
        "✅ GT target distance verified"
    )


    # ========================================================
    # SAVE FILES
    # ========================================================

    rgb_path = os.path.join(
        OUT_DIR,
        "frame_000001_rgb.png",
    )

    z_path = os.path.join(
        OUT_DIR,
        "frame_000001_gt_z.npy",
    )

    radial_path = os.path.join(
        OUT_DIR,
        "frame_000001_gt_radial.npy",
    )

    metadata_path = os.path.join(
        OUT_DIR,
        "frame_000001_metadata.json",
    )


    Image.fromarray(
        rgb
    ).save(
        rgb_path
    )


    np.save(
        z_path,
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


    metadata = {

        "sample_name":
            SAMPLE_NAME,

        "experiment_type":
            "controlled_range_test",

        "camera_prim":
            CAMERA_PRIM,

        "resolution": [
            WIDTH,
            HEIGHT,
        ],

        "nominal_target_distance_m":
            DISTANCE_M,

        "measured_center_gt_m":
            center_gt_median,

        "target_width_m":
            TARGET_WIDTH_M,

        "target_height_m":
            TARGET_HEIGHT_M,

        "target_thickness_m":
            TARGET_THICKNESS_M,

        "target_position":
            "camera_optical_axis",

        "target_surface":
            "plain_red",

        "lighting_condition":
            "baseline",

        "primary_ground_truth":
            "distance_to_image_plane",

        "secondary_ground_truth":
            "distance_to_camera",

        "gt_z_valid_pixels":
            int(
                valid_z.sum()
            ),

        "gt_z_valid_fraction":
            float(
                valid_z.mean()
            ),

        "gt_radial_valid_pixels":
            int(
                valid_radial.sum()
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


    # ========================================================
    # CLEANUP ANNOTATORS
    # ========================================================

    rgb_annot.detach(
        [render_product]
    )

    z_annot.detach(
        [render_product]
    )

    radial_annot.detach(
        [render_product]
    )


    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print(rgb_path)
    print(z_path)
    print(radial_path)
    print(metadata_path)

    print()
    print(
        "✅ CONTROLLED SAMPLE COMPLETE"
    )


asyncio.ensure_future(
    run_sample()
)
