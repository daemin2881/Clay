import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import genesis as gs

from vs_config import (
    BASE_DIR,
    GENERATED_DIR,
    DT,
    SUBSTEPS,
    SHOW_VIEWER,
    MPM_GRID_DENSITY,
    MPM_PARTICLE_SIZE,
    MPM_LOWER_BOUND,
    MPM_UPPER_BOUND,
    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_SIZE_X,
    CLAY_SIZE_Y,
    CLAY_SIZE_Z,
    CLAY_BOTTOM_Z,
    CLAY_TOP_Z,
    MATERIAL,
    TOOL_RADIUS,
    TOOL_SHAFT_RADIUS,
    TOOL_TIP_RADIUS,
    TOOL_LENGTH,
    TOOL_MOUNT_GAP,
    TOOL_LINK_NAME,
    TOOL_TIP_LINK_NAME,
    TOOL_JOINT_NAME,
    TIP_JOINT_NAME,
    HEIGHT_MAP_RES,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    HOVER_HEIGHT,
    PRECONTACT_HEIGHT,
    TRAVEL_STEPS,
    DOWN_STEPS,
    PUSH_STEPS,
    UP_STEPS,
    HOLD_STEPS,
    SETTLE_STEPS,
)


BASE_DIR = Path(BASE_DIR)
GENERATED_DIR = Path(GENERATED_DIR)

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BASE_IIWA_URDF = (
    GENERATED_DIR
    /
    "ssk_iiwa7_genesis.urdf"
)

ROBOT_URDF = (
    GENERATED_DIR
    /
    "ssk_iiwa7_visual_sculpting.urdf"
)


VIS_EVERY = 12

VIEWER_REFRESH_RATE = 20

VIEWER_RESOLUTION = (
    640,
    400,
)


PARTICLE_SURFACE_OFFSET = (
    MPM_PARTICLE_SIZE
    *
    0.5
)


_GENESIS_INITIALIZED = False

_SIM = None

DEVICE = None

FLOAT_DTYPE = None

BOOL_DTYPE = None


# ============================================================
# TOOL URDF
# ============================================================

def create_tool_urdf():

    if not BASE_IIWA_URDF.exists():

        raise FileNotFoundError(
            f"\nBase IIWA URDF not found:\n"
            f"{BASE_IIWA_URDF}\n\n"
            f"Expected:\n"
            f"assets/generated/ssk_iiwa7_genesis.urdf\n"
        )


    tree = ET.parse(
        BASE_IIWA_URDF
    )

    root = tree.getroot()


    # 기존 도구가 있으면 제거
    for element in list(
        root
    ):

        if (
            element.tag == "link"
            and
            element.attrib.get(
                "name"
            )
            in (
                TOOL_LINK_NAME,
                TOOL_TIP_LINK_NAME,
            )
        ):

            root.remove(
                element
            )


        if (
            element.tag == "joint"
            and
            element.attrib.get(
                "name"
            )
            in (
                TOOL_JOINT_NAME,
                TIP_JOINT_NAME,
            )
        ):

            root.remove(
                element
            )


    link_names = [

        link.attrib[
            "name"
        ]

        for link in root.findall(
            "link"
        )

        if "name" in link.attrib
    ]


    candidates = [

        "iiwa_link_ee",
        "iiwa_link_7",
        "iiwa7_link_ee",
        "iiwa7_link_7",
        "link_ee",
        "link_7",
        "tool0",
    ]


    parent_link = next(

        (
            name

            for name in candidates

            if name in link_names
        ),

        link_names[
            -1
        ],
    )


    shaft_length = max(

        TOOL_LENGTH
        -
        TOOL_TIP_RADIUS,

        0.001,
    )


    # ========================================================
    # TOOL SHAFT
    # ========================================================

    tool_link = ET.SubElement(

        root,

        "link",

        {
            "name":
            TOOL_LINK_NAME
        },
    )


    inertial = ET.SubElement(

        tool_link,

        "inertial",
    )


    ET.SubElement(

        inertial,

        "origin",

        {
            "xyz":
            f"0 0 {shaft_length / 2.0}",

            "rpy":
            "0 0 0",
        },
    )


    shaft_mass = 0.08


    ET.SubElement(

        inertial,

        "mass",

        {
            "value":
            str(
                shaft_mass
            )
        },
    )


    ixx = (

        shaft_mass

        *

        (
            3.0
            *
            TOOL_SHAFT_RADIUS
            *
            TOOL_SHAFT_RADIUS

            +

            shaft_length
            *
            shaft_length
        )

        /

        12.0
    )


    izz = (

        0.5

        *

        shaft_mass

        *

        TOOL_SHAFT_RADIUS
        *
        TOOL_SHAFT_RADIUS
    )


    ET.SubElement(

        inertial,

        "inertia",

        {
            "ixx": str(ixx),
            "iyy": str(ixx),
            "izz": str(izz),
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )


    visual = ET.SubElement(

        tool_link,

        "visual",
    )


    ET.SubElement(

        visual,

        "origin",

        {
            "xyz":
            f"0 0 {shaft_length / 2.0}",

            "rpy":
            "0 0 0",
        },
    )


    geometry = ET.SubElement(

        visual,

        "geometry",
    )


    ET.SubElement(

        geometry,

        "cylinder",

        {
            "radius":
            str(
                TOOL_SHAFT_RADIUS
            ),

            "length":
            str(
                shaft_length
            ),
        },
    )


    material = ET.SubElement(

        visual,

        "material",

        {
            "name":
            "visual_sculpting_tool_white"
        },
    )


    ET.SubElement(

        material,

        "color",

        {
            "rgba":
            "0.93 0.93 0.93 1.0"
        },
    )


    collision = ET.SubElement(

        tool_link,

        "collision",
    )


    ET.SubElement(

        collision,

        "origin",

        {
            "xyz":
            f"0 0 {shaft_length / 2.0}",

            "rpy":
            "0 0 0",
        },
    )


    collision_geometry = ET.SubElement(

        collision,

        "geometry",
    )


    ET.SubElement(

        collision_geometry,

        "cylinder",

        {
            "radius":
            str(
                TOOL_SHAFT_RADIUS
            ),

            "length":
            str(
                shaft_length
            ),
        },
    )


    tool_joint = ET.SubElement(

        root,

        "joint",

        {
            "name":
            TOOL_JOINT_NAME,

            "type":
            "fixed",
        },
    )


    ET.SubElement(

        tool_joint,

        "parent",

        {
            "link":
            parent_link
        },
    )


    ET.SubElement(

        tool_joint,

        "child",

        {
            "link":
            TOOL_LINK_NAME
        },
    )


    ET.SubElement(

        tool_joint,

        "origin",

        {
            "xyz":
            f"0 0 {TOOL_MOUNT_GAP}",

            "rpy":
            "0 0 0",
        },
    )


    # ========================================================
    # ROUNDED TOOL TIP
    #
    # TOOL_TIP_LINK frame은 실제 최하단 접촉점이다.
    # ========================================================

    tip_link = ET.SubElement(

        root,

        "link",

        {
            "name":
            TOOL_TIP_LINK_NAME
        },
    )


    tip_inertial = ET.SubElement(

        tip_link,

        "inertial",
    )


    ET.SubElement(

        tip_inertial,

        "origin",

        {
            "xyz":
            f"0 0 {-TOOL_TIP_RADIUS}",

            "rpy":
            "0 0 0",
        },
    )


    tip_mass = 0.02


    ET.SubElement(

        tip_inertial,

        "mass",

        {
            "value":
            str(
                tip_mass
            )
        },
    )


    sphere_inertia = (

        0.4

        *

        tip_mass

        *

        TOOL_TIP_RADIUS
        *
        TOOL_TIP_RADIUS
    )


    ET.SubElement(

        tip_inertial,

        "inertia",

        {
            "ixx":
            str(
                sphere_inertia
            ),

            "iyy":
            str(
                sphere_inertia
            ),

            "izz":
            str(
                sphere_inertia
            ),

            "ixy":
            "0",

            "ixz":
            "0",

            "iyz":
            "0",
        },
    )


    tip_visual = ET.SubElement(

        tip_link,

        "visual",
    )


    ET.SubElement(

        tip_visual,

        "origin",

        {
            "xyz":
            f"0 0 {-TOOL_TIP_RADIUS}",

            "rpy":
            "0 0 0",
        },
    )


    tip_geometry = ET.SubElement(

        tip_visual,

        "geometry",
    )


    ET.SubElement(

        tip_geometry,

        "sphere",

        {
            "radius":
            str(
                TOOL_TIP_RADIUS
            )
        },
    )


    tip_material = ET.SubElement(

        tip_visual,

        "material",

        {
            "name":
            "visual_sculpting_tip_white"
        },
    )


    ET.SubElement(

        tip_material,

        "color",

        {
            "rgba":
            "0.96 0.96 0.96 1.0"
        },
    )


    tip_collision = ET.SubElement(

        tip_link,

        "collision",
    )


    ET.SubElement(

        tip_collision,

        "origin",

        {
            "xyz":
            f"0 0 {-TOOL_TIP_RADIUS}",

            "rpy":
            "0 0 0",
        },
    )


    tip_collision_geometry = ET.SubElement(

        tip_collision,

        "geometry",
    )


    ET.SubElement(

        tip_collision_geometry,

        "sphere",

        {
            "radius":
            str(
                TOOL_TIP_RADIUS
            )
        },
    )


    tip_joint = ET.SubElement(

        root,

        "joint",

        {
            "name":
            TIP_JOINT_NAME,

            "type":
            "fixed",
        },
    )


    ET.SubElement(

        tip_joint,

        "parent",

        {
            "link":
            TOOL_LINK_NAME
        },
    )


    ET.SubElement(

        tip_joint,

        "child",

        {
            "link":
            TOOL_TIP_LINK_NAME
        },
    )


    ET.SubElement(

        tip_joint,

        "origin",

        {
            "xyz":
            f"0 0 {TOOL_LENGTH}",

            "rpy":
            "0 0 0",
        },
    )


    tree.write(

        ROBOT_URDF,

        encoding="utf-8",

        xml_declaration=True,
    )


    print()

    print(
        "==============================================="
    )

    print(
        "ROUNDED END EFFECTOR"
    )

    print(
        "==============================================="
    )

    print(
        "parent link      :",
        parent_link
    )

    print(
        "shaft diameter   :",
        TOOL_SHAFT_RADIUS
        *
        2000.0,
        "mm",
    )

    print(
        "tip diameter     :",
        TOOL_TIP_RADIUS
        *
        2000.0,
        "mm",
    )

    print(
        "total length     :",
        TOOL_LENGTH
        *
        1000.0,
        "mm",
    )

    print(
        "tip frame        : physical contact point"
    )

    print(
        "==============================================="
    )


# ============================================================
# PARTICLES -> 512×512 HEIGHT MAP
# ============================================================

@torch.no_grad()
def particles_to_height_gpu(
    particle_pos,
):

    device = particle_pos.device

    dtype = particle_pos.dtype


    x0 = (

        CLAY_CENTER_X

        -

        CLAY_SIZE_X
        /
        2.0
    )


    x1 = (

        CLAY_CENTER_X

        +

        CLAY_SIZE_X
        /
        2.0
    )


    y0 = (

        CLAY_CENTER_Y

        -

        CLAY_SIZE_Y
        /
        2.0
    )


    y1 = (

        CLAY_CENTER_Y

        +

        CLAY_SIZE_Y
        /
        2.0
    )


    x = particle_pos[
        :,
        0
    ]


    y = particle_pos[
        :,
        1
    ]


    z_center = particle_pos[
        :,
        2
    ]


    valid = (

        (x >= x0)

        &

        (x <= x1)

        &

        (y >= y0)

        &

        (y <= y1)

        &

        torch.isfinite(
            x
        )

        &

        torch.isfinite(
            y
        )

        &

        torch.isfinite(
            z_center
        )
    )


    x = x[
        valid
    ]


    y = y[
        valid
    ]


    z_center = z_center[
        valid
    ]


    if x.numel() == 0:

        return torch.full(

            (
                HEIGHT_MAP_RES,
                HEIGHT_MAP_RES,
            ),

            CLAY_TOP_Z,

            device=device,

            dtype=dtype,
        )


    pixel_size_x = (

        CLAY_SIZE_X

        /

        (
            HEIGHT_MAP_RES
            -
            1
        )
    )


    pixel_size_y = (

        CLAY_SIZE_Y

        /

        (
            HEIGHT_MAP_RES
            -
            1
        )
    )


    pixel_size = min(

        pixel_size_x,

        pixel_size_y,
    )


    support_radius = (

        MPM_PARTICLE_SIZE

        *

        1.1
    )


    support_radius_sq = (

        support_radius

        *

        support_radius
    )


    particle_top_offset = (

        MPM_PARTICLE_SIZE

        *

        0.5
    )


    radial_drop = (

        MPM_PARTICLE_SIZE

        *

        0.18
    )


    radius_px = int(

        math.ceil(

            support_radius

            /

            pixel_size
        )
    )


    radius_px = max(

        1,

        radius_px,
    )


    fx = (

        (
            x
            -
            x0
        )

        /

        (
            x1
            -
            x0
        )

        *

        (
            HEIGHT_MAP_RES
            -
            1
        )
    )


    fy = (

        (
            y
            -
            y0
        )

        /

        (
            y1
            -
            y0
        )

        *

        (
            HEIGHT_MAP_RES
            -
            1
        )
    )


    base_x = torch.floor(
        fx
    ).long()


    base_y = torch.floor(
        fy
    ).long()


    num_pixels = (

        HEIGHT_MAP_RES

        *

        HEIGHT_MAP_RES
    )


    height_flat = torch.full(

        (
            num_pixels,
        ),

        -torch.inf,

        device=device,

        dtype=dtype,
    )


    for oy in range(

        -radius_px,

        radius_px
        +
        1,
    ):

        iy = (

            base_y

            +

            oy
        )


        valid_y = (

            (iy >= 0)

            &

            (
                iy
                <
                HEIGHT_MAP_RES
            )
        )


        for ox in range(

            -radius_px,

            radius_px
            +
            1,
        ):

            ix = (

                base_x

                +

                ox
            )


            valid_xy = (

                valid_y

                &

                (ix >= 0)

                &

                (
                    ix
                    <
                    HEIGHT_MAP_RES
                )
            )


            if not bool(
                valid_xy.any()
            ):

                continue


            ix_v = ix[
                valid_xy
            ]


            iy_v = iy[
                valid_xy
            ]


            x_v = x[
                valid_xy
            ]


            y_v = y[
                valid_xy
            ]


            z_v = z_center[
                valid_xy
            ]


            px = (

                x0

                +

                ix_v.to(
                    dtype
                )

                *

                pixel_size_x
            )


            py = (

                y0

                +

                iy_v.to(
                    dtype
                )

                *

                pixel_size_y
            )


            dx = (

                px

                -

                x_v
            )


            dy = (

                py

                -

                y_v
            )


            dist_sq = (

                dx
                *
                dx

                +

                dy
                *
                dy
            )


            inside = (

                dist_sq

                <=

                support_radius_sq
            )


            if not bool(
                inside.any()
            ):

                continue


            ix_i = ix_v[
                inside
            ]


            iy_i = iy_v[
                inside
            ]


            z_i = z_v[
                inside
            ]


            d2_i = dist_sq[
                inside
            ]


            normalized_d2 = (

                d2_i

                /

                support_radius_sq
            ).clamp(

                0.0,

                1.0,
            )


            candidate_height = (

                z_i

                +

                particle_top_offset

                -

                radial_drop

                *

                normalized_d2
            )


            flat_idx = (

                iy_i

                *

                HEIGHT_MAP_RES

                +

                ix_i
            )


            height_flat.scatter_reduce_(

                0,

                flat_idx,

                candidate_height,

                reduce="amax",

                include_self=True,
            )


    height_map = height_flat.view(

        HEIGHT_MAP_RES,

        HEIGHT_MAP_RES,
    )


    known = torch.isfinite(
        height_map
    )


    known_float = known.to(
        dtype
    )


    height_map = torch.where(

        known,

        height_map,

        torch.zeros_like(
            height_map
        ),
    )


    kernel = torch.ones(

        (
            1,
            1,
            3,
            3,
        ),

        device=device,

        dtype=dtype,
    )


    for _ in range(
        4
    ):

        missing = (

            known_float

            <

            0.5
        )


        value_sum = F.conv2d(

            (
                height_map
                *
                known_float
            )[
                None,
                None,
                :,
                :,
            ],

            kernel,

            padding=1,

        )[
            0,
            0,
        ]


        count = F.conv2d(

            known_float[
                None,
                None,
                :,
                :,
            ],

            kernel,

            padding=1,

        )[
            0,
            0,
        ]


        can_fill = (

            missing

            &

            (
                count

                >

                0.0
            )
        )


        interpolated = (

            value_sum

            /

            count.clamp_min(
                1.0
            )
        )


        height_map = torch.where(

            can_fill,

            interpolated,

            height_map,
        )


        known_float = torch.where(

            can_fill,

            torch.ones_like(
                known_float
            ),

            known_float,
        )


    remaining = (

        known_float

        <

        0.5
    )


    valid_count = known_float.sum().clamp_min(
        1.0
    )


    mean_height = (

        (
            height_map
            *
            known_float
        ).sum()

        /

        valid_count
    )


    height_map = torch.where(

        remaining,

        mean_height.expand_as(
            height_map
        ),

        height_map,
    )


    smoothed = F.avg_pool2d(

        height_map[
            None,
            None,
            :,
            :,
        ],

        kernel_size=3,

        stride=1,

        padding=1,

        count_include_pad=False,

    )[
        0,
        0,
    ]


    height_map = (

        0.9
        *
        height_map

        +

        0.1
        *
        smoothed
    )


    height_map = torch.nan_to_num(

        height_map,

        nan=CLAY_TOP_Z,

        posinf=(
            CLAY_TOP_Z
            +
            0.03
        ),

        neginf=CLAY_BOTTOM_Z,
    )


    border_px = 4


    interior = height_map[

        border_px:
        -border_px,

        border_px:
        -border_px,
    ]


    height_map = F.pad(

        interior[
            None,
            None,
            :,
            :,
        ],

        (
            border_px,
            border_px,
            border_px,
            border_px,
        ),

        mode="replicate",

    )[
        0,
        0,
    ]


    return height_map


# ============================================================
# VISUAL SCULPTING SIMULATOR
# ============================================================

class VisualSculptingSimulator:

    def __init__(
        self,
    ):

        global _GENESIS_INITIALIZED

        global DEVICE

        global FLOAT_DTYPE

        global BOOL_DTYPE


        if not _GENESIS_INITIALIZED:

            gs.init(

                backend=gs.gpu,

                precision="32",

                performance_mode=True,
            )


            _GENESIS_INITIALIZED = True


        self.device = gs.device

        self.dtype = gs.tc_float


        DEVICE = self.device

        FLOAT_DTYPE = self.dtype

        BOOL_DTYPE = gs.tc_bool


        create_tool_urdf()


        self.scene = gs.Scene(

            sim_options=gs.options.SimOptions(

                dt=DT,

                substeps=SUBSTEPS,

                gravity=(
                    0.0,
                    0.0,
                    -9.81,
                ),
            ),

            mpm_options=gs.options.MPMOptions(

                lower_bound=MPM_LOWER_BOUND,

                upper_bound=MPM_UPPER_BOUND,

                grid_density=MPM_GRID_DENSITY,

                particle_size=MPM_PARTICLE_SIZE,

                enable_CPIC=False,
            ),

            coupler_options=(
                gs.options.LegacyCouplerOptions(

                    rigid_mpm=True,
                )
            ),

            viewer_options=gs.options.ViewerOptions(

                res=VIEWER_RESOLUTION,

                refresh_rate=VIEWER_REFRESH_RATE,

                realtime_factor=None,

                camera_pos=(
                    0.92,
                    -1.0,
                    0.78,
                ),

                camera_lookat=(
                    CLAY_CENTER_X,
                    0.0,
                    0.16,
                ),

                camera_fov=42,
            ),

            vis_options=gs.options.VisOptions(

                visualize_mpm_boundary=False,

                shadow=False,

                plane_reflection=False,

                background_color=(
                    0.025,
                    0.030,
                    0.040,
                ),

                ambient_light=(
                    0.055,
                    0.055,
                    0.060,
                ),

                lights=[
                    gs.options.vis.DirectionalLight(
                        dir=(
                            0.65,
                            -0.75,
                            -1.0,
                        ),
                        color=(
                            1.0,
                            0.92,
                            0.82,
                        ),
                        intensity=3.5,
                    ),

                    gs.options.vis.DirectionalLight(
                        dir=(
                            -0.45,
                            0.65,
                            -1.0,
                        ),
                        color=(
                            0.70,
                            0.80,
                            1.0,
                        ),
                        intensity=1.0,
                    ),
                ],
            ),

            show_viewer=SHOW_VIEWER,
        )


        self.floor = self.scene.add_entity(

            morph=gs.morphs.Plane(),

            material=gs.materials.Rigid(

                needs_coup=True,

                coup_friction=1.6,

                coup_softness=0.0015,

                coup_restitution=0.0,
            ),
        )


        self.robot = self.scene.add_entity(

            morph=gs.morphs.URDF(

                file=str(
                    ROBOT_URDF
                ),

                pos=(
                    0.0,
                    0.0,
                    0.0,
                ),

                fixed=True,

                merge_fixed_links=False,
            ),

            material=gs.materials.Rigid(

                needs_coup=True,

                coup_friction=0.55,

                coup_softness=0.0015,

                coup_restitution=0.0,

                sdf_cell_size=0.006,

                sdf_min_res=16,

                sdf_max_res=48,
            ),
        )


        self.clay = self.scene.add_entity(

            morph=gs.morphs.Box(

                pos=(
                    CLAY_CENTER_X,

                    CLAY_CENTER_Y,

                    CLAY_BOTTOM_Z
                    +
                    CLAY_SIZE_Z
                    /
                    2.0,
                ),

                size=(
                    CLAY_SIZE_X,
                    CLAY_SIZE_Y,
                    CLAY_SIZE_Z,
                ),
            ),

            material=gs.materials.MPM.ElastoPlastic(

                E=MATERIAL[
                    "E"
                ],

                nu=MATERIAL[
                    "nu"
                ],

                rho=MATERIAL[
                    "rho"
                ],

                use_von_mises=True,

                von_mises_yield_stress=(
                    MATERIAL[
                        "yield_stress"
                    ]
                ),
            ),

            surface=gs.surfaces.Default(

                color=(
                    0.82,
                    0.82,
                    0.78,
                    1.0,
                ),

                vis_mode="recon",
            ),
        )


        print()

        print(
            "==============================================="
        )

        print(
            "VISUAL SCULPTING SIMULATOR"
        )

        print(
            "==============================================="
        )

        print(
            "device           :",
            self.device
        )

        print(
            "material         :",
            MATERIAL
        )

        print(
            "grid density     :",
            MPM_GRID_DENSITY
        )

        print(
            "particle size    :",
            MPM_PARTICLE_SIZE
        )

        print(
            "dt               :",
            DT
        )

        print(
            "substeps         :",
            SUBSTEPS
        )

        print(
            "substep dt       :",
            DT
            /
            SUBSTEPS
        )

        print(
            "tip diameter     :",
            TOOL_TIP_RADIUS
            *
            2000,
            "mm",
        )

        print(
            "==============================================="
        )


        self.scene.build()


        self.tip = self.robot.get_link(

            TOOL_TIP_LINK_NAME
        )


        dof_indices = []


        for joint in self.robot.joints:

            try:

                dof_indices.extend(

                    list(
                        joint.dofs_idx_local
                    )
                )


            except Exception:

                pass


        self.dof_idx = np.asarray(

            list(

                dict.fromkeys(

                    dof_indices
                )
            )[
                :7
            ],

            dtype=np.int32,
        )


        if len(
            self.dof_idx
        ) < 7:

            raise RuntimeError(

                "Could not identify "
                "7 IIWA arm DOFs."
            )


        self.initial_q = torch.tensor(

            [
                0.0,
                0.55,
                0.0,
                -1.15,
                0.0,
                1.2,
                0.0,
            ],

            device=self.device,

            dtype=self.dtype,
        )


        self.tool_quat_down = torch.tensor(

            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],

            device=self.device,

            dtype=self.dtype,
        )


        self.last_q = self.initial_q.clone()


        self.robot.set_dofs_position(

            self.initial_q,

            dofs_idx_local=self.dof_idx,

            zero_velocity=True,
        )


        self.robot.set_dofs_kp(
            torch.tensor(
                [
                    5000.0,
                    5000.0,
                    4500.0,
                    4000.0,
                    3000.0,
                    2200.0,
                    1600.0,
                ],
                device=self.device,
                dtype=self.dtype,
            ),
            dofs_idx_local=self.dof_idx,
        )

        self.robot.set_dofs_kv(
            torch.tensor(
                [
                    500.0,
                    500.0,
                    450.0,
                    400.0,
                    300.0,
                    220.0,
                    160.0,
                ],
                device=self.device,
                dtype=self.dtype,
            ),
            dofs_idx_local=self.dof_idx,
        )


        self.visual_counter = 0

        self.viewer_alive = SHOW_VIEWER

        self.initialized = False


    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def step(
        self,
        force_visual=False,
    ):

        self.visual_counter += 1


        show = (

            self.viewer_alive

            and

            (
                force_visual

                or

                self.visual_counter
                %
                VIS_EVERY
                ==
                0
            )
        )


        try:

            self.scene.step(

                update_visualizer=show,

                refresh_visualizer=show,
            )


        except Exception as exc:

            message = str(
                exc
            )


            if "Viewer closed" in message:

                self.viewer_alive = False

                print(
                    "[Viewer closed]"
                )

                raise SystemExit(
                    0
                )


            raise


    # ========================================================
    # INVERSE KINEMATICS
    # ========================================================

    def solve_ik(
        self,
        position,
    ):

        position = torch.as_tensor(

            position,

            device=self.device,

            dtype=self.dtype,
        )


        q = self.robot.inverse_kinematics(

            link=self.tip,

            pos=position,

            quat=self.tool_quat_down,

            init_qpos=self.last_q,

            respect_joint_limit=True,

            max_samples=1,

            max_solver_iters=40,

            damping=0.08,

            pos_tol=0.0015,

            rot_tol=0.008,

            max_step_size=0.10,

            dofs_idx_local=self.dof_idx,
        )


        if not torch.is_tensor(
            q
        ):

            q = torch.as_tensor(

                q,

                device=self.device,

                dtype=self.dtype,
            )


        if q.ndim > 1:

            q = q[
                0
            ]


        q = q[
            :7
        ]


        if not torch.isfinite(
            q
        ).all():

            raise RuntimeError(

                "IK returned "
                "non-finite q."
            )


        self.last_q = (

            q
            .detach()
            .clone()
        )


        return q


    # ========================================================
    # JOINT MOVE
    # ========================================================

    def move_q(
        self,
        q_target,
        steps,
    ):

        q_start = self.robot.get_dofs_position(

            dofs_idx_local=self.dof_idx
        )


        if q_start.ndim > 1:

            q_start = q_start[
                0
            ]


        steps = max(

            1,

            int(
                steps
            ),
        )


        for i in range(
            steps
        ):

            u = (

                i
                +
                1

            ) / steps


            u = (

                u
                *
                u
                *
                (
                    3.0

                    -

                    2.0
                    *
                    u
                )
            )


            q = torch.lerp(

                q_start,

                q_target,

                float(
                    u
                ),
            )


            self.robot.control_dofs_position(

                position=q,

                dofs_idx_local=self.dof_idx,
            )


            self.step()


        self.last_q = (

            q_target
            .detach()
            .clone()
        )


        return q_target


    # ========================================================
    # MOVE TIP
    # ========================================================

    def move_tip(
        self,
        target,
        steps,
    ):

        q_target = self.solve_ik(

            target
        )


        return self.move_q(

            q_target,

            steps,
        )


    # ========================================================
    # LINEAR CARTESIAN MOVE
    # ========================================================

    def linear_tip(
        self,
        start,
        end,
        steps,
    ):

        start = torch.as_tensor(

            start,

            device=self.device,

            dtype=self.dtype,
        )


        end = torch.as_tensor(

            end,

            device=self.device,

            dtype=self.dtype,
        )


        steps = max(

            1,

            int(
                steps
            ),
        )


        last_q = None


        for i in range(
            steps
        ):

            u = (

                i
                +
                1

            ) / steps


            position = torch.lerp(

                start,

                end,

                float(
                    u
                ),
            )


            q = self.solve_ik(

                position
            )


            self.robot.control_dofs_position(

                position=q,

                dofs_idx_local=self.dof_idx,
            )


            self.step()


            last_q = q


        return last_q


    # ========================================================
    # INITIALIZE SCENE
    # ========================================================

    def initialize_scene(
        self,
    ):

        if self.initialized:

            return


        print()

        print(
            "Initial clay settling..."
        )


        for _ in range(
            8
        ):

            self.step()


        print(
            "Setting upright robot pose..."
        )


        for _ in range(
            10
        ):

            self.robot.control_dofs_position(

                position=self.initial_q,

                dofs_idx_local=self.dof_idx,
            )


            self.step()


        home_tip = torch.tensor(

            [
                CLAY_CENTER_X,
                CLAY_CENTER_Y,
                CLAY_TOP_Z
                +
                HOVER_HEIGHT,
            ],

            device=self.device,

            dtype=self.dtype,
        )


        print(
            "Moving rounded tool above clay..."
        )


        self.move_tip(

            home_tip,

            10,
        )


        self.step(

            force_visual=True
        )


        self.initialized = True


    # ========================================================
    # HEIGHT MAP
    # ========================================================

    @torch.no_grad()
    def get_height_gpu(
        self,
    ):

        particles = self.clay.get_particles_pos()


        return particles_to_height_gpu(

            particles
        )


    # ========================================================
    # SAMPLE HEIGHT MAP
    # ========================================================

    @torch.no_grad()
    def sample_surface_height_gpu(
        self,
        height_map,
        x,
        y,
    ):

        x = torch.as_tensor(

            x,

            device=self.device,

            dtype=self.dtype,
        )


        y = torch.as_tensor(

            y,

            device=self.device,

            dtype=self.dtype,
        )


        gx = (

            2.0

            *

            (
                x

                -

                (
                    CLAY_CENTER_X

                    -

                    CLAY_SIZE_X
                    /
                    2.0
                )
            )

            /

            CLAY_SIZE_X

            -

            1.0
        )


        gy = (

            2.0

            *

            (
                y

                -

                (
                    CLAY_CENTER_Y

                    -

                    CLAY_SIZE_Y
                    /
                    2.0
                )
            )

            /

            CLAY_SIZE_Y

            -

            1.0
        )


        grid = torch.stack(

            [
                gx,
                gy,
            ],

            dim=-1,
        )


        grid = grid.reshape(

            1,
            1,
            -1,
            2,
        )


        sampled = F.grid_sample(

            height_map[
                None,
                None,
                :,
                :,
            ],

            grid,

            mode="bilinear",

            padding_mode="border",

            align_corners=True,
        )


        return sampled[

            0,
            0,
            0,
            :
        ]


    # ========================================================
    # VISUAL SCULPTING PUSH TRAJECTORY
    #
    # a = [x, y, theta, length, depth]
    # ========================================================

    @torch.no_grad()
    def build_push_trajectory_gpu(
        self,
        action,
        height_map,
    ):

        action = torch.as_tensor(

            action,

            device=self.device,

            dtype=self.dtype,
        )


        if action.numel() != 5:

            raise ValueError(

                "Visual Sculpting push action must be "
                "[x, y, theta, length, depth]. "

                f"Got shape={tuple(action.shape)} "
                f"numel={action.numel()}."
            )


        x0 = action[
            0
        ]


        y0 = action[
            1
        ]


        theta = action[
            2
        ]


        length = action[
            3
        ]


        depth = action[
            4
        ]


        x0 = torch.clamp(

            x0,

            X_MIN,

            X_MAX,
        )


        y0 = torch.clamp(

            y0,

            Y_MIN,

            Y_MAX,
        )


        length = torch.clamp(

            length,

            min=0.0,
        )


        depth = torch.clamp(

            depth,

            min=0.0,
        )


        t = torch.linspace(

            0.0,

            1.0,

            PUSH_STEPS
            +
            1,

            device=self.device,

            dtype=self.dtype,
        )


        xs = (

            x0

            +

            t
            *
            length
            *
            torch.cos(
                theta
            )
        )


        ys = (

            y0

            +

            t
            *
            length
            *
            torch.sin(
                theta
            )
        )


        xs = torch.clamp(

            xs,

            X_MIN,

            X_MAX,
        )


        ys = torch.clamp(

            ys,

            Y_MIN,

            Y_MAX,
        )


        surface_z = self.sample_surface_height_gpu(

            height_map,

            xs,

            ys,
        )


        zs = (

            surface_z

            +

            PRECONTACT_HEIGHT

            *

            (
                1.0
                -
                t
            )

            -

            depth

            *

            t
        )


        trajectory = torch.stack(

            [
                xs,
                ys,
                zs,
            ],

            dim=1,
        )


        return trajectory


    # ========================================================
    # VISUAL SCULPTING LINEAR PUSH
    #
    # 경로 생성 방식은 기존 Visual Sculpting 방식 유지.
    # 각 목표 위치를 여러 step 유지해 추종 시간을 준다.
    # ========================================================

    def linear_push(
        self,
        action,
        action_index=None,
    ):

        if not self.initialized:

            self.initialize_scene()


        action = torch.as_tensor(

            action,

            device=self.device,

            dtype=self.dtype,
        )


        if action.numel() != 5:

            raise ValueError(

                "linear_push requires action "
                "[x, y, theta, length, depth]. "

                f"Got numel={action.numel()}."
            )


        height_map = self.get_height_gpu()


        trajectory = self.build_push_trajectory_gpu(

            action,

            height_map,
        )


        start = trajectory[
            0
        ]


        end = trajectory[
            -1
        ]


        surface_path = self.sample_surface_height_gpu(

            height_map,

            trajectory[
                :,
                0
            ],

            trajectory[
                :,
                1
            ],
        )


        max_surface = torch.max(

            surface_path
        )


        nominal_hover = torch.tensor(

            CLAY_TOP_Z

            +

            HOVER_HEIGHT,

            device=self.device,

            dtype=self.dtype,
        )


        local_hover = (

            max_surface

            +

            HOVER_HEIGHT
        )


        hover_z = torch.maximum(

            nominal_hover,

            local_hover,
        )


        hover_start = torch.stack(

            [
                start[
                    0
                ],

                start[
                    1
                ],

                hover_z,
            ]
        )


        hover_end = torch.stack(

            [
                end[
                    0
                ],

                end[
                    1
                ],

                hover_z,
            ]
        )


        if action_index is not None:

            x_v = float(

                action[
                    0
                ]
                .detach()
                .cpu()
            )


            y_v = float(

                action[
                    1
                ]
                .detach()
                .cpu()
            )


            theta_v = float(

                action[
                    2
                ]
                .detach()
                .cpu()
            )


            length_v = float(

                action[
                    3
                ]
                .detach()
                .cpu()
            )


            depth_v = float(

                action[
                    4
                ]
                .detach()
                .cpu()
            )


            print(

                f"[push {action_index:04d}] "

                f"x={x_v:.3f} "

                f"y={y_v:.3f} "

                f"theta={theta_v:+.2f} "

                f"length="
                f"{length_v * 1000.0:.1f}mm "

                f"depth="
                f"{depth_v * 1000.0:.1f}mm"
            )


        def print_tip_error(
            label,
            target,
        ):

            actual = (

                self.tip
                .get_pos()
                .detach()
                .reshape(
                    -1
                )[
                    :3
                ]
            )


            target = (

                target
                .detach()
                .reshape(
                    -1
                )[
                    :3
                ]
            )


            error_mm = (

                actual

                -

                target

            ) * 1000.0


            print(

                f"[{label}] "

                f"target=("
                f"{target[0].item():.4f}, "
                f"{target[1].item():.4f}, "
                f"{target[2].item():.4f}"
                f") m | "

                f"actual=("
                f"{actual[0].item():.4f}, "
                f"{actual[1].item():.4f}, "
                f"{actual[2].item():.4f}"
                f") m | "

                f"error=("
                f"{error_mm[0].item():+.1f}, "
                f"{error_mm[1].item():+.1f}, "
                f"{error_mm[2].item():+.1f}"
                f") mm"
            )


        # 1. 시작점 위로 이동
        self.move_tip(

            hover_start,

            TRAVEL_STEPS,
        )


        print_tip_error(

            "HOVER CHECK",

            hover_start,
        )


        # 2. 점토 표면으로 접근
        self.linear_tip(

            hover_start,

            trajectory[
                0
            ],

            DOWN_STEPS,
        )


        print_tip_error(

            "APPROACH CHECK",

            trajectory[
                0
            ],
        )


        # 3. 기존 Visual Sculpting 직선 push 경로 실행
        q_end = None


        for i in range(

            1,

            trajectory.shape[
                0
            ],
        ):

            position = trajectory[
                i
            ]


            q = self.solve_ik(

                position
            )


            # 각 지점의 관절 목표를 50 step 동안 유지
            for _ in range(
                50
            ):

                self.robot.control_dofs_position(

                    position=q,

                    dofs_idx_local=self.dof_idx,
                )


                self.step()


            print_tip_error(

                f"TIP CHECK {i:02d}",

                position,
            )


            q_end = q


        # 4. 마지막 위치 유지
        if q_end is not None:

            for _ in range(
                HOLD_STEPS
            ):

                self.robot.control_dofs_position(

                    position=q_end,

                    dofs_idx_local=self.dof_idx,
                )


                self.step()


        print_tip_error(

            "END HOLD CHECK",

            trajectory[
                -1
            ],
        )


        # 5. 끝 위치에서 수직으로 들어 올리기
        q_up = self.linear_tip(

            trajectory[
                -1
            ],

            hover_end,

            UP_STEPS,
        )


        print_tip_error(

            "LIFT CHECK",

            hover_end,
        )


        # 6. 짧게 안정화
        if q_up is not None:

            for _ in range(
                SETTLE_STEPS
            ):

                self.robot.control_dofs_position(

                    position=q_up,

                    dofs_idx_local=self.dof_idx,
                )


                self.step()


    # ========================================================
    # HOLD
    # ========================================================

    def hold(
        self,
        steps=1,
    ):

        q = self.robot.get_dofs_position(

            dofs_idx_local=self.dof_idx
        )


        if q.ndim > 1:

            q = q[
                0
            ]


        for _ in range(

            int(
                steps
            )
        ):

            self.robot.control_dofs_position(

                position=q,

                dofs_idx_local=self.dof_idx,
            )


            self.step()


# ============================================================
# MODULE API
# ============================================================

def initialize():

    global _SIM


    if _SIM is None:

        _SIM = VisualSculptingSimulator()


    _SIM.initialize_scene()


    return _SIM


def get_height_gpu():

    simulator = initialize()


    return simulator.get_height_gpu()


def sample_surface_height_gpu(
    height_map,
    x,
    y,
):

    simulator = initialize()


    return simulator.sample_surface_height_gpu(

        height_map,

        x,

        y,
    )


def build_push_trajectory_gpu(
    action,
    height_map,
):

    simulator = initialize()


    return simulator.build_push_trajectory_gpu(

        action,

        height_map,
    )


def linear_push(
    action,
    action_index=None,
):

    simulator = initialize()


    return simulator.linear_push(

        action,

        action_index=action_index,
    )


def hold(
    steps=1,
):

    simulator = initialize()


    return simulator.hold(
        steps
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    simulator = initialize()


    print()

    print(
        "==============================================="
    )

    print(
        "SIMULATOR READY"
    )

    print(
        "==============================================="
    )

    print(
        "DEVICE       =",
        DEVICE
    )

    print(
        "FLOAT_DTYPE  =",
        FLOAT_DTYPE
    )

    print(
        "BOOL_DTYPE   =",
        BOOL_DTYPE
    )

    print(
        "ACTION       = "
        "[x, y, theta, length, depth]"
    )


    height = simulator.get_height_gpu()


    print(
        "height shape =",
        height.shape
    )

    print(
        "height device=",
        height.device
    )

    print(
        "height min   =",
        float(
            height.min()
        )
    )

    print(
        "height max   =",
        float(
            height.max()
        )
    )

    print(
        "Rounded 30 mm Visual Sculpting "
        "push tool ready."
    )


    if SHOW_VIEWER:

        print(
            "Viewer test running."
        )


        test_action = torch.tensor(

            [
                0.44,
                0.0,
                0.0,
                0.03,
                0.005,
            ],

            device=DEVICE,

            dtype=FLOAT_DTYPE,
        )


        linear_push(

            test_action,

            action_index=0,
        )


        while True:

            simulator.hold(
                1
            )