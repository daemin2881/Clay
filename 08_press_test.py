import os
import time
import csv
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
import genesis as gs

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# 0. PURPOSE
# ============================================================
#
# GPU-OPTIMIZED VISUAL SCULPTING BASELINE
#
# 목표:
#
#   Heart Target Height Map
#              ↓
#   Current MPM Clay Particles
#              ↓
#   GPU Height Map
#              ↓
#   GPU Height + Gradient Error
#              ↓
#   GPU Action Selection
#              ↓
#   (x, y, press depth)
#              ↓
#   Upright Vertical Stamp
#              ↓
#   Repeat
#
# 핵심:
#
# - MPM physics            : GPU
# - particle state         : GPU
# - height rasterization   : GPU
# - hole filling           : GPU
# - target map             : GPU
# - gradient               : GPU
# - error map              : GPU
# - tool footprint scoring : GPU
# - action argmax          : GPU
#
# CPU sync는:
# - 최종 action 몇 개 scalar
# - PNG/CSV 저장
# 정도만 허용.
#
# ============================================================


# ============================================================
# 1. PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GENERATED_DIR = os.path.join(
    SCRIPT_DIR,
    "assets",
    "generated",
)

BASE_IIWA_URDF = os.path.join(
    GENERATED_DIR,
    "ssk_iiwa7_genesis.urdf",
)

ROBOT_URDF = os.path.join(
    GENERATED_DIR,
    "ssk_iiwa7_gpu_heart.urdf",
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "gpu_heart_output",
)

STATE_DIR = os.path.join(
    OUTPUT_DIR,
    "state",
)

ACTION_DIR = os.path.join(
    OUTPUT_DIR,
    "action",
)

os.makedirs(
    GENERATED_DIR,
    exist_ok=True,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    STATE_DIR,
    exist_ok=True,
)

os.makedirs(
    ACTION_DIR,
    exist_ok=True,
)

if not os.path.isfile(BASE_IIWA_URDF):
    raise FileNotFoundError(
        f"\nBase iiwa URDF not found:\n"
        f"{BASE_IIWA_URDF}\n"
    )


# ============================================================
# 2. PERFORMANCE PRESET
# ============================================================
#
# 화면을 보고 싶으면 True.
#
# 대량 dataset 생성 시:
#
# SHOW_VIEWER = False
#
# 로 바꾸면 더 빨라짐.
#
# ============================================================

SHOW_VIEWER = True

DT = 0.004

SUBSTEPS = 8

VIS_EVERY = 12

VIEWER_REFRESH_RATE = 20

VIEWER_RESOLUTION = (
    640,
    400,
)

_visual_counter = 0


# ============================================================
# 3. MPM FAST / BALANCED
# ============================================================
#
# 개발용 균형값.
#
# 더 빠르게:
# particle_size = 0.008
#
# 더 정밀:
# grid_density = 80
# particle_size = 0.0045
#
# ============================================================

MPM_GRID_DENSITY = 64

MPM_PARTICLE_SIZE = 0.006


# ============================================================
# 4. BIG CLAY SLAB
# ============================================================
#
# 36 cm × 36 cm
#
# ============================================================

CLAY_CENTER_X = 0.44

CLAY_CENTER_Y = 0.0

CLAY_WIDTH = 0.36

CLAY_LENGTH = 0.36

CLAY_HEIGHT = 0.045

CLAY_BOTTOM_Z = 0.0

CLAY_TOP_Z = (
    CLAY_BOTTOM_Z
    +
    CLAY_HEIGHT
)


# ============================================================
# 5. CLAY MATERIAL
# ============================================================
#
# Genesis MPM ElastoPlastic
#
# ============================================================

CLAY_E = 1.2e4

CLAY_NU = 0.30

CLAY_RHO = 1200.0

CLAY_YIELD_STRESS = 180.0


# ============================================================
# 6. END EFFECTOR
# ============================================================

TOOL_RADIUS = 0.018

TOOL_LENGTH = 0.080

TOOL_MOUNT_GAP = 0.0015

TOOL_LINK_NAME = (
    "visual_sculpting_tool"
)

TOOL_TIP_LINK_NAME = (
    "visual_sculpting_tip"
)

TOOL_JOINT_NAME = (
    "visual_sculpting_tool_joint"
)

TIP_JOINT_NAME = (
    "visual_sculpting_tip_joint"
)


# ============================================================
# 7. HEIGHT MAP
# ============================================================

HEIGHT_MAP_RES = 128

# 점토 가장자리는 로봇 workspace 때문에 사용하지 않음.
ACTIVE_MARGIN = 0.060

PARTICLE_SURFACE_OFFSET = (
    MPM_PARTICLE_SIZE
    *
    0.5
)


# ============================================================
# 8. HEART TARGET
# ============================================================
#
# 하트 모양으로 약 8 mm 움푹 파기.
#
# HEART_SCALE을 크게 하면
# 하트가 커짐.
#
# ============================================================

HEART_SCALE_X = 0.095

HEART_SCALE_Y = 0.095

HEART_DEPTH = 0.008

# 하트 edge smoothing
HEART_BLUR_KERNEL_1 = 7

HEART_BLUR_KERNEL_2 = 5


# ============================================================
# 9. PLANNER
# ============================================================

MAX_ACTIONS = 80

# score가 1.2 mm 이하이면 종료
ERROR_THRESHOLD = 0.0012

PRESS_GAIN = 0.90

MIN_PRESS_DEPTH = 0.0015

MAX_PRESS_DEPTH = 0.0070

# shape 경계도 맞추기 위해 gradient 사용
GRADIENT_WEIGHT = 0.20


# ============================================================
# 10. OUTPUT / SAVE
# ============================================================
#
# 저장할 때는 GPU -> CPU copy가 필요하므로
# 너무 자주 하지 않음.
#
# ============================================================

SAVE_EVERY = 10

SAVE_PNG = True


# ============================================================
# 11. MOTION
# ============================================================

HOVER_HEIGHT = 0.060

PRECONTACT_HEIGHT = 0.004

TRAVEL_STEPS = 8

DOWN_STEPS = 6

PRESS_STEPS = 8

UP_STEPS = 7

STAMP_HOLD_STEPS = 3

STAMP_SETTLE_STEPS = 2


# ============================================================
# 12. CREATE TOOL URDF
# ============================================================

def make_robot_with_tool():

    tree = ET.parse(
        BASE_IIWA_URDF
    )

    root = tree.getroot()

    # --------------------------------------------------------
    # remove old generated tool
    # --------------------------------------------------------

    for element in list(root):

        if (
            element.tag == "link"
            and
            element.attrib.get("name")
            in (
                TOOL_LINK_NAME,
                TOOL_TIP_LINK_NAME,
            )
        ):
            root.remove(element)

        if (
            element.tag == "joint"
            and
            element.attrib.get("name")
            in (
                TOOL_JOINT_NAME,
                TIP_JOINT_NAME,
            )
        ):
            root.remove(element)

    # --------------------------------------------------------
    # end link
    # --------------------------------------------------------

    link_names = [
        link.attrib["name"]
        for link in root.findall("link")
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

    parent_link = None

    for name in candidates:

        if name in link_names:

            parent_link = name

            break

    if parent_link is None:

        parent_link = (
            link_names[-1]
        )

    print()
    print(
        "==============================================="
    )
    print(
        "END EFFECTOR"
    )
    print(
        "==============================================="
    )
    print(
        "parent   :",
        parent_link,
    )
    print(
        "diameter :",
        TOOL_RADIUS * 2000,
        "mm",
    )
    print(
        "length   :",
        TOOL_LENGTH * 1000,
        "mm",
    )
    print(
        "==============================================="
    )

    # --------------------------------------------------------
    # TOOL LINK
    # --------------------------------------------------------

    tool_link = ET.SubElement(
        root,
        "link",
        {
            "name":
            TOOL_LINK_NAME
        },
    )

    # --------------------------------------------------------
    # inertial
    # --------------------------------------------------------

    inertial = ET.SubElement(
        tool_link,
        "inertial",
    )

    ET.SubElement(
        inertial,
        "origin",
        {
            "xyz":
            f"0 0 {TOOL_LENGTH / 2.0}",
            "rpy":
            "0 0 0",
        },
    )

    mass = 0.10

    ET.SubElement(
        inertial,
        "mass",
        {
            "value":
            str(mass)
        },
    )

    radius = TOOL_RADIUS

    length = TOOL_LENGTH

    ixx = (
        mass
        *
        (
            3.0
            *
            radius
            *
            radius
            +
            length
            *
            length
        )
        /
        12.0
    )

    izz = (
        0.5
        *
        mass
        *
        radius
        *
        radius
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

    # --------------------------------------------------------
    # visual
    # --------------------------------------------------------

    visual = ET.SubElement(
        tool_link,
        "visual",
    )

    ET.SubElement(
        visual,
        "origin",
        {
            "xyz":
            f"0 0 {TOOL_LENGTH / 2.0}",
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
            str(TOOL_RADIUS),
            "length":
            str(TOOL_LENGTH),
        },
    )

    material = ET.SubElement(
        visual,
        "material",
        {
            "name":
            "tool_white"
        },
    )

    ET.SubElement(
        material,
        "color",
        {
            "rgba":
            "0.94 0.94 0.92 1.0"
        },
    )

    # --------------------------------------------------------
    # collision
    # --------------------------------------------------------

    collision = ET.SubElement(
        tool_link,
        "collision",
    )

    ET.SubElement(
        collision,
        "origin",
        {
            "xyz":
            f"0 0 {TOOL_LENGTH / 2.0}",
            "rpy":
            "0 0 0",
        },
    )

    collision_geometry = (
        ET.SubElement(
            collision,
            "geometry",
        )
    )

    ET.SubElement(
        collision_geometry,
        "cylinder",
        {
            "radius":
            str(TOOL_RADIUS),
            "length":
            str(TOOL_LENGTH),
        },
    )

    # --------------------------------------------------------
    # tool joint
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # tip
    # --------------------------------------------------------

    ET.SubElement(
        root,
        "link",
        {
            "name":
            TOOL_TIP_LINK_NAME
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


make_robot_with_tool()


# ============================================================
# 13. GENESIS GPU INIT
# ============================================================

gs.init(
    backend=gs.gpu,
    precision="32",
    performance_mode=True,
)


# ============================================================
# 14. TORCH / DEVICE
# ============================================================

DEVICE = gs.device

FLOAT_DTYPE = gs.tc_float

BOOL_DTYPE = gs.tc_bool


print()
print(
    "==============================================="
)
print(
    "GPU CONFIGURATION"
)
print(
    "==============================================="
)
print(
    "Genesis backend :",
    gs.backend,
)
print(
    "Torch device    :",
    DEVICE,
)
print(
    "Float dtype     :",
    FLOAT_DTYPE,
)
print(
    "==============================================="
)


# ============================================================
# 15. SCENE
# ============================================================

scene = gs.Scene(

    sim_options=(
        gs.options.SimOptions(

            dt=DT,

            substeps=SUBSTEPS,

            gravity=(
                0.0,
                0.0,
                -9.81,
            ),
        )
    ),

    # --------------------------------------------------------
    # MPM DOMAIN
    #
    # 큰 36cm 점토판 +
    # MPM safety padding 고려
    # --------------------------------------------------------

    mpm_options=(
        gs.options.MPMOptions(

            lower_bound=(
                CLAY_CENTER_X - 0.28,
                -0.28,
                -0.08,
            ),

            upper_bound=(
                CLAY_CENTER_X + 0.28,
                0.28,
                0.20,
            ),

            grid_density=(
                MPM_GRID_DENSITY
            ),

            particle_size=(
                MPM_PARTICLE_SIZE
            ),

            enable_CPIC=False,
        )
    ),

    coupler_options=(
        gs.options.LegacyCouplerOptions(
            rigid_mpm=True,
        )
    ),

    viewer_options=(
        gs.options.ViewerOptions(

            res=VIEWER_RESOLUTION,

            refresh_rate=(
                VIEWER_REFRESH_RATE
            ),

            # viewer 때문에
            # realtime pacing하지 않음.
            realtime_factor=None,

            camera_pos=(
                0.92,
                -1.00,
                0.78,
            ),

            camera_lookat=(
                0.44,
                0.0,
                0.16,
            ),

            camera_fov=42,
        )
    ),

    vis_options=(
        gs.options.VisOptions(

            visualize_mpm_boundary=False,

            shadow=False,

            plane_reflection=False,
        )
    ),

    show_viewer=SHOW_VIEWER,
)


# ============================================================
# 16. FLOOR
# ============================================================

floor = scene.add_entity(

    morph=gs.morphs.Plane(),

    material=(
        gs.materials.Rigid(

            needs_coup=True,

            coup_friction=1.6,

            coup_softness=0.0015,

            coup_restitution=0.0,
        )
    ),
)


# ============================================================
# 17. ROBOT
# ============================================================

iiwa = scene.add_entity(

    morph=(
        gs.morphs.URDF(

            file=ROBOT_URDF,

            pos=(
                0.0,
                0.0,
                0.0,
            ),

            fixed=True,

            merge_fixed_links=False,
        )
    ),

    material=(
        gs.materials.Rigid(

            needs_coup=True,

            coup_friction=0.55,

            coup_softness=0.0015,

            coup_restitution=0.0,

            sdf_cell_size=0.006,

            sdf_min_res=16,

            sdf_max_res=48,
        )
    ),
)


# ============================================================
# 18. CLAY
# ============================================================

clay = scene.add_entity(

    morph=(
        gs.morphs.Box(

            pos=(
                CLAY_CENTER_X,
                CLAY_CENTER_Y,
                CLAY_BOTTOM_Z
                +
                CLAY_HEIGHT / 2.0,
            ),

            size=(
                CLAY_WIDTH,
                CLAY_LENGTH,
                CLAY_HEIGHT,
            ),
        )
    ),

    material=(
        gs.materials.MPM.ElastoPlastic(

            E=CLAY_E,

            nu=CLAY_NU,

            rho=CLAY_RHO,

            use_von_mises=True,

            von_mises_yield_stress=(
                CLAY_YIELD_STRESS
            ),
        )
    ),

    surface=(
        gs.surfaces.Default(

            color=(
                0.82,
                0.70,
                0.53,
                1.0,
            ),

            vis_mode="recon",
        )
    ),
)


# ============================================================
# 19. BUILD
# ============================================================

print()
print(
    "==============================================="
)
print(
    "GPU HEART VISUAL SCULPTING"
)
print(
    "==============================================="
)
print(
    "Clay:",
    CLAY_WIDTH,
    "x",
    CLAY_LENGTH,
    "m",
)
print(
    "Map:",
    HEIGHT_MAP_RES,
    "x",
    HEIGHT_MAP_RES,
)
print(
    "Heart depth:",
    HEART_DEPTH * 1000,
    "mm",
)
print(
    "Grid:",
    MPM_GRID_DENSITY,
)
print(
    "Particle:",
    MPM_PARTICLE_SIZE,
)
print(
    "Substeps:",
    SUBSTEPS,
)
print(
    "Max actions:",
    MAX_ACTIONS,
)
print(
    "==============================================="
)

scene.build()


# ============================================================
# 20. TOOL TIP
# ============================================================

tool_tip_link = iiwa.get_link(
    TOOL_TIP_LINK_NAME
)

if tool_tip_link is None:

    raise RuntimeError(
        "visual_sculpting_tip not found."
    )


# ============================================================
# 21. ARM DOFS
# ============================================================
#
# iiwa = 7 actuated joints.
# fixed tool joints add no DOFs.
#
# controller hot-path에서는 slice 사용.
#
# ============================================================

ARM_DOF_COUNT = 7

DOF_SLICE = slice(
    0,
    ARM_DOF_COUNT,
)

DOF_IK_IDX = list(
    range(
        ARM_DOF_COUNT
    )
)


# ============================================================
# 22. ROBOT POSES - GPU
# ============================================================

INITIAL_Q = torch.tensor(

    [
        0.0,
        0.55,
        0.0,
        -1.15,
        0.0,
        1.20,
        0.0,
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


IK_SEEDS = [

    INITIAL_Q.clone(),

    torch.tensor(
        [
            0.0,
            0.45,
            0.0,
            -1.05,
            0.0,
            1.15,
            0.0,
        ],
        device=DEVICE,
        dtype=FLOAT_DTYPE,
    ),

    torch.tensor(
        [
            0.0,
            0.65,
            0.0,
            -1.25,
            0.0,
            1.30,
            0.0,
        ],
        device=DEVICE,
        dtype=FLOAT_DTYPE,
    ),
]


_last_valid_ik_q = (
    INITIAL_Q.clone()
)


# ============================================================
# 23. UPRIGHT CONSTRAINT
# ============================================================

UPRIGHT_Q2_MIN = 0.05

UPRIGHT_Q2_MAX = 1.60

UPRIGHT_Q4_MIN = -2.60

UPRIGHT_Q4_MAX = -0.25

UPRIGHT_Q6_MIN = 0.25

UPRIGHT_Q6_MAX = 2.60

MAX_SEED_DISTANCE = 2.8


UPRIGHT_WEIGHTS = torch.tensor(

    [
        0.35,
        1.00,
        0.35,
        1.20,
        0.30,
        1.00,
        0.20,
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


# ============================================================
# 24. TOOL ORIENTATION
# ============================================================

TOOL_QUAT_DOWN = torch.tensor(

    [
        0.0,
        1.0,
        0.0,
        0.0,
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


# ============================================================
# 25. ROBOT CONTROLLER
# ============================================================

iiwa.set_dofs_position(

    INITIAL_Q,

    dofs_idx_local=DOF_SLICE,

    zero_velocity=True,
)


iiwa.set_dofs_kp(

    torch.tensor(

        [
            1000,
            1000,
            950,
            850,
            500,
            350,
            250,
        ],

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    ),

    dofs_idx_local=DOF_SLICE,
)


iiwa.set_dofs_kv(

    torch.tensor(

        [
            100,
            100,
            95,
            85,
            50,
            35,
            25,
        ],

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    ),

    dofs_idx_local=DOF_SLICE,
)


# ============================================================
# 26. SIM STEP
# ============================================================

def sim_step(
    force_visual=False,
):

    global _visual_counter

    _visual_counter += 1

    if not SHOW_VIEWER:

        scene.step(
            update_visualizer=False,
            refresh_visualizer=False,
        )

        return

    visual = (

        force_visual

        or

        (
            _visual_counter
            %
            VIS_EVERY
            ==
            0
        )
    )

    scene.step(
        update_visualizer=visual,
        refresh_visualizer=visual,
    )


# ============================================================
# 27. SMOOTHSTEP
# ============================================================

def smoothstep_scalar(t):

    t = max(
        0.0,
        min(
            1.0,
            float(t),
        )
    )

    return (
        t
        *
        t
        *
        (
            3.0
            -
            2.0 * t
        )
    )


# ============================================================
# 28. GET Q - GPU
# ============================================================

def get_q_gpu():

    q = iiwa.get_dofs_position(
        dofs_idx_local=DOF_SLICE
    )

    if q.ndim > 1:

        q = q[0]

    return q


# ============================================================
# 29. IK HELPERS
# ============================================================

def normalize_ik_tensor(
    q_solution,
):

    if not torch.is_tensor(
        q_solution
    ):

        q_solution = torch.as_tensor(

            q_solution,

            device=DEVICE,

            dtype=FLOAT_DTYPE,
        )

    else:

        q_solution = q_solution.to(

            device=DEVICE,

            dtype=FLOAT_DTYPE,
        )

    if q_solution.ndim > 1:

        q_solution = (
            q_solution[0]
        )

    if q_solution.shape[0] > 7:

        q_solution = (
            q_solution[:7]
        )

    if q_solution.shape[0] != 7:

        raise RuntimeError(
            "Unexpected IK shape: "
            f"{tuple(q_solution.shape)}"
        )

    return q_solution


def upright_pose_score_gpu(q):

    delta = (

        q
        -
        INITIAL_Q

    ) * UPRIGHT_WEIGHTS

    return torch.linalg.vector_norm(
        delta
    )


def valid_upright_pose_gpu(q):

    finite = torch.all(
        torch.isfinite(q)
    )

    shoulder = (
        (q[1] >= UPRIGHT_Q2_MIN)
        &
        (q[1] <= UPRIGHT_Q2_MAX)
    )

    elbow = (
        (q[3] >= UPRIGHT_Q4_MIN)
        &
        (q[3] <= UPRIGHT_Q4_MAX)
    )

    wrist = (
        (q[5] >= UPRIGHT_Q6_MIN)
        &
        (q[5] <= UPRIGHT_Q6_MAX)
    )

    distance_ok = (
        upright_pose_score_gpu(q)
        <=
        MAX_SEED_DISTANCE
    )

    result = (
        finite
        &
        shoulder
        &
        elbow
        &
        wrist
        &
        distance_ok
    )

    # IK branch 판단용 scalar 한 개만 sync.
    return bool(
        result.item()
    )


def solve_tip_ik_gpu(
    target,
):

    global _last_valid_ik_q

    if not torch.is_tensor(
        target
    ):

        target = torch.tensor(
            target,
            device=DEVICE,
            dtype=FLOAT_DTYPE,
        )

    else:

        target = target.to(
            device=DEVICE,
            dtype=FLOAT_DTYPE,
        )

    seed_candidates = [

        _last_valid_ik_q,

        IK_SEEDS[0],

        IK_SEEDS[1],

        IK_SEEDS[2],
    ]

    best_q = None

    best_score = None

    for seed_index, seed_q in enumerate(
        seed_candidates
    ):

        try:

            q_solution = (
                iiwa.inverse_kinematics(

                    link=tool_tip_link,

                    pos=target,

                    quat=TOOL_QUAT_DOWN,

                    init_qpos=seed_q,

                    respect_joint_limit=True,

                    max_samples=1,

                    max_solver_iters=40,

                    damping=0.08,

                    pos_tol=0.0015,

                    rot_tol=0.008,

                    max_step_size=0.10,

                    dofs_idx_local=DOF_IK_IDX,
                )
            )

            q_solution = (
                normalize_ik_tensor(
                    q_solution
                )
            )

            if not valid_upright_pose_gpu(
                q_solution
            ):

                continue

            if seed_index == 0:

                _last_valid_ik_q = (
                    q_solution.detach().clone()
                )

                return q_solution

            score = (
                upright_pose_score_gpu(
                    q_solution
                )
            )

            score_value = float(
                score.item()
            )

            if (
                best_score is None
                or
                score_value
                <
                best_score
            ):

                best_score = (
                    score_value
                )

                best_q = (
                    q_solution
                )

        except Exception:

            pass

    if best_q is None:

        target_cpu = (
            target
            .detach()
            .cpu()
            .tolist()
        )

        raise RuntimeError(
            "\nNo upright IK solution.\n"
            f"Target: {target_cpu}\n"
        )

    _last_valid_ik_q = (
        best_q.detach().clone()
    )

    return best_q


# ============================================================
# 30. ROBOT MOVE - TORCH
# ============================================================

def move_q_gpu(
    q_target,
    steps,
):

    q_start = get_q_gpu()

    q_target = q_target.to(
        device=DEVICE,
        dtype=FLOAT_DTYPE,
    )

    steps = max(
        1,
        int(steps),
    )

    for i in range(
        steps
    ):

        alpha = smoothstep_scalar(
            (i + 1)
            /
            steps
        )

        q_cmd = torch.lerp(
            q_start,
            q_target,
            alpha,
        )

        iiwa.control_dofs_position(

            position=q_cmd,

            dofs_idx_local=DOF_SLICE,
        )

        sim_step()

    return q_target


def move_tip_air_gpu(
    target,
    steps,
):

    q_target = (
        solve_tip_ik_gpu(
            target
        )
    )

    return move_q_gpu(
        q_target,
        steps,
    )


def move_tip_linear_gpu(
    start_xyz,
    end_xyz,
    steps,
):

    start_xyz = torch.as_tensor(

        start_xyz,

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )

    end_xyz = torch.as_tensor(

        end_xyz,

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )

    steps = max(
        1,
        int(steps),
    )

    q_last = None

    for i in range(
        steps
    ):

        alpha = smoothstep_scalar(
            (i + 1)
            /
            steps
        )

        target_xyz = torch.lerp(
            start_xyz,
            end_xyz,
            alpha,
        )

        q_target = (
            solve_tip_ik_gpu(
                target_xyz
            )
        )

        iiwa.control_dofs_position(

            position=q_target,

            dofs_idx_local=DOF_SLICE,
        )

        sim_step()

        q_last = q_target

    return q_last


# ============================================================
# 31. HEIGHT MAP CONSTANTS
# ============================================================

PIXEL_SIZE_X = (
    CLAY_WIDTH
    /
    (
        HEIGHT_MAP_RES
        -
        1
    )
)

PIXEL_SIZE_Y = (
    CLAY_LENGTH
    /
    (
        HEIGHT_MAP_RES
        -
        1
    )
)

X_MIN = (
    CLAY_CENTER_X
    -
    CLAY_WIDTH / 2.0
)

Y_MIN = (
    CLAY_CENTER_Y
    -
    CLAY_LENGTH / 2.0
)


# ============================================================
# 32. GPU GRID
# ============================================================

X_VALUES = torch.linspace(

    CLAY_CENTER_X
    -
    CLAY_WIDTH / 2.0,

    CLAY_CENTER_X
    +
    CLAY_WIDTH / 2.0,

    HEIGHT_MAP_RES,

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


Y_VALUES = torch.linspace(

    CLAY_CENTER_Y
    -
    CLAY_LENGTH / 2.0,

    CLAY_CENTER_Y
    +
    CLAY_LENGTH / 2.0,

    HEIGHT_MAP_RES,

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


GRID_Y, GRID_X = torch.meshgrid(

    Y_VALUES,

    X_VALUES,

    indexing="ij",
)


# ============================================================
# 33. ACTIVE WORKSPACE MASK
# ============================================================

ACTIVE_MASK = (

    (
        GRID_X
        >=
        CLAY_CENTER_X
        -
        CLAY_WIDTH / 2.0
        +
        ACTIVE_MARGIN
    )

    &

    (
        GRID_X
        <=
        CLAY_CENTER_X
        +
        CLAY_WIDTH / 2.0
        -
        ACTIVE_MARGIN
    )

    &

    (
        GRID_Y
        >=
        CLAY_CENTER_Y
        -
        CLAY_LENGTH / 2.0
        +
        ACTIVE_MARGIN
    )

    &

    (
        GRID_Y
        <=
        CLAY_CENTER_Y
        +
        CLAY_LENGTH / 2.0
        -
        ACTIVE_MARGIN
    )
)


# ============================================================
# 34. PARTICLE RASTER FOOTPRINT
# ============================================================

PARTICLE_RADIUS_PX = max(

    1,

    int(
        np.ceil(
            (
                MPM_PARTICLE_SIZE
                *
                0.60
            )
            /
            min(
                PIXEL_SIZE_X,
                PIXEL_SIZE_Y,
            )
        )
    )
)


spread_offsets = []

for dr in range(
    -PARTICLE_RADIUS_PX,
    PARTICLE_RADIUS_PX + 1,
):

    for dc in range(
        -PARTICLE_RADIUS_PX,
        PARTICLE_RADIUS_PX + 1,
    ):

        if (
            dr * dr
            +
            dc * dc
            <=
            PARTICLE_RADIUS_PX
            *
            PARTICLE_RADIUS_PX
        ):

            spread_offsets.append(
                (
                    dr,
                    dc,
                )
            )


SPREAD_OFFSETS = torch.tensor(

    spread_offsets,

    device=DEVICE,

    dtype=torch.long,
)


# ============================================================
# 35. HEART TARGET - GPU
# ============================================================
#
# implicit heart:
#
# (x² + y² - 1)³ - x² y³ <= 0
#
# ============================================================

def make_heart_target_gpu():

    hx = (
        GRID_X
        -
        CLAY_CENTER_X
    ) / HEART_SCALE_X

    hy = (
        GRID_Y
        -
        CLAY_CENTER_Y
    ) / HEART_SCALE_Y

    # 하트의 뾰족한 부분이 아래로 가도록
    hy = -hy

    heart_equation = (

        (
            hx * hx
            +
            hy * hy
            -
            1.0
        )
        ** 3

        -

        (
            hx
            *
            hx
            *
            hy
            *
            hy
            *
            hy
        )
    )

    heart_mask = (
        heart_equation
        <=
        0.0
    ).to(
        dtype=FLOAT_DTYPE
    )

    # --------------------------------------------------------
    # soft boundary
    # --------------------------------------------------------

    heart_soft = F.avg_pool2d(

        heart_mask[
            None,
            None
        ],

        kernel_size=(
            HEART_BLUR_KERNEL_1
        ),

        stride=1,

        padding=(
            HEART_BLUR_KERNEL_1
            //
            2
        ),
    )[
        0,
        0
    ]

    heart_soft = F.avg_pool2d(

        heart_soft[
            None,
            None
        ],

        kernel_size=(
            HEART_BLUR_KERNEL_2
        ),

        stride=1,

        padding=(
            HEART_BLUR_KERNEL_2
            //
            2
        ),
    )[
        0,
        0
    ]

    heart_soft = torch.clamp(
        heart_soft,
        0.0,
        1.0,
    )

    target = (

        torch.full(

            (
                HEIGHT_MAP_RES,
                HEIGHT_MAP_RES,
            ),

            CLAY_TOP_Z,

            device=DEVICE,

            dtype=FLOAT_DTYPE,
        )

        -

        HEART_DEPTH
        *
        heart_soft
    )

    return torch.clamp(

        target,

        min=CLAY_BOTTOM_Z,

        max=CLAY_TOP_Z,
    )


TARGET_HEIGHT = (
    make_heart_target_gpu()
)


# ============================================================
# 36. GPU SOBEL GRADIENT
# ============================================================
#
# torch.gradient도 GPU지만
# kernel을 미리 만들어 재사용.
#
# ============================================================

SOBEL_X = torch.tensor(

    [
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
).unsqueeze(0)


SOBEL_Y = torch.tensor(

    [
        [
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ]
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
).unsqueeze(0)


SOBEL_X = (
    SOBEL_X
    /
    (
        8.0
        *
        PIXEL_SIZE_X
    )
)


SOBEL_Y = (
    SOBEL_Y
    /
    (
        8.0
        *
        PIXEL_SIZE_Y
    )
)


def gradient_gpu(
    height_map,
):

    h = height_map[
        None,
        None
    ]

    gx = F.conv2d(
        h,
        SOBEL_X,
        padding=1,
    )[
        0,
        0
    ]

    gy = F.conv2d(
        h,
        SOBEL_Y,
        padding=1,
    )[
        0,
        0
    ]

    return gx, gy


TARGET_GRAD_X, TARGET_GRAD_Y = (
    gradient_gpu(
        TARGET_HEIGHT
    )
)


# ============================================================
# 37. CIRCULAR TOOL KERNEL - GPU
# ============================================================

TOOL_RADIUS_PX = max(

    1,

    int(
        round(
            TOOL_RADIUS
            /
            min(
                PIXEL_SIZE_X,
                PIXEL_SIZE_Y,
            )
        )
    )
)


TOOL_KERNEL_SIZE = (
    2
    *
    TOOL_RADIUS_PX
    +
    1
)


coords = torch.arange(

    -TOOL_RADIUS_PX,

    TOOL_RADIUS_PX + 1,

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


kernel_y, kernel_x = torch.meshgrid(

    coords,

    coords,

    indexing="ij",
)


tool_kernel = (

    (
        kernel_x
        *
        kernel_x

        +

        kernel_y
        *
        kernel_y
    )

    <=

    (
        TOOL_RADIUS_PX
        *
        TOOL_RADIUS_PX
    )
).to(
    FLOAT_DTYPE
)


tool_kernel = (
    tool_kernel
    /
    torch.clamp(
        tool_kernel.sum(),
        min=1.0,
    )
)


TOOL_KERNEL = (
    tool_kernel[
        None,
        None
    ]
)


# ============================================================
# 38. GPU HEIGHT MAP
# ============================================================

@torch.no_grad()
def get_current_height_map_gpu():

    # --------------------------------------------------------
    # Genesis API returns torch.Tensor
    # on gs.device.
    #
    # NO .cpu()
    # NO .numpy()
    # --------------------------------------------------------

    particles = (
        clay.get_particles_pos()
    )

    if particles.ndim == 3:

        particles = (
            particles[0]
        )

    px = particles[:, 0]

    py = particles[:, 1]

    pz = (
        particles[:, 2]

        +

        PARTICLE_SURFACE_OFFSET
    )

    # --------------------------------------------------------
    # world -> pixel
    # --------------------------------------------------------

    cols = torch.round(

        (
            px
            -
            X_MIN
        )

        /
        PIXEL_SIZE_X

    ).to(
        torch.long
    )


    rows = torch.round(

        (
            py
            -
            Y_MIN
        )

        /
        PIXEL_SIZE_Y

    ).to(
        torch.long
    )

    # --------------------------------------------------------
    # spread footprint
    # --------------------------------------------------------

    rr = (

        rows[:, None]

        +

        SPREAD_OFFSETS[
            None,
            :,
            0
        ]
    )


    cc = (

        cols[:, None]

        +

        SPREAD_OFFSETS[
            None,
            :,
            1
        ]
    )


    valid = (

        (rr >= 0)

        &

        (
            rr
            <
            HEIGHT_MAP_RES
        )

        &

        (cc >= 0)

        &

        (
            cc
            <
            HEIGHT_MAP_RES
        )
    )


    flat_indices = (

        rr
        *
        HEIGHT_MAP_RES

        +

        cc
    )


    z_values = (
        pz[:, None]
        .expand_as(
            flat_indices
        )
    )


    flat_indices = (
        flat_indices[
            valid
        ]
    )


    z_values = (
        z_values[
            valid
        ]
    )

    # --------------------------------------------------------
    # GPU z-buffer
    # --------------------------------------------------------

    height_flat = torch.full(

        (
            HEIGHT_MAP_RES
            *
            HEIGHT_MAP_RES,
        ),

        -torch.inf,

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )


    height_flat.scatter_reduce_(

        dim=0,

        index=flat_indices,

        src=z_values,

        reduce="amax",

        include_self=True,
    )


    height = (
        height_flat.view(

            HEIGHT_MAP_RES,

            HEIGHT_MAP_RES,
        )
    )

    # --------------------------------------------------------
    # GPU hole fill
    #
    # fixed passes:
    # Python sync 없음
    # --------------------------------------------------------

    h = height[
        None,
        None
    ]


    for _ in range(6):

        pooled = (
            F.max_pool2d(

                h,

                kernel_size=3,

                stride=1,

                padding=1,
            )
        )

        missing = (
            ~torch.isfinite(h)
        )

        fillable = (

            missing

            &

            torch.isfinite(
                pooled
            )
        )

        h = torch.where(

            fillable,

            pooled,

            h,
        )


    height = (
        h[
            0,
            0
        ]
    )


    height = torch.where(

        torch.isfinite(
            height
        ),

        height,

        torch.full_like(

            height,

            CLAY_BOTTOM_Z,
        ),
    )


    return torch.clamp(

        height,

        min=CLAY_BOTTOM_Z,

        max=(
            CLAY_TOP_Z
            +
            0.03
        ),
    )


# ============================================================
# 39. GPU ACTION PLANNER
# ============================================================

@torch.no_grad()
def plan_next_action_gpu(
    current_height,
):

    # --------------------------------------------------------
    # positive:
    #
    # current > target
    #
    # -> 현재 점토가 목표보다 높다
    # -> 눌러야 한다
    #
    # --------------------------------------------------------

    raw_error = (

        current_height

        -

        TARGET_HEIGHT
    )


    positive_error = torch.clamp(

        raw_error,

        min=0.0,
    )


    # --------------------------------------------------------
    # gradient
    # --------------------------------------------------------

    current_grad_x, current_grad_y = (
        gradient_gpu(
            current_height
        )
    )


    gradient_error = torch.sqrt(

        (
            current_grad_x
            -
            TARGET_GRAD_X
        )
        ** 2

        +

        (
            current_grad_y
            -
            TARGET_GRAD_Y
        )
        ** 2
    )


    # slope -> meter-scale approximation

    gradient_error_m = (

        gradient_error

        *

        TOOL_RADIUS
    )


    combined_error = (

        positive_error

        +

        GRADIENT_WEIGHT

        *

        gradient_error_m
    )


    # --------------------------------------------------------
    # 실제 원통 tool footprint convolution
    # --------------------------------------------------------

    score = F.conv2d(

        combined_error[
            None,
            None
        ],

        TOOL_KERNEL,

        padding=TOOL_RADIUS_PX,
    )[
        0,
        0
    ]


    # --------------------------------------------------------
    # 이미 충분히 낮은 곳은
    # 다시 찍지 않음.
    # --------------------------------------------------------

    score = torch.where(

        positive_error
        >
        0.0002,

        score,

        torch.zeros_like(
            score
        ),
    )


    # --------------------------------------------------------
    # workspace mask
    # --------------------------------------------------------

    masked_score = torch.where(

        ACTIVE_MASK,

        score,

        torch.full_like(

            score,

            -torch.inf,
        ),
    )


    # --------------------------------------------------------
    # GPU argmax
    # --------------------------------------------------------

    flat_index = torch.argmax(
        masked_score
    )


    row = (
        flat_index
        //
        HEIGHT_MAP_RES
    )


    col = (
        flat_index
        %
        HEIGHT_MAP_RES
    )


    score_flat = (
        masked_score.flatten()
    )


    error_flat = (
        positive_error.flatten()
    )


    height_flat = (
        current_height.flatten()
    )


    grid_x_flat = (
        GRID_X.flatten()
    )


    grid_y_flat = (
        GRID_Y.flatten()
    )


    best_score = (
        score_flat[
            flat_index
        ]
    )


    pixel_error = (
        error_flat[
            flat_index
        ]
    )


    surface_z = (
        height_flat[
            flat_index
        ]
    )


    x = (
        grid_x_flat[
            flat_index
        ]
    )


    y = (
        grid_y_flat[
            flat_index
        ]
    )


    # --------------------------------------------------------
    # adaptive press depth
    # --------------------------------------------------------

    press_depth = torch.clamp(

        torch.maximum(

            best_score,

            pixel_error,
        )

        *

        PRESS_GAIN,

        min=MIN_PRESS_DEPTH,

        max=MAX_PRESS_DEPTH,
    )


    active_errors = (
        positive_error[
            ACTIVE_MASK
        ]
    )


    mean_error = torch.mean(
        active_errors
    )


    max_error = torch.max(
        active_errors
    )


    # --------------------------------------------------------
    # 이 한 번만 CPU sync.
    #
    # 수만 개 particle이 아니라
    # action scalar 10개 정도만 이동.
    # --------------------------------------------------------

    action_pack = torch.stack(

        [
            row.to(
                FLOAT_DTYPE
            ),

            col.to(
                FLOAT_DTYPE
            ),

            x,

            y,

            surface_z,

            press_depth,

            best_score,

            pixel_error,

            mean_error,

            max_error,
        ]
    )


    action_cpu = (

        action_pack
        .detach()
        .cpu()
        .tolist()
    )


    action = {

        "row":
        int(
            action_cpu[0]
        ),

        "col":
        int(
            action_cpu[1]
        ),

        "x":
        float(
            action_cpu[2]
        ),

        "y":
        float(
            action_cpu[3]
        ),

        "surface_z":
        float(
            action_cpu[4]
        ),

        "press_depth":
        float(
            action_cpu[5]
        ),

        "score":
        float(
            action_cpu[6]
        ),

        "pixel_error":
        float(
            action_cpu[7]
        ),

        "mean_error":
        float(
            action_cpu[8]
        ),

        "max_error":
        float(
            action_cpu[9]
        ),
    }


    return (
        action,
        raw_error,
        score,
    )


# ============================================================
# 40. SAVE GPU MAP
# ============================================================

def save_gpu_map(
    tensor,
    path,
    vmin,
    vmax,
):

    if not SAVE_PNG:

        return

    if Image is None:

        return

    # 저장시에만 CPU transfer

    array = (

        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    normalized = (

        array
        -
        vmin

    ) / max(

        vmax
        -
        vmin,

        1e-8,
    )


    normalized = np.clip(

        normalized,

        0.0,

        1.0,
    )


    image = (

        normalized

        *

        255.0

    ).astype(
        np.uint8
    )


    Image.fromarray(
        image
    ).save(
        path
    )


# ============================================================
# 41. SAVE TARGET
# ============================================================

save_gpu_map(

    TARGET_HEIGHT,

    os.path.join(

        STATE_DIR,

        "target_heart.png",
    ),

    CLAY_BOTTOM_Z,

    CLAY_TOP_Z,
)


# ============================================================
# 42. PERIODIC SAVE
# ============================================================

def save_iteration(

    step,

    current,

    raw_error,

    score,
):

    if (
        step
        %
        SAVE_EVERY
        !=
        0
    ):

        return


    save_gpu_map(

        current,

        os.path.join(

            STATE_DIR,

            f"current_{step:03d}.png",
        ),

        CLAY_BOTTOM_Z,

        CLAY_TOP_Z
        +
        0.015,
    )


    save_gpu_map(

        torch.clamp(

            raw_error,

            min=0.0,
        ),

        os.path.join(

            STATE_DIR,

            f"error_{step:03d}.png",
        ),

        0.0,

        HEART_DEPTH,
    )


    save_gpu_map(

        score,

        os.path.join(

            STATE_DIR,

            f"score_{step:03d}.png",
        ),

        0.0,

        HEART_DEPTH,
    )


# ============================================================
# 43. VERTICAL STAMP
# ============================================================

def vertical_stamp(

    x,

    y,

    surface_z,

    press_depth,

    stamp_index,
):

    hover_z = max(

        CLAY_TOP_Z
        +
        HOVER_HEIGHT,

        surface_z
        +
        HOVER_HEIGHT,
    )


    precontact_z = (

        surface_z

        +

        PRECONTACT_HEIGHT
    )


    press_z = (

        surface_z

        -

        press_depth
    )


    hover = torch.tensor(

        [
            x,
            y,
            hover_z,
        ],

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )


    precontact = torch.tensor(

        [
            x,
            y,
            precontact_z,
        ],

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )


    press = torch.tensor(

        [
            x,
            y,
            press_z,
        ],

        device=DEVICE,

        dtype=FLOAT_DTYPE,
    )


    print()

    print(
        f"[ACTION {stamp_index:03d}]"
    )

    print(
        f"  XY            = "
        f"({x:.3f}, {y:.3f})"
    )

    print(
        f"  surface       = "
        f"{surface_z * 1000:.2f} mm"
    )

    print(
        f"  press depth   = "
        f"{press_depth * 1000:.2f} mm"
    )


    # --------------------------------------------------------
    # AIR TRAVEL
    # --------------------------------------------------------

    move_tip_air_gpu(

        hover,

        TRAVEL_STEPS,
    )


    # --------------------------------------------------------
    # DOWN
    # --------------------------------------------------------

    move_tip_linear_gpu(

        hover,

        precontact,

        DOWN_STEPS,
    )


    # --------------------------------------------------------
    # PRESS
    # --------------------------------------------------------

    q_press = (
        move_tip_linear_gpu(

            precontact,

            press,

            PRESS_STEPS,
        )
    )


    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    for _ in range(
        STAMP_HOLD_STEPS
    ):

        iiwa.control_dofs_position(

            position=q_press,

            dofs_idx_local=DOF_SLICE,
        )

        sim_step()


    # --------------------------------------------------------
    # UP
    # --------------------------------------------------------

    q_hover = (
        move_tip_linear_gpu(

            press,

            hover,

            UP_STEPS,
        )
    )


    # --------------------------------------------------------
    # SETTLE
    # --------------------------------------------------------

    for _ in range(
        STAMP_SETTLE_STEPS
    ):

        iiwa.control_dofs_position(

            position=q_hover,

            dofs_idx_local=DOF_SLICE,
        )

        sim_step()


# ============================================================
# 44. INITIAL SETTLE
# ============================================================

print()

print(
    "Initial clay settling..."
)


for _ in range(8):

    sim_step()


print(
    "Setting upright robot pose..."
)


for _ in range(10):

    iiwa.control_dofs_position(

        position=INITIAL_Q,

        dofs_idx_local=DOF_SLICE,
    )

    sim_step()


# ============================================================
# 45. HOME
# ============================================================

HOME_TIP = torch.tensor(

    [
        CLAY_CENTER_X,

        CLAY_CENTER_Y,

        CLAY_TOP_Z
        +
        HOVER_HEIGHT,
    ],

    device=DEVICE,

    dtype=FLOAT_DTYPE,
)


print(
    "Moving tool above clay..."
)


move_tip_air_gpu(

    HOME_TIP,

    10,
)


sim_step(
    force_visual=True,
)


# ============================================================
# 46. LOG
# ============================================================

ACTION_LOG_PATH = os.path.join(

    ACTION_DIR,

    "heart_actions.csv",
)


# ============================================================
# 47. CLOSED LOOP
# ============================================================

print()

print(
    "###############################################"
)

print(
    "#                                             #"
)

print(
    "#       GPU HEART SCULPTING START             #"
)

print(
    "#                                             #"
)

print(
    "###############################################"
)

print()


start_time = (
    time.perf_counter()
)


executed_actions = 0


with open(

    ACTION_LOG_PATH,

    "w",

    newline="",

) as action_file:

    writer = csv.writer(
        action_file
    )


    writer.writerow(

        [
            "step",

            "row",

            "col",

            "x",

            "y",

            "surface_z",

            "press_depth",

            "score",

            "pixel_error",

            "mean_error",

            "max_error",
        ]
    )


    for step_idx in range(
        MAX_ACTIONS
    ):

        # ====================================================
        # A. MPM PARTICLES -> HEIGHT MAP
        #
        # GPU ONLY
        # ====================================================

        current_height = (
            get_current_height_map_gpu()
        )


        # ====================================================
        # B. PLAN
        #
        # GPU ONLY
        # ====================================================

        (
            action,
            raw_error,
            score_map,

        ) = plan_next_action_gpu(

            current_height
        )


        print()

        print(
            f"[PLAN {step_idx:03d}]"
        )

        print(
            "  mean error = "
            f"{action['mean_error'] * 1000:.2f} mm"
        )

        print(
            "  max error  = "
            f"{action['max_error'] * 1000:.2f} mm"
        )

        print(
            "  score      = "
            f"{action['score'] * 1000:.2f} mm"
        )

        print(
            "  selected   = "
            f"({action['row']}, "
            f"{action['col']})"
        )


        # ====================================================
        # C. STOP
        # ====================================================

        if (
            action[
                "score"
            ]
            <
            ERROR_THRESHOLD
        ):

            print()

            print(
                "Heart target threshold reached."
            )

            break


        # ====================================================
        # D. PERIODIC SAVE
        # ====================================================

        save_iteration(

            step_idx,

            current_height,

            raw_error,

            score_map,
        )


        # ====================================================
        # E. LOG
        # ====================================================

        writer.writerow(

            [
                step_idx,

                action[
                    "row"
                ],

                action[
                    "col"
                ],

                action[
                    "x"
                ],

                action[
                    "y"
                ],

                action[
                    "surface_z"
                ],

                action[
                    "press_depth"
                ],

                action[
                    "score"
                ],

                action[
                    "pixel_error"
                ],

                action[
                    "mean_error"
                ],

                action[
                    "max_error"
                ],
            ]
        )


        # ====================================================
        # F. EXECUTE STAMP
        # ====================================================

        vertical_stamp(

            x=action[
                "x"
            ],

            y=action[
                "y"
            ],

            surface_z=action[
                "surface_z"
            ],

            press_depth=action[
                "press_depth"
            ],

            stamp_index=(
                step_idx
                +
                1
            ),
        )


        executed_actions += 1


# ============================================================
# 48. RETURN HOME
# ============================================================

print()

print(
    "Returning home..."
)


move_tip_air_gpu(

    HOME_TIP,

    8,
)


q_final = get_q_gpu()


for _ in range(10):

    iiwa.control_dofs_position(

        position=q_final,

        dofs_idx_local=DOF_SLICE,
    )

    sim_step()


# ============================================================
# 49. FINAL GPU MEASUREMENT
# ============================================================

final_height = (
    get_current_height_map_gpu()
)


final_error = (

    final_height

    -

    TARGET_HEIGHT
)


final_positive = torch.clamp(

    final_error[
        ACTIVE_MASK
    ],

    min=0.0,
)


final_mean = torch.mean(
    final_positive
)


final_max = torch.max(
    final_positive
)


# ============================================================
# 50. FINAL SAVE
# ============================================================

save_gpu_map(

    final_height,

    os.path.join(

        STATE_DIR,

        "final_height.png",
    ),

    CLAY_BOTTOM_Z,

    CLAY_TOP_Z
    +
    0.015,
)


save_gpu_map(

    torch.clamp(

        final_error,

        min=0.0,
    ),

    os.path.join(

        STATE_DIR,

        "final_error.png",
    ),

    0.0,

    HEART_DEPTH,
)


# 최종 통계만 CPU sync

final_stats = (

    torch.stack(

        [
            final_mean,
            final_max,
        ]
    )

    .detach()

    .cpu()

    .tolist()
)


elapsed = (

    time.perf_counter()

    -

    start_time
)


# ============================================================
# 51. RESULT
# ============================================================

print()

print(
    "==============================================="
)

print(
    "GPU HEART SCULPTING COMPLETE"
)

print(
    "==============================================="
)

print(
    "Actions:",
    executed_actions,
)

print(
    "Wall time:",
    round(
        elapsed,
        2,
    ),
    "sec",
)

print(
    "Final mean error:",
    round(
        final_stats[0]
        *
        1000,
        3,
    ),
    "mm",
)

print(
    "Final max error:",
    round(
        final_stats[1]
        *
        1000,
        3,
    ),
    "mm",
)

print(
    "Output:",
    OUTPUT_DIR,
)

print(
    "==============================================="
)


sim_step(
    force_visual=True,
)


# ============================================================
# 52. HOLD FINAL STATE
# ============================================================

while True:

    iiwa.control_dofs_position(

        position=q_final,

        dofs_idx_local=DOF_SLICE,
    )

    sim_step()