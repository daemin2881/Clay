import os
import math


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
)


CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
)


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)


GENERATED_DIR = os.path.join(
    BASE_DIR,
    "assets",
    "generated",
)


os.makedirs(
    DATASET_DIR,
    exist_ok=True,
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


os.makedirs(
    GENERATED_DIR,
    exist_ok=True,
)


# ============================================================
# SIMULATION
# ============================================================

DT = 0.004

SUBSTEPS = 16


SHOW_VIEWER = (
    os.environ.get(
        "VS_SHOW_VIEWER",
        "1",
    )
    == "1"
)


# ============================================================
# MPM
# ============================================================

MPM_GRID_DENSITY = 64

MPM_PARTICLE_SIZE = 0.006


MPM_LOWER_BOUND = (
    0.16,
    -0.28,
    -0.08,
)


MPM_UPPER_BOUND = (
    0.72,
    0.28,
    0.20,
)


# ============================================================
# CLAY GEOMETRY
# ============================================================

CLAY_CENTER_X = 0.44

CLAY_CENTER_Y = 0.0


CLAY_SIZE_X = 0.36

CLAY_SIZE_Y = 0.36

CLAY_SIZE_Z = 0.060


CLAY_BOTTOM_Z = 0.0


CLAY_TOP_Z = (
    CLAY_BOTTOM_Z
    +
    CLAY_SIZE_Z
)


# ============================================================
# LEGACY CLAY ALIASES
# ============================================================

CLAY_X = CLAY_CENTER_X

CLAY_Y = CLAY_CENTER_Y


CLAY_WIDTH = CLAY_SIZE_X

CLAY_LENGTH = CLAY_SIZE_Y

CLAY_DEPTH = CLAY_SIZE_Y

CLAY_HEIGHT = CLAY_SIZE_Z


# ============================================================
# REPRESENTATION CONSTANTS
# ============================================================

ACTIVE_MARGIN = 0.060


PARTICLE_SURFACE_OFFSET = (
    MPM_PARTICLE_SIZE
    *
    0.5
)


# ============================================================
# MATERIAL
# ============================================================

MATERIAL_PRESETS = {

    # --------------------------------------------------------
    # SOFT
    #
    # Kept unchanged for now.
    # --------------------------------------------------------

    "soft": {

        "E":
        8.0e3,

        "nu":
        0.30,

        "rho":
        1200.0,

        "yield_stress":
        130.0,
    },


    # --------------------------------------------------------
    # MEDIUM
    #
    # UPDATED:
    #
    # Previous:
    # E            = 12,000 Pa
    # yield_stress =    180 Pa
    #
    # Problem:
    # after one push, passive relaxation was larger than
    # the actual immediate push deformation.
    #
    # Test:
    # E            = 30,000 Pa
    # yield_stress =  1,000 Pa
    #
    # Goal:
    # - resist gravity/passive flow
    # - preserve local plastic deformation
    # - still deform under 1~8 mm tool pushes
    # --------------------------------------------------------

    "medium": {
        "E": 2.0e4,
        "nu": 0.30,
        "rho": 1200.0,
        "yield_stress": 300.0,
    },

    # --------------------------------------------------------
    # HARD
    #
    # Kept unchanged for now.
    # --------------------------------------------------------

    "hard": {

        "E":
        2.0e4,

        "nu":
        0.30,

        "rho":
        1200.0,

        "yield_stress":
        260.0,
    },
}


MATERIAL_NAME = os.environ.get(
    "VS_MATERIAL",
    "medium",
)


if MATERIAL_NAME not in MATERIAL_PRESETS:

    raise ValueError(
        f"Unknown material: {MATERIAL_NAME}"
    )


MATERIAL = MATERIAL_PRESETS[
    MATERIAL_NAME
]


CLAY_E = MATERIAL[
    "E"
]


CLAY_NU = MATERIAL[
    "nu"
]


CLAY_RHO = MATERIAL[
    "rho"
]


CLAY_YIELD_STRESS = MATERIAL[
    "yield_stress"
]


# ============================================================
# END EFFECTOR
# ============================================================

TOOL_RADIUS = 0.015

TOOL_SHAFT_RADIUS = 0.014

TOOL_TIP_RADIUS = 0.015

TOOL_LENGTH = 0.075

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


TOOL_URDF_PATH = os.path.join(
    GENERATED_DIR,
    "visual_sculpting_tool.urdf",
)


TOOL_SHAFT_LENGTH = max(
    TOOL_LENGTH
    -
    TOOL_TIP_RADIUS,
    0.001,
)


TOOL_TIP_CENTER_Z = (
    TOOL_SHAFT_LENGTH
)


TOOL_CONTACT_OFFSET = (
    TOOL_LENGTH
)


# ============================================================
# HEIGHT MAP
# ============================================================

HEIGHT_MAP_RES = 512

MAP_H = HEIGHT_MAP_RES

MAP_W = HEIGHT_MAP_RES


HEIGHT_MAP_HEIGHT = HEIGHT_MAP_RES

HEIGHT_MAP_WIDTH = HEIGHT_MAP_RES


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


PARTICLE_RADIUS_PX = max(
    1,
    int(
        math.ceil(
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
    ),
)


# ============================================================
# CLAY BOUNDS
# ============================================================

CLAY_X_MIN = (
    CLAY_CENTER_X
    -
    CLAY_WIDTH
    /
    2.0
)


CLAY_X_MAX = (
    CLAY_CENTER_X
    +
    CLAY_WIDTH
    /
    2.0
)


CLAY_Y_MIN = (
    CLAY_CENTER_Y
    -
    CLAY_LENGTH
    /
    2.0
)


CLAY_Y_MAX = (
    CLAY_CENTER_Y
    +
    CLAY_LENGTH
    /
    2.0
)


# ============================================================
# WORKSPACE
#
# 30 mm spherical tip:
# radius = 15 mm
#
# 18 mm margin keeps the tool away from the outermost
# unstable reconstruction boundary.
# ============================================================

WORKSPACE_MARGIN = 0.018


WORKSPACE_X_MIN = (
    CLAY_CENTER_X
    -
    CLAY_SIZE_X
    /
    2.0
    +
    WORKSPACE_MARGIN
)


WORKSPACE_X_MAX = (
    CLAY_CENTER_X
    +
    CLAY_SIZE_X
    /
    2.0
    -
    WORKSPACE_MARGIN
)


WORKSPACE_Y_MIN = (
    CLAY_CENTER_Y
    -
    CLAY_SIZE_Y
    /
    2.0
    +
    WORKSPACE_MARGIN
)


WORKSPACE_Y_MAX = (
    CLAY_CENTER_Y
    +
    CLAY_SIZE_Y
    /
    2.0
    -
    WORKSPACE_MARGIN
)


X_MIN = WORKSPACE_X_MIN

X_MAX = WORKSPACE_X_MAX

Y_MIN = WORKSPACE_Y_MIN

Y_MAX = WORKSPACE_Y_MAX


# ============================================================
# ACTION SPACE
# ============================================================

ACTION_THETA_MIN = -math.pi

ACTION_THETA_MAX = math.pi



# 5mm의 짧은 push부터
# 점토 사용 가능 영역의 한쪽 끝에서 반대쪽 끝까지
ACTION_LENGTH_MIN = 0.005

ACTION_LENGTH_MAX = min(
    CLAY_SIZE_X,
    CLAY_SIZE_Y,
) - 2.0 * WORKSPACE_MARGIN


# 1mm의 얕은 push부터
# 바닥 위 10mm를 남기는 깊은 push까지
ACTION_DEPTH_MIN = 0.001

ACTION_DEPTH_MAX = (
    CLAY_SIZE_Z
    -
    0.010
)


THETA_MIN = ACTION_THETA_MIN

THETA_MAX = ACTION_THETA_MAX


LENGTH_MIN = ACTION_LENGTH_MIN

LENGTH_MAX = ACTION_LENGTH_MAX


DEPTH_MIN = ACTION_DEPTH_MIN

DEPTH_MAX = ACTION_DEPTH_MAX


PUSH_LENGTH_MIN = ACTION_LENGTH_MIN

PUSH_LENGTH_MAX = ACTION_LENGTH_MAX


PUSH_DEPTH_MIN = ACTION_DEPTH_MIN

PUSH_DEPTH_MAX = ACTION_DEPTH_MAX


# ============================================================
# DATA COLLECTION SPATIAL COVERAGE
# ============================================================

COLLECTION_GRID_X = 8

COLLECTION_GRID_Y = 8


COLLECTION_RECENT_HISTORY = 6


# 40 mm
COLLECTION_MIN_START_DISTANCE = 0.040


COLLECTION_CELL_JITTER_MARGIN = 0.10


COLLECTION_MAX_RESAMPLE = 20


# ============================================================
# DEPTH STRATIFICATION
# ============================================================

COLLECTION_DEPTH_SHALLOW_RATIO = 0.30

COLLECTION_DEPTH_MEDIUM_RATIO = 0.45

COLLECTION_DEPTH_DEEP_RATIO = 0.25


COLLECTION_DEPTH_SHALLOW_MIN = 0.001

COLLECTION_DEPTH_SHALLOW_MAX = 0.003


COLLECTION_DEPTH_MEDIUM_MIN = 0.003

COLLECTION_DEPTH_MEDIUM_MAX = 0.006


COLLECTION_DEPTH_DEEP_MIN = 0.006

COLLECTION_DEPTH_DEEP_MAX = 0.008


# ============================================================
# LENGTH STRATIFICATION
# ============================================================

COLLECTION_LENGTH_SHORT_RATIO = 0.30

COLLECTION_LENGTH_MEDIUM_RATIO = 0.45

COLLECTION_LENGTH_LONG_RATIO = 0.25


COLLECTION_LENGTH_SHORT_MIN = 0.012

COLLECTION_LENGTH_SHORT_MAX = 0.028


COLLECTION_LENGTH_MEDIUM_MIN = 0.028

COLLECTION_LENGTH_MEDIUM_MAX = 0.048


COLLECTION_LENGTH_LONG_MIN = 0.048

COLLECTION_LENGTH_LONG_MAX = 0.065


# ============================================================
# EDGE THETA SAMPLING
#
# edge cells:
# 70% inward-directed
# 30% random
# ============================================================

COLLECTION_EDGE_INWARD_RATIO = 0.70


COLLECTION_EDGE_THETA_NOISE = 0.45


# ============================================================
# ROBOT MOTION
# ============================================================

HOVER_HEIGHT = 0.060

PRECONTACT_HEIGHT = 0.002


TRAVEL_STEPS = 300
DOWN_STEPS = 200
PUSH_STEPS = 12
UP_STEPS = 300

HOLD_STEPS = 20
SETTLE_STEPS = 20


N_TRAVEL_STEPS = TRAVEL_STEPS

N_DOWN_STEPS = DOWN_STEPS

N_PUSH_STEPS = PUSH_STEPS

N_UP_STEPS = UP_STEPS

N_HOLD_STEPS = HOLD_STEPS

N_SETTLE_STEPS = SETTLE_STEPS


# ============================================================
# TARGET
# ============================================================

TARGET_EDGE_RATIO = 0.10

TARGET_EDGE_FRACTION = TARGET_EDGE_RATIO

TARGET_OUTER_RATIO = TARGET_EDGE_RATIO

TARGET_OUTER_FRACTION = TARGET_EDGE_RATIO


# ============================================================
# DATASET
# ============================================================

COLLECTION_BUFFER_SIZE = 32


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

EPOCHS = 150

TRAIN_RATIO = 0.80

SEED = 0


# ============================================================
# PLANNER
# ============================================================

PLAN_HORIZON = 40

MPC_EXECUTE = 5

GREEDY_CANDIDATES = 64

REFINE_STEPS = 15

PLAN_LR = 0.03


ERROR_GUIDED_RATIO = 0.70

UNIFORM_RATIO = 0.30


THETA_NOISE_STD = 0.30


REPEAT_RADIUS = 0.025

REPEAT_PENALTY = 0.15

RECENT_ACTION_HISTORY = 12


PLAN_W_3D = 1.0

PLAN_W_VIZ = 1.0

LOSS_EPS = 1e-8


# ============================================================
# TERMINATION
# ============================================================

TARGET_MEAN_ERROR = 0.001

MAX_ACTIONS = 500


# ============================================================
# LETTER EXPERIMENT
# ============================================================

LETTER_ACTION_MILESTONES = {

    "A":
    190,

    "B":
    250,

    "C":
    310,

    "D":
    370,

    "E":
    430,

    "F":
    490,
}


LETTER_SEQUENCE = [

    "A",

    "B",

    "C",

    "D",

    "E",

    "F",
]


# ============================================================
# EXPERIMENTS
# ============================================================

SAMPLE_COUNTS = [

    25,

    50,

    100,

    200,

    300,

    400,

    500,

    600,

    700,
]


NOISE_LEVELS = [

    0.00,

    0.05,

    0.10,

    0.15,

    0.20,

    0.25,

    0.30,
]


# ============================================================
# DEBUG
# ============================================================

def print_config_summary():

    print()

    print(
        "==============================================="
    )

    print(
        "VISUAL SCULPTING CONFIG"
    )

    print(
        "==============================================="
    )


    print(
        "material =",
        MATERIAL_NAME
    )


    print(
        "E =",
        CLAY_E
    )


    print(
        "yield stress =",
        CLAY_YIELD_STRESS
    )


    print(
        "workspace X =",
        X_MIN,
        X_MAX,
    )


    print(
        "workspace Y =",
        Y_MIN,
        Y_MAX,
    )


    print(
        "workspace margin =",
        WORKSPACE_MARGIN,
    )


    print(
        "collection grid =",
        COLLECTION_GRID_X,
        "x",
        COLLECTION_GRID_Y,
    )


    print(
        "length range =",
        LENGTH_MIN,
        LENGTH_MAX,
    )


    print(
        "depth range =",
        DEPTH_MIN,
        DEPTH_MAX,
    )


    print(
        "tip diameter =",
        TOOL_TIP_RADIUS
        *
        2.0,
    )


    print(
        "==============================================="
    )


if __name__ == "__main__":

    print_config_summary()