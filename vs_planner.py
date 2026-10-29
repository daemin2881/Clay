import math

import torch


from vs_config import (
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,

    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_SIZE_X,
    CLAY_SIZE_Y,

    LENGTH_MIN,
    LENGTH_MAX,

    DEPTH_MIN,
    DEPTH_MAX,

    PLAN_HORIZON,
    GREEDY_CANDIDATES,
    REFINE_STEPS,
    PLAN_LR,

    ERROR_GUIDED_RATIO,
    UNIFORM_RATIO,

    THETA_NOISE_STD,

    REPEAT_RADIUS,
    REPEAT_PENALTY,
    RECENT_ACTION_HISTORY,

    PLAN_W_3D,
    PLAN_W_VIZ,

    LOSS_EPS,
)


from vs_representation import (
    ensure_bchw,
    actual_to_canonical,
    canonical_to_actual,
    make_state_representation,
    spatial_gradient,
)


from vs_dataset import (
    normalize_shape_params,
)


# ============================================================
# UTILITY
# ============================================================

def wrap_angle(
    theta,
):
    """
    angle -> [-pi, pi]
    """

    return torch.atan2(
        torch.sin(theta),
        torch.cos(theta),
    )


# ============================================================
# ENDPOINT REPAIR
#
# action =
# [x, y, theta, length, depth]
#
# push endpoint가 workspace 밖으로 나가면
# length를 GPU에서 줄인다.
# ============================================================

def repair_action_endpoints(
    actions,
):

    actions = actions.clone()


    for _ in range(
        6
    ):

        end_x = (

            actions[
                :,
                0
            ]

            +

            actions[
                :,
                3
            ]

            *

            torch.cos(
                actions[
                    :,
                    2
                ]
            )
        )


        end_y = (

            actions[
                :,
                1
            ]

            +

            actions[
                :,
                3
            ]

            *

            torch.sin(
                actions[
                    :,
                    2
                ]
            )
        )


        invalid = (

            (end_x < X_MIN)

            |

            (end_x > X_MAX)

            |

            (end_y < Y_MIN)

            |

            (end_y > Y_MAX)
        )


        actions[
            invalid,
            3
        ] *= 0.70


    # --------------------------------------------------------
    # 너무 짧아졌다면 방향을 workspace 중앙으로 돌림
    # --------------------------------------------------------

    end_x = (

        actions[:, 0]

        +

        actions[:, 3]

        *

        torch.cos(
            actions[:, 2]
        )
    )


    end_y = (

        actions[:, 1]

        +

        actions[:, 3]

        *

        torch.sin(
            actions[:, 2]
        )
    )


    invalid = (

        (end_x < X_MIN)

        |

        (end_x > X_MAX)

        |

        (end_y < Y_MIN)

        |

        (end_y > Y_MAX)
    )


    if torch.any(
        invalid
    ):

        center_x = torch.full_like(

            actions[
                invalid,
                0
            ],

            (
                X_MIN
                +
                X_MAX
            )
            /
            2.0,
        )


        center_y = torch.full_like(

            actions[
                invalid,
                1
            ],

            (
                Y_MIN
                +
                Y_MAX
            )
            /
            2.0,
        )


        theta_center = torch.atan2(

            center_y
            -
            actions[
                invalid,
                1
            ],

            center_x
            -
            actions[
                invalid,
                0
            ],
        )


        actions[
            invalid,
            2
        ] = theta_center


    actions[
        :,
        3
    ] = torch.clamp(

        actions[
            :,
            3
        ],

        LENGTH_MIN,

        LENGTH_MAX,
    )


    actions[
        :,
        4
    ] = torch.clamp(

        actions[
            :,
            4
        ],

        DEPTH_MIN,

        DEPTH_MAX,
    )


    return actions


# ============================================================
# PURE UNIFORM RANDOM ACTION BATCH
#
# dataset collection / noise experiments에서도 사용 가능.
# ============================================================

def sample_actions_gpu(
    n,
    device,
    dtype=torch.float32,
):

    r = torch.rand(

        (
            n,
            5,
        ),

        device=device,

        dtype=dtype,
    )


    actions = torch.empty_like(
        r
    )


    # --------------------------------------------------------
    # x
    # --------------------------------------------------------

    actions[
        :,
        0
    ] = (

        X_MIN

        +

        r[
            :,
            0
        ]

        *

        (
            X_MAX
            -
            X_MIN
        )
    )


    # --------------------------------------------------------
    # y
    # --------------------------------------------------------

    actions[
        :,
        1
    ] = (

        Y_MIN

        +

        r[
            :,
            1
        ]

        *

        (
            Y_MAX
            -
            Y_MIN
        )
    )


    # --------------------------------------------------------
    # theta
    # --------------------------------------------------------

    actions[
        :,
        2
    ] = (

        -math.pi

        +

        r[
            :,
            2
        ]

        *

        (
            2.0
            *
            math.pi
        )
    )


    # --------------------------------------------------------
    # length
    # --------------------------------------------------------

    actions[
        :,
        3
    ] = (

        LENGTH_MIN

        +

        r[
            :,
            3
        ]

        *

        (
            LENGTH_MAX
            -
            LENGTH_MIN
        )
    )


    # --------------------------------------------------------
    # depth
    # --------------------------------------------------------

    actions[
        :,
        4
    ] = (

        DEPTH_MIN

        +

        r[
            :,
            4
        ]

        *

        (
            DEPTH_MAX
            -
            DEPTH_MIN
        )
    )


    actions = repair_action_endpoints(
        actions
    )


    return actions


# ============================================================
# TARGET ERROR MAP
#
# current와 target이 크게 다른 위치일수록
# 높은 sampling probability를 갖는다.
#
# height error + 약한 gradient error
# ============================================================

@torch.no_grad()
def make_error_sampling_map(
    current,
    target,
):

    current = ensure_bchw(
        current
    )


    target = ensure_bchw(
        target
    )


    if target.shape[
        0
    ] == 1:

        target = target.expand(

            current.shape[
                0
            ],

            -1,
            -1,
            -1,
        )


    # --------------------------------------------------------
    # HEIGHT ERROR
    # --------------------------------------------------------

    height_error = torch.mean(

        torch.abs(
            current
            -
            target
        ),

        dim=1,
    )


    # --------------------------------------------------------
    # VISUAL / GRADIENT ERROR
    # --------------------------------------------------------

    cgx, cgy = spatial_gradient(
        current
    )


    tgx, tgy = spatial_gradient(
        target
    )


    grad_error = torch.mean(

        torch.abs(
            cgx
            -
            tgx
        )

        +

        torch.abs(
            cgy
            -
            tgy
        ),

        dim=1,
    )


    # --------------------------------------------------------
    # 각각 normalize
    # --------------------------------------------------------

    h_scale = torch.mean(

        height_error,

        dim=(
            1,
            2,
        ),

        keepdim=True,
    ).clamp_min(
        LOSS_EPS
    )


    g_scale = torch.mean(

        grad_error,

        dim=(
            1,
            2,
        ),

        keepdim=True,
    ).clamp_min(
        LOSS_EPS
    )


    height_error = (

        height_error

        /

        h_scale
    )


    grad_error = (

        grad_error

        /

        g_scale
    )


    # height를 조금 더 강하게 사용.
    error_map = (

        height_error

        +

        0.35
        *
        grad_error
    )


    # planner는 single state를 사용하므로 첫 배치.
    error_map = error_map[
        0
    ]


    # --------------------------------------------------------
    # clay 전체 map 중 실제 reachable workspace만 허용
    # --------------------------------------------------------

    H = error_map.shape[
        0
    ]

    W = error_map.shape[
        1
    ]


    ys = torch.linspace(

        CLAY_CENTER_Y
        -
        CLAY_SIZE_Y
        /
        2.0,

        CLAY_CENTER_Y
        +
        CLAY_SIZE_Y
        /
        2.0,

        H,

        device=error_map.device,

        dtype=error_map.dtype,
    )


    xs = torch.linspace(

        CLAY_CENTER_X
        -
        CLAY_SIZE_X
        /
        2.0,

        CLAY_CENTER_X
        +
        CLAY_SIZE_X
        /
        2.0,

        W,

        device=error_map.device,

        dtype=error_map.dtype,
    )


    valid_y = (

        (ys >= Y_MIN)

        &

        (ys <= Y_MAX)
    )


    valid_x = (

        (xs >= X_MIN)

        &

        (xs <= X_MAX)
    )


    valid_mask = (

        valid_y[
            :,
            None
        ]

        &

        valid_x[
            None,
            :
        ]
    )


    error_map = (

        error_map

        *

        valid_mask.to(
            error_map.dtype
        )
    )


    # --------------------------------------------------------
    # error가 큰 영역을 더 자주 뽑기 위해 power
    # --------------------------------------------------------

    error_map = torch.clamp(

        error_map,

        min=0.0,
    )


    error_map = (

        error_map
        +
        1e-6
    ) ** 1.50


    return error_map


# ============================================================
# PIXEL INDEX -> WORLD XY
# ============================================================

def pixel_indices_to_world(
    indices,
    H,
    W,
    device,
    dtype,
):

    row = torch.div(

        indices,

        W,

        rounding_mode="floor",
    )


    col = (

        indices

        %

        W
    )


    x = (

        CLAY_CENTER_X
        -
        CLAY_SIZE_X
        /
        2.0

        +

        col.to(
            dtype
        )

        /

        max(
            W
            -
            1,
            1,
        )

        *

        CLAY_SIZE_X
    )


    y = (

        CLAY_CENTER_Y
        -
        CLAY_SIZE_Y
        /
        2.0

        +

        row.to(
            dtype
        )

        /

        max(
            H
            -
            1,
            1,
        )

        *

        CLAY_SIZE_Y
    )


    # --------------------------------------------------------
    # 작은 pixel jitter
    # --------------------------------------------------------

    pixel_size_x = (

        CLAY_SIZE_X

        /

        max(
            W
            -
            1,
            1,
        )
    )


    pixel_size_y = (

        CLAY_SIZE_Y

        /

        max(
            H
            -
            1,
            1,
        )
    )


    jitter_x = (

        torch.rand_like(
            x
        )

        -
        0.5
    ) * pixel_size_x


    jitter_y = (

        torch.rand_like(
            y
        )

        -
        0.5
    ) * pixel_size_y


    x = torch.clamp(

        x
        +
        jitter_x,

        X_MIN,

        X_MAX,
    )


    y = torch.clamp(

        y
        +
        jitter_y,

        Y_MIN,

        Y_MAX,
    )


    return (
        x,
        y,
    )


# ============================================================
# ERROR-GUIDED ACTION SAMPLING
#
# 70%:
#   target error가 큰 곳
#
# 30%:
#   전체 random exploration
#
# guided action의 theta:
#
#   error point 1 -> error point 2
#
# 방향 + small noise
# ============================================================

@torch.no_grad()
def sample_error_guided_actions_gpu(
    current,
    target,
    n,
    device=None,
    dtype=None,
):

    current = ensure_bchw(
        current
    )


    if device is None:

        device = current.device


    if dtype is None:

        dtype = current.dtype


    n_guided = int(

        round(

            n

            *

            ERROR_GUIDED_RATIO
        )
    )


    n_guided = max(

        0,

        min(
            n,
            n_guided,
        ),
    )


    n_uniform = (

        n

        -
        n_guided
    )


    # ========================================================
    # NO GUIDED CASE
    # ========================================================

    if n_guided <= 0:

        return sample_actions_gpu(

            n,

            device,

            dtype,
        )


    error_map = make_error_sampling_map(

        current,

        target,
    )


    H = error_map.shape[
        0
    ]

    W = error_map.shape[
        1
    ]


    weights = error_map.reshape(
        -1
    )


    weights = (

        weights

        /

        torch.sum(
            weights
        ).clamp_min(
            LOSS_EPS
        )
    )


    # --------------------------------------------------------
    # start error pixels
    # --------------------------------------------------------

    idx_start = torch.multinomial(

        weights,

        n_guided,

        replacement=True,
    )


    # --------------------------------------------------------
    # another high-error point determines action direction
    # --------------------------------------------------------

    idx_goal = torch.multinomial(

        weights,

        n_guided,

        replacement=True,
    )


    x0, y0 = pixel_indices_to_world(

        idx_start,

        H,

        W,

        device,

        dtype,
    )


    x1, y1 = pixel_indices_to_world(

        idx_goal,

        H,

        W,

        device,

        dtype,
    )


    dx = (

        x1
        -
        x0
    )


    dy = (

        y1
        -
        y0
    )


    distance = torch.sqrt(

        dx
        *
        dx

        +

        dy
        *
        dy

        +

        1e-12
    )


    theta_target = torch.atan2(

        dy,

        dx,
    )


    # --------------------------------------------------------
    # target points가 거의 같은 곳이면 random direction.
    # --------------------------------------------------------

    random_theta = (

        torch.rand(

            n_guided,

            device=device,

            dtype=dtype,
        )

        *

        (
            2.0
            *
            math.pi
        )

        -

        math.pi
    )


    theta_target = torch.where(

        distance
        >
        0.012,

        theta_target,

        random_theta,
    )


    # --------------------------------------------------------
    # exploration noise
    # --------------------------------------------------------

    theta = (

        theta_target

        +

        torch.randn(

            n_guided,

            device=device,

            dtype=dtype,
        )

        *

        THETA_NOISE_STD
    )


    theta = wrap_angle(
        theta
    )


    # --------------------------------------------------------
    # length
    #
    # target 방향 점까지의 거리를 참고하되
    # action bounds 안으로 제한
    # --------------------------------------------------------

    length_noise = (

        0.75

        +

        0.50

        *

        torch.rand(

            n_guided,

            device=device,

            dtype=dtype,
        )
    )


    length = (

        distance

        *

        length_noise
    )


    random_length = (

        LENGTH_MIN

        +

        torch.rand(

            n_guided,

            device=device,

            dtype=dtype,
        )

        *

        (
            LENGTH_MAX
            -
            LENGTH_MIN
        )
    )


    # distance가 너무 작으면 random length
    length = torch.where(

        distance
        >
        LENGTH_MIN,

        length,

        random_length,
    )


    length = torch.clamp(

        length,

        LENGTH_MIN,

        LENGTH_MAX,
    )


    # --------------------------------------------------------
    # depth
    #
    # 새로운 tool이 1~5 mm이므로
    # 전체 범위 uniform exploration 유지
    # --------------------------------------------------------

    depth = (

        DEPTH_MIN

        +

        torch.rand(

            n_guided,

            device=device,

            dtype=dtype,
        )

        *

        (
            DEPTH_MAX
            -
            DEPTH_MIN
        )
    )


    guided = torch.stack(

        [
            x0,
            y0,
            theta,
            length,
            depth,
        ],

        dim=1,
    )


    guided = repair_action_endpoints(
        guided
    )


    # ========================================================
    # UNIFORM EXPLORATION
    # ========================================================

    if n_uniform > 0:

        uniform = sample_actions_gpu(

            n_uniform,

            device,

            dtype,
        )


        actions = torch.cat(

            [
                guided,
                uniform,
            ],

            dim=0,
        )

    else:

        actions = guided


    # --------------------------------------------------------
    # candidate 순서 섞기
    # --------------------------------------------------------

    permutation = torch.randperm(

        actions.shape[
            0
        ],

        device=device,
    )


    actions = actions[
        permutation
    ]


    return actions


# ============================================================
# ONE STEP MODEL PREDICTION - BATCH
# ============================================================

def predict_next_batch(
    model,
    state,
    actions,
):

    state = ensure_bchw(
        state
    )


    N = actions.shape[
        0
    ]


    if state.shape[
        0
    ] == 1:

        state_batch = state.expand(

            N,

            -1,
            -1,
            -1,
        )


    elif state.shape[
        0
    ] == N:

        state_batch = state


    else:

        raise ValueError(

            "state batch must be 1 "
            "or match action batch."
        )


    x = actions[
        :,
        0
    ]


    y = actions[
        :,
        1
    ]


    theta = actions[
        :,
        2
    ]


    length = actions[
        :,
        3
    ]


    depth = actions[
        :,
        4
    ]


    state_can = actual_to_canonical(

        state_batch,

        x,

        y,

        theta,
    )


    rep = make_state_representation(
        state_can
    )


    params = normalize_shape_params(

        length,

        depth,
    )


    delta_can = model(

        rep,

        params,
    )


    delta_actual = canonical_to_actual(

        delta_can,

        x,

        y,

        theta,
    )


    return (

        state_batch

        +

        delta_actual
    )


# ============================================================
# SINGLE ACTION PREDICTION
# ============================================================

def predict_next(
    model,
    state,
    action,
):

    return predict_next_batch(

        model,

        state,

        action[
            None
        ],
    )


# ============================================================
# LOSS COMPONENTS
#
# returns per-state losses.
# ============================================================

def loss_components(
    predicted,
    target,
):

    predicted = ensure_bchw(
        predicted
    )


    target = ensure_bchw(
        target
    )


    if target.shape[
        0
    ] == 1:

        target = target.expand(

            predicted.shape[
                0
            ],

            -1,
            -1,
            -1,
        )


    # ========================================================
    # 3D / HEIGHT
    # ========================================================

    diff = (

        predicted

        -

        target
    )


    l3d = torch.mean(

        diff
        **
        2,

        dim=(
            1,
            2,
            3,
        ),
    )


    # ========================================================
    # VISUAL GRADIENT
    # ========================================================

    pgx, pgy = spatial_gradient(
        predicted
    )


    tgx, tgy = spatial_gradient(
        target
    )


    lviz = torch.mean(

        (
            pgx
            -
            tgx
        )
        **
        2

        +

        (
            pgy
            -
            tgy
        )
        **
        2,

        dim=(
            1,
            2,
            3,
        ),
    )


    return (
        l3d,
        lviz,
    )


# ============================================================
# LOSS BASELINE
#
# 현재 state -> target error를 기준 1.0으로 놓는다.
#
# 예:
#
# raw:
#   L3D  = 2.5e-5
#   Lviz = 0.30
#
# normalized:
#
#   L3D/base_L3D
#   Lviz/base_Lviz
#
# 따라서 scale 차이가 planner를 압도하지 않는다.
# ============================================================

@torch.no_grad()
def compute_loss_baseline(
    state,
    target,
):

    l3d, lviz = loss_components(

        state,

        target,
    )


    base_l3d = torch.mean(
        l3d
    ).clamp_min(
        LOSS_EPS
    )


    base_lviz = torch.mean(
        lviz
    ).clamp_min(
        LOSS_EPS
    )


    return (
        base_l3d,
        base_lviz,
    )


# ============================================================
# BATCH LOSS PER CANDIDATE
# ============================================================

def candidate_losses(
    predicted,
    target,
    w3d=PLAN_W_3D,
    wviz=PLAN_W_VIZ,
    base_l3d=None,
    base_lviz=None,
):

    l3d, lviz = loss_components(

        predicted,

        target,
    )


    if base_l3d is None:

        base_l3d = torch.mean(
            l3d.detach()
        ).clamp_min(
            LOSS_EPS
        )


    if base_lviz is None:

        base_lviz = torch.mean(
            lviz.detach()
        ).clamp_min(
            LOSS_EPS
        )


    normalized_l3d = (

        l3d

        /

        base_l3d
    )


    normalized_lviz = (

        lviz

        /

        base_lviz
    )


    loss = (

        w3d

        *

        normalized_l3d

        +

        wviz

        *

        normalized_lviz
    )


    return loss


# ============================================================
# RECENT ACTION TENSOR
# ============================================================

def prepare_recent_actions(
    recent_actions,
    device,
    dtype,
):

    if recent_actions is None:

        return None


    if torch.is_tensor(
        recent_actions
    ):

        recent = recent_actions.to(

            device=device,

            dtype=dtype,
        )


    else:

        if len(
            recent_actions
        ) == 0:

            return None


        recent = torch.as_tensor(

            recent_actions,

            device=device,

            dtype=dtype,
        )


    if recent.ndim == 1:

        recent = recent[
            None,
            :
        ]


    if recent.shape[
        0
    ] > RECENT_ACTION_HISTORY:

        recent = recent[

            -RECENT_ACTION_HISTORY:
        ]


    return recent


# ============================================================
# REPEAT POSITION PENALTY
#
# 최근 action start와 가까울수록 penalty 증가.
#
# 완전 금지는 아님.
# ============================================================

def repeat_action_penalty(
    candidate_actions,
    recent_actions,
):

    if recent_actions is None:

        return torch.zeros(

            candidate_actions.shape[
                0
            ],

            device=(
                candidate_actions.device
            ),

            dtype=(
                candidate_actions.dtype
            ),
        )


    if recent_actions.shape[
        0
    ] == 0:

        return torch.zeros(

            candidate_actions.shape[
                0
            ],

            device=(
                candidate_actions.device
            ),

            dtype=(
                candidate_actions.dtype
            ),
        )


    candidate_xy = candidate_actions[
        :,
        None,
        0:2
    ]


    recent_xy = recent_actions[
        None,
        :,
        0:2
    ]


    distances = torch.sqrt(

        torch.sum(

            (
                candidate_xy
                -
                recent_xy
            )
            **
            2,

            dim=-1,
        )

        +

        1e-12
    )


    min_distance = torch.min(

        distances,

        dim=1,
    ).values


    penalty = (

        REPEAT_PENALTY

        *

        torch.exp(

            -0.5

            *

            (
                min_distance

                /

                REPEAT_RADIUS
            )
            **
            2
        )
    )


    return penalty


# ============================================================
# GPU BATCHED GREEDY PLANNER
#
# NEW:
#
# 1. 70% error-guided candidates
# 2. 30% random exploration
# 3. normalized L3D/Lviz
# 4. repeat-action penalty
# 5. recent physical action history support
# ============================================================


# ============================================================
# SAFE PLANNING ACTION RANGE
#
# x/y는 로봇이 안정적으로 도달하는 영역으로 제한한다.
# length는 지나치게 긴 궤적만 제한한다.
# depth는 학습/정규화 전체 범위를 그대로 사용하여
# planner가 얕은/중간/깊은 push 중 손실이 가장 작은 것을
# 직접 선택하게 한다.
# ============================================================

SAFE_PLAN_X_MIN = max(0.380, X_MIN)
SAFE_PLAN_X_MAX = min(0.560, X_MAX)

SAFE_PLAN_Y_MIN = max(-0.110, Y_MIN)
SAFE_PLAN_Y_MAX = min(0.110, Y_MAX)

SAFE_PLAN_LENGTH_MAX = min(
    0.065,
    LENGTH_MAX,
)

SAFE_PLAN_DEPTH_MIN = DEPTH_MIN
SAFE_PLAN_DEPTH_MAX = DEPTH_MAX


def remap_actions_to_safe_range(actions):

    if actions.ndim != 2 or actions.shape[1] < 5:
        raise ValueError(
            "actions must have shape (N, 5), "
            f"but got {tuple(actions.shape)}"
        )

    remapped = actions.clone()

    x_ratio = (
        (
            remapped[:, 0]
            -
            X_MIN
        )
        /
        max(
            float(X_MAX - X_MIN),
            1e-8,
        )
    ).clamp(
        0.0,
        1.0,
    )

    y_ratio = (
        (
            remapped[:, 1]
            -
            Y_MIN
        )
        /
        max(
            float(Y_MAX - Y_MIN),
            1e-8,
        )
    ).clamp(
        0.0,
        1.0,
    )

    remapped[:, 0] = (
        SAFE_PLAN_X_MIN
        +
        x_ratio
        *
        (
            SAFE_PLAN_X_MAX
            -
            SAFE_PLAN_X_MIN
        )
    )

    remapped[:, 1] = (
        SAFE_PLAN_Y_MIN
        +
        y_ratio
        *
        (
            SAFE_PLAN_Y_MAX
            -
            SAFE_PLAN_Y_MIN
        )
    )

    length_ratio = (
        (
            remapped[:, 3]
            -
            LENGTH_MIN
        )
        /
        max(
            float(LENGTH_MAX - LENGTH_MIN),
            1e-8,
        )
    ).clamp(
        0.0,
        1.0,
    )

    depth_ratio = (
        (
            remapped[:, 4]
            -
            DEPTH_MIN
        )
        /
        max(
            float(DEPTH_MAX - DEPTH_MIN),
            1e-8,
        )
    ).clamp(
        0.0,
        1.0,
    )

    remapped[:, 3] = (
        LENGTH_MIN
        +
        length_ratio
        *
        (
            SAFE_PLAN_LENGTH_MAX
            -
            LENGTH_MIN
        )
    )

    remapped[:, 4] = (
        SAFE_PLAN_DEPTH_MIN
        +
        depth_ratio
        *
        (
            SAFE_PLAN_DEPTH_MAX
            -
            SAFE_PLAN_DEPTH_MIN
        )
    )

    return remapped


def clamp_actions_to_planning_range(
    actions,
    repair_endpoints=True,
):
    """
    refinement 결과가 실행 가능한 planning 범위를 벗어나지
    않도록 값을 직접 제한한다.

    remap_actions_to_safe_range()와 달리 기존 값을 다른 범위로
    재분배하지 않고 범위 밖의 값만 clamp한다.
    """

    if actions.ndim != 2 or actions.shape[1] < 5:
        raise ValueError(
            "actions must have shape (N, 5), "
            f"but got {tuple(actions.shape)}"
        )

    constrained = actions.clone()

    constrained[:, 0] = torch.clamp(
        constrained[:, 0],
        SAFE_PLAN_X_MIN,
        SAFE_PLAN_X_MAX,
    )

    constrained[:, 1] = torch.clamp(
        constrained[:, 1],
        SAFE_PLAN_Y_MIN,
        SAFE_PLAN_Y_MAX,
    )

    constrained[:, 2] = wrap_angle(
        constrained[:, 2]
    )

    constrained[:, 3] = torch.clamp(
        constrained[:, 3],
        LENGTH_MIN,
        SAFE_PLAN_LENGTH_MAX,
    )

    constrained[:, 4] = torch.clamp(
        constrained[:, 4],
        SAFE_PLAN_DEPTH_MIN,
        SAFE_PLAN_DEPTH_MAX,
    )

    if repair_endpoints:
        constrained = repair_action_endpoints(
            constrained
        )

    return constrained


@torch.no_grad()
def greedy_plan_gpu(
    model,
    state,
    target,
    horizon=PLAN_HORIZON,
    candidates=GREEDY_CANDIDATES,
    w3d=PLAN_W_3D,
    wviz=PLAN_W_VIZ,
    recent_actions=None,
    candidate_batch_size=1,
):
    """
    메모리 절약형 greedy MPC planner.
    Candidate를 작은 batch로 나누어 평가한다.
    """

    current = ensure_bchw(
        state
    )

    target = ensure_bchw(
        target
    ).to(
        device=current.device,
        dtype=current.dtype,
    )

    horizon = max(
        1,
        int(horizon),
    )

    candidates = max(
        1,
        int(candidates),
    )

    candidate_batch_size = max(
        1,
        min(
            int(candidate_batch_size),
            candidates,
        ),
    )

    history = prepare_recent_actions(
        recent_actions,
        current.device,
        current.dtype,
    )

    plan = []

    for horizon_index in range(
        horizon
    ):

        actions = sample_error_guided_actions_gpu(
            current,
            target,
            candidates,
            device=current.device,
            dtype=current.dtype,
        )

        # 로봇이 안정적으로 실행할 수 있는 action 범위
        # 모델의 원래 정규화 범위는 변경하지 않는다.
        actions = actions.clone()

        actions = remap_actions_to_safe_range(
            actions
        )

        # ====================================================
        # TARGET-GUIDED PRESS POSITION
        #
        # current > target인 곳은 더 눌러야 하는 배경이다.
        # A 글자처럼 target이 현재보다 높은 곳은 제외한다.
        # ====================================================

        press_need = torch.relu(
            current[
                0,
                0
            ]
            -
            target[
                0,
                0
            ]
            -
            0.0005
        )

        press_h = press_need.shape[
            0
        ]

        press_w = press_need.shape[
            1
        ]

        all_pixel_indices = torch.arange(
            press_h
            *
            press_w,
            device=current.device,
        )

        all_world_x, all_world_y = pixel_indices_to_world(
            all_pixel_indices,
            press_h,
            press_w,
            current.device,
            current.dtype,
        )

        # 현재 로봇이 안정적으로 도달하는 press 작업 영역
        safe_position_mask = (
            (
                all_world_x
                >=
                SAFE_PLAN_X_MIN
            )
            &
            (
                all_world_x
                <=
                SAFE_PLAN_X_MAX
            )
            &
            (
                all_world_y
                >=
                SAFE_PLAN_Y_MIN
            )
            &
            (
                all_world_y
                <=
                SAFE_PLAN_Y_MAX
            )
        )

        press_weights = (
            press_need.reshape(
                -1
            )
            *
            safe_position_mask.to(
                dtype=current.dtype
            )
        )

        press_weight_sum = torch.sum(
            press_weights
        )

        if (
            float(
                press_weight_sum.item()
            )
            >
            LOSS_EPS
        ):

            press_weights = (
                press_weights
                /
                press_weight_sum
            )

            selected_pixels = torch.multinomial(
                press_weights,
                actions.shape[
                    0
                ],
                replacement=True,
            )

            selected_x, selected_y = pixel_indices_to_world(
                selected_pixels,
                press_h,
                press_w,
                current.device,
                current.dtype,
            )

            actions = actions.clone()

            actions[
                :,
                0
            ] = selected_x

            actions[
                :,
                1
            ] = selected_y

        actions = clamp_actions_to_planning_range(
            actions
        )

        base_l3d, base_lviz = compute_loss_baseline(
            current,
            target,
        )

        best_loss_value = None
        best_action = None
        best_prediction = None

        for begin in range(
            0,
            candidates,
            candidate_batch_size,
        ):

            chunk_end = min(
                begin
                +
                candidate_batch_size,
                candidates,
            )

            action_chunk = actions[
                begin:chunk_end
            ]

            prediction_chunk = predict_next_batch(
                model,
                current,
                action_chunk,
            )

            loss_chunk = candidate_losses(
                prediction_chunk,
                target,
                w3d=w3d,
                wviz=wviz,
                base_l3d=base_l3d,
                base_lviz=base_lviz,
            )

            loss_chunk = (
                loss_chunk
                +
                repeat_action_penalty(
                    action_chunk,
                    history,
                )
            )

            local_index = torch.argmin(
                loss_chunk
            )

            local_loss_value = float(
                loss_chunk[
                    local_index
                ].item()
            )

            if (
                best_loss_value is None
                or
                local_loss_value
                <
                best_loss_value
            ):

                best_loss_value = local_loss_value

                best_action = (
                    action_chunk[
                        local_index
                    ]
                    .detach()
                    .clone()
                )

                best_prediction = (
                    prediction_chunk[
                        local_index
                        :
                        local_index
                        +
                        1
                    ]
                    .detach()
                    .clone()
                )

            del prediction_chunk
            del loss_chunk

        if (
            best_action is None
            or
            best_prediction is None
        ):

            raise RuntimeError(
                "Greedy planner failed to select an action."
            )

        current = best_prediction

        plan.append(
            best_action
        )

        best_action_batch = best_action[
            None,
            :
        ]

        if history is None:

            history = best_action_batch

        else:

            history = torch.cat(
                [
                    history,
                    best_action_batch,
                ],
                dim=0,
            )

            if (
                history.shape[0]
                >
                RECENT_ACTION_HISTORY
            ):

                history = history[
                    -RECENT_ACTION_HISTORY:
                ]

        print(
            f"[GREEDY {horizon_index + 1:02d}/"
            f"{horizon:02d}] "
            f"loss={best_loss_value:.6f} "
            f"x={float(best_action[0].item()):.4f} "
            f"y={float(best_action[1].item()):.4f} "
            f"theta={float(best_action[2].item()):+.3f} "
            f"length={float(best_action[3].item()) * 1000.0:.1f}mm "
            f"depth={float(best_action[4].item()) * 1000.0:.1f}mm"
        )

        del actions

    return torch.stack(
        plan,
        dim=0,
    )


# ============================================================
# INVERSE SIGMOID
# ============================================================

def inverse_sigmoid(
    x,
):

    x = torch.clamp(

        x,

        1e-5,

        1.0
        -
        1e-5,
    )


    return torch.log(

        x

        /

        (
            1.0
            -
            x
        )
    )


# ============================================================
# BOUNDED ACTION -> RAW
# ============================================================

def encode_actions(
    actions,
):

    raw = torch.zeros_like(
        actions
    )


    raw[
        :,
        0
    ] = inverse_sigmoid(

        (
            actions[
                :,
                0
            ]
            -
            X_MIN
        )

        /

        (
            X_MAX
            -
            X_MIN
        )
    )


    raw[
        :,
        1
    ] = inverse_sigmoid(

        (
            actions[
                :,
                1
            ]
            -
            Y_MIN
        )

        /

        (
            Y_MAX
            -
            Y_MIN
        )
    )


    raw[
        :,
        2
    ] = torch.atanh(

        torch.clamp(

            actions[
                :,
                2
            ]

            /

            math.pi,

            -0.999,

            0.999,
        )
    )


    raw[
        :,
        3
    ] = inverse_sigmoid(

        (
            actions[
                :,
                3
            ]
            -
            LENGTH_MIN
        )

        /

        (
            LENGTH_MAX
            -
            LENGTH_MIN
        )
    )


    raw[
        :,
        4
    ] = inverse_sigmoid(

        (
            actions[
                :,
                4
            ]
            -
            DEPTH_MIN
        )

        /

        (
            DEPTH_MAX
            -
            DEPTH_MIN
        )
    )


    return raw


# ============================================================
# RAW -> BOUNDED ACTION
# ============================================================

def decode_actions(
    raw,
):

    actions = torch.empty_like(
        raw
    )


    actions[
        :,
        0
    ] = (

        X_MIN

        +

        torch.sigmoid(
            raw[
                :,
                0
            ]
        )

        *

        (
            X_MAX
            -
            X_MIN
        )
    )


    actions[
        :,
        1
    ] = (

        Y_MIN

        +

        torch.sigmoid(
            raw[
                :,
                1
            ]
        )

        *

        (
            Y_MAX
            -
            Y_MIN
        )
    )


    actions[
        :,
        2
    ] = (

        math.pi

        *

        torch.tanh(
            raw[
                :,
                2
            ]
        )
    )


    actions[
        :,
        3
    ] = (

        LENGTH_MIN

        +

        torch.sigmoid(
            raw[
                :,
                3
            ]
        )

        *

        (
            LENGTH_MAX
            -
            LENGTH_MIN
        )
    )


    actions[
        :,
        4
    ] = (

        DEPTH_MIN

        +

        torch.sigmoid(
            raw[
                :,
                4
            ]
        )

        *

        (
            DEPTH_MAX
            -
            DEPTH_MIN
        )
    )


    return actions


# ============================================================
# ROLLOUT
# ============================================================

def rollout(
    model,
    state,
    actions,
):

    current = ensure_bchw(
        state
    )


    for i in range(

        actions.shape[
            0
        ]
    ):

        current = predict_next(

            model,

            current,

            actions[
                i
            ],
        )


    return current


# ============================================================
# PLAN INTERNAL REPEAT PENALTY
#
# refine 단계에서 40개 action이 같은 곳으로
# 다시 몰리는 현상 방지.
# ============================================================

def plan_spread_penalty(
    actions,
):

    n = actions.shape[
        0
    ]


    if n <= 1:

        return torch.zeros(

            (),

            device=(
                actions.device
            ),

            dtype=(
                actions.dtype
            ),
        )


    xy = actions[
        :,
        0:2
    ]


    distances = torch.cdist(

        xy,

        xy,
    )


    eye = torch.eye(

        n,

        device=(
            actions.device
        ),

        dtype=torch.bool,
    )


    valid = ~eye


    d = distances[
        valid
    ]


    penalty = torch.exp(

        -0.5

        *

        (
            d

            /

            REPEAT_RADIUS
        )
        **
        2
    )


    return (

        REPEAT_PENALTY

        *

        torch.mean(
            penalty
        )
    )


# ============================================================
# HISTORY PENALTY FOR WHOLE PLAN
# ============================================================

def plan_history_penalty(
    actions,
    recent_actions,
):

    if recent_actions is None:

        return torch.zeros(

            (),

            device=(
                actions.device
            ),

            dtype=(
                actions.dtype
            ),
        )


    penalties = repeat_action_penalty(

        actions,

        recent_actions,
    )


    return torch.mean(
        penalties
    )


# ============================================================
# ENDPOINT PENALTY
#
# gradient refinement에서 endpoint가 workspace를
# 빠져나가지 못하게 한다.
# ============================================================

def endpoint_penalty(
    actions,
):

    end_x = (

        actions[
            :,
            0
        ]

        +

        actions[
            :,
            3
        ]

        *

        torch.cos(
            actions[
                :,
                2
            ]
        )
    )


    end_y = (

        actions[
            :,
            1
        ]

        +

        actions[
            :,
            3
        ]

        *

        torch.sin(
            actions[
                :,
                2
            ]
        )
    )


    px = (

        torch.relu(
            X_MIN
            -
            end_x
        )

        +

        torch.relu(
            end_x
            -
            X_MAX
        )
    )


    py = (

        torch.relu(
            Y_MIN
            -
            end_y
        )

        +

        torch.relu(
            end_y
            -
            Y_MAX
        )
    )


    scale = max(

        X_MAX
        -
        X_MIN,

        Y_MAX
        -
        Y_MIN,
    )


    return torch.mean(

        (
            px
            +
            py
        )

        /

        scale
    )


# ============================================================
# GPU AUTOGRAD REFINEMENT
#
# 이전:
#
# raw L3D + raw Lviz
#
# 변경:
#
# normalized L3D
# +
# normalized Lviz
# +
# repeated action penalty
# +
# endpoint penalty
# ============================================================

def refinement_objective(
    model,
    current,
    target,
    action,
    base_l3d,
    base_lviz,
    w3d,
    wviz,
    history,
):
    """단일 refinement action의 전체 목적함수와 예측 상태."""

    predicted = predict_next_batch(
        model,
        current,
        action,
    )

    l3d, lviz = loss_components(
        predicted,
        target,
    )

    normalized_l3d = (
        torch.mean(l3d)
        /
        base_l3d
    )

    normalized_lviz = (
        torch.mean(lviz)
        /
        base_lviz
    )

    target_loss = (
        w3d
        *
        normalized_l3d
        +
        wviz
        *
        normalized_lviz
    )

    history_loss = plan_history_penalty(
        action,
        history,
    )

    boundary_loss = endpoint_penalty(
        action
    )

    loss = (
        target_loss
        +
        history_loss
        +
        2.0
        *
        boundary_loss
    )

    return loss, predicted

def refine_plan_gpu(
    model,
    state,
    target,
    initial_plan,
    w3d=PLAN_W_3D,
    wviz=PLAN_W_VIZ,
    recent_actions=None,
    max_refine_actions=5,
    refine_steps=REFINE_STEPS,
):
    """
    메모리 절약형 differentiable refinement.

    앞쪽 action을 한 개씩 refinement하며, refinement 결과가
    원래 greedy action보다 나빠지면 원래 action으로 되돌린다.
    모든 결과는 planning 실행 범위 안으로 제한한다.
    """

    state = ensure_bchw(state)
    target = ensure_bchw(target).to(
        device=state.device,
        dtype=state.dtype,
    )

    refined_plan = clamp_actions_to_planning_range(
        initial_plan.detach().clone()
    )

    refine_steps = max(0, int(refine_steps))
    max_refine_actions = max(0, int(max_refine_actions))

    if refine_steps == 0 or max_refine_actions == 0:
        return refined_plan

    refine_count = min(
        refined_plan.shape[0],
        max_refine_actions,
    )

    current = state.detach()
    history = prepare_recent_actions(
        recent_actions,
        state.device,
        state.dtype,
    )

    for action_index in range(refine_count):
        with torch.no_grad():
            base_l3d, base_lviz = compute_loss_baseline(
                current,
                target,
            )

            original_action = refined_plan[
                action_index:action_index + 1
            ].detach().clone()

            (
                original_loss,
                original_prediction,
            ) = refinement_objective(
                model,
                current,
                target,
                original_action,
                base_l3d,
                base_lviz,
                w3d,
                wviz,
                history,
            )

            original_loss_value = float(
                original_loss.item()
            )

        raw = encode_actions(
            original_action
        ).detach()
        raw.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [raw],
            lr=PLAN_LR,
        )

        for _ in range(refine_steps):
            # repair_action_endpoints()의 indexed in-place 연산은
            # gradient 경로를 불안정하게 만들 수 있으므로 최적화
            # 중에는 값 범위만 제한하고 endpoint는 penalty로 다룬다.
            action = clamp_actions_to_planning_range(
                decode_actions(raw),
                repair_endpoints=False,
            )

            loss, predicted = refinement_objective(
                model,
                current,
                target,
                action,
                base_l3d,
                base_lviz,
                w3d,
                wviz,
                history,
            )

            optimizer.zero_grad(set_to_none=True)

            raw_gradient = torch.autograd.grad(
                loss,
                raw,
                retain_graph=False,
                create_graph=False,
            )[0]

            raw.grad = raw_gradient

            torch.nn.utils.clip_grad_norm_(
                [raw],
                max_norm=5.0,
            )

            optimizer.step()

            del predicted
            del loss
            del raw_gradient

        refined_action = clamp_actions_to_planning_range(
            decode_actions(raw).detach()
        )

        with torch.no_grad():
            (
                refined_loss,
                refined_prediction,
            ) = refinement_objective(
                model,
                current,
                target,
                refined_action,
                base_l3d,
                base_lviz,
                w3d,
                wviz,
                history,
            )

            refined_loss_value = float(
                refined_loss.item()
            )

        # refinement가 greedy action을 악화시키면 원래 action 유지
        if refined_loss_value <= original_loss_value:
            selected_action = refined_action
            selected_prediction = refined_prediction.detach()
            final_loss_value = refined_loss_value
            refine_status = "accepted"
        else:
            selected_action = original_action
            selected_prediction = original_prediction.detach()
            final_loss_value = original_loss_value
            refine_status = "reverted"

        refined_plan[
            action_index:action_index + 1
        ] = selected_action

        current = selected_prediction

        if history is None:
            history = selected_action
        else:
            history = torch.cat(
                [history, selected_action],
                dim=0,
            )

            if history.shape[0] > RECENT_ACTION_HISTORY:
                history = history[-RECENT_ACTION_HISTORY:]

        print(
            f"[REFINE {action_index + 1:02d}/"
            f"{refine_count:02d}] "
            f"loss={final_loss_value:.6f} "
            f"status={refine_status}"
        )

        del raw
        del optimizer

    return clamp_actions_to_planning_range(
        refined_plan
    )


# ============================================================
# NORMALIZED ACTION NOISE
# ============================================================

def add_normalized_action_noise(
    actions,
    sigma,
):

    raw = encode_actions(
        actions
    )


    noisy = (

        raw

        +

        torch.randn_like(
            raw
        )

        *

        sigma
    )


    return decode_actions(
        noisy
    )