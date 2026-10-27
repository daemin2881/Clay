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
):

    current = ensure_bchw(
        state
    )


    target = ensure_bchw(
        target
    )


    history = prepare_recent_actions(

        recent_actions,

        current.device,

        current.dtype,
    )


    plan = []


    for _ in range(
        horizon
    ):

        # ====================================================
        # ERROR-GUIDED + RANDOM CANDIDATES
        # ====================================================

        actions = sample_error_guided_actions_gpu(

            current,

            target,

            candidates,

            device=(
                current.device
            ),

            dtype=(
                current.dtype
            ),
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        predictions = (
            predict_next_batch(

                model,

                current,

                actions,
            )
        )


        # ====================================================
        # CURRENT LOSS BASELINE
        # ====================================================

        base_l3d, base_lviz = (
            compute_loss_baseline(

                current,

                target,
            )
        )


        # ====================================================
        # NORMALIZED TARGET LOSS
        # ====================================================

        losses = candidate_losses(

            predictions,

            target,

            w3d=(
                w3d
            ),

            wviz=(
                wviz
            ),

            base_l3d=(
                base_l3d
            ),

            base_lviz=(
                base_lviz
            ),
        )


        # ====================================================
        # SAME-AREA REPEAT PENALTY
        # ====================================================

        losses = (

            losses

            +

            repeat_action_penalty(

                actions,

                history,
            )
        )


        # ====================================================
        # BEST
        # ====================================================

        best_idx = torch.argmin(
            losses
        )


        best_action = actions[
            best_idx
        ]


        current = predictions[

            best_idx
            :
            best_idx
            +
            1
        ]


        plan.append(
            best_action
        )


        # ====================================================
        # UPDATE INTERNAL PLAN HISTORY
        #
        # greedy horizon 안에서도 동일 지점만
        # 계속 고르는 것을 방지한다.
        # ====================================================

        best_action_batch = best_action[
            None,
            :
        ]


        if history is None:

            history = (
                best_action_batch
            )


        else:

            history = torch.cat(

                [
                    history,
                    best_action_batch,
                ],

                dim=0,
            )


            if history.shape[
                0
            ] > RECENT_ACTION_HISTORY:

                history = history[

                    -RECENT_ACTION_HISTORY:
                ]


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

def refine_plan_gpu(
    model,
    state,
    target,
    initial_plan,
    w3d=PLAN_W_3D,
    wviz=PLAN_W_VIZ,
    recent_actions=None,
):

    state = ensure_bchw(
        state
    )


    target = ensure_bchw(
        target
    )


    recent = prepare_recent_actions(

        recent_actions,

        state.device,

        state.dtype,
    )


    # --------------------------------------------------------
    # fixed baseline from the current physical state.
    # --------------------------------------------------------

    with torch.no_grad():

        base_l3d, base_lviz = (
            compute_loss_baseline(

                state,

                target,
            )
        )


    raw = encode_actions(

        initial_plan
    ).detach()


    raw.requires_grad_(
        True
    )


    optimizer = torch.optim.Adam(

        [
            raw
        ],

        lr=(
            PLAN_LR
        ),
    )


    for _ in range(
        REFINE_STEPS
    ):

        actions = decode_actions(
            raw
        )


        predicted = rollout(

            model,

            state,

            actions,
        )


        l3d, lviz = loss_components(

            predicted,

            target,
        )


        normalized_l3d = (

            torch.mean(
                l3d
            )

            /

            base_l3d
        )


        normalized_lviz = (

            torch.mean(
                lviz
            )

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


        spread_loss = (
            plan_spread_penalty(
                actions
            )
        )


        history_loss = (
            plan_history_penalty(

                actions,

                recent,
            )
        )


        boundary_loss = (
            endpoint_penalty(
                actions
            )
        )


        loss = (

            target_loss

            +

            spread_loss

            +

            history_loss

            +

            2.0
            *
            boundary_loss
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            [
                raw
            ],

            max_norm=5.0,
        )


        optimizer.step()


    actions = decode_actions(
        raw
    ).detach()


    # --------------------------------------------------------
    # final GPU safety correction
    # --------------------------------------------------------

    actions = repair_action_endpoints(
        actions
    )


    return actions


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