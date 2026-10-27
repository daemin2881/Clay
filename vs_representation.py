import torch
import torch.nn.functional as F

from vs_config import (
    MAP_H,
    MAP_W,
    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_WIDTH,
    CLAY_LENGTH,
    CLAY_BOTTOM_Z,
    CLAY_TOP_Z,
    PIXEL_SIZE_X,
    PIXEL_SIZE_Y,
    TARGET_EDGE_FRACTION,
)


# ============================================================
# SOBEL
# ============================================================

_SOBEL_CACHE = {}


def get_sobel(
    device,
    dtype,
):

    key = (
        str(device),
        str(dtype),
    )

    if key in _SOBEL_CACHE:
        return _SOBEL_CACHE[
            key
        ]

    kx = torch.tensor(
        [[[
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    ky = torch.tensor(
        [[[
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    kx = (
        kx
        /
        (
            8.0
            *
            PIXEL_SIZE_X
        )
    )

    ky = (
        ky
        /
        (
            8.0
            *
            PIXEL_SIZE_Y
        )
    )

    _SOBEL_CACHE[
        key
    ] = (
        kx,
        ky,
    )

    return (
        kx,
        ky,
    )


def ensure_bchw(
    x,
):

    if x.ndim == 2:
        x = x[
            None,
            None
        ]

    elif x.ndim == 3:
        x = x[
            :,
            None
        ]

    return x


def spatial_gradient(
    height,
):

    height = ensure_bchw(
        height
    )

    kx, ky = get_sobel(
        height.device,
        height.dtype,
    )

    gx = F.conv2d(
        height,
        kx,
        padding=1,
    )

    gy = F.conv2d(
        height,
        ky,
        padding=1,
    )

    return (
        gx,
        gy,
    )


def make_state_representation(
    height,
):

    height = ensure_bchw(
        height
    )

    gx, gy = spatial_gradient(
        height
    )

    return torch.cat(
        [
            height,
            gx,
            gy,
        ],
        dim=1,
    )


# ============================================================
# WORLD GRID CACHE
# ============================================================

_WORLD_GRID_CACHE = {}


def world_grid(
    device,
    dtype,
):

    key = (
        str(device),
        str(dtype),
    )

    if key in _WORLD_GRID_CACHE:
        return _WORLD_GRID_CACHE[
            key
        ]

    xs = torch.linspace(
        CLAY_CENTER_X
        -
        CLAY_WIDTH / 2.0,
        CLAY_CENTER_X
        +
        CLAY_WIDTH / 2.0,
        MAP_W,
        device=device,
        dtype=dtype,
    )

    ys = torch.linspace(
        CLAY_CENTER_Y
        -
        CLAY_LENGTH / 2.0,
        CLAY_CENTER_Y
        +
        CLAY_LENGTH / 2.0,
        MAP_H,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    _WORLD_GRID_CACHE[
        key
    ] = (
        xx,
        yy,
    )

    return (
        xx,
        yy,
    )


def world_x_to_norm(
    x,
):

    x0 = (
        CLAY_CENTER_X
        -
        CLAY_WIDTH / 2.0
    )

    return (
        2.0
        *
        (x - x0)
        /
        CLAY_WIDTH
        -
        1.0
    )


def world_y_to_norm(
    y,
):

    y0 = (
        CLAY_CENTER_Y
        -
        CLAY_LENGTH / 2.0
    )

    return (
        2.0
        *
        (y - y0)
        /
        CLAY_LENGTH
        -
        1.0
    )


# ============================================================
# ACTUAL -> CANONICAL
# ============================================================

def actual_to_canonical(
    image,
    x,
    y,
    theta,
):

    image = ensure_bchw(
        image
    )

    B = image.shape[0]

    device = image.device

    dtype = image.dtype

    xx, yy = world_grid(
        device,
        dtype,
    )

    xx = xx[
        None
    ].expand(
        B,
        -1,
        -1,
    )

    yy = yy[
        None
    ].expand(
        B,
        -1,
        -1,
    )

    x = torch.as_tensor(
        x,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    y = torch.as_tensor(
        y,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    theta = torch.as_tensor(
        theta,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    dx = (
        xx
        -
        CLAY_CENTER_X
    )

    dy = (
        yy
        -
        CLAY_CENTER_Y
    )

    c = torch.cos(
        theta
    )

    s = torch.sin(
        theta
    )

    actual_x = (
        x
        +
        c * dx
        -
        s * dy
    )

    actual_y = (
        y
        +
        s * dx
        +
        c * dy
    )

    grid = torch.stack(
        [
            world_x_to_norm(
                actual_x
            ),
            world_y_to_norm(
                actual_y
            ),
        ],
        dim=-1,
    )

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ============================================================
# CANONICAL -> ACTUAL
# ============================================================

def canonical_to_actual(
    image,
    x,
    y,
    theta,
):

    image = ensure_bchw(
        image
    )

    B = image.shape[0]

    device = image.device

    dtype = image.dtype

    xx, yy = world_grid(
        device,
        dtype,
    )

    xx = xx[
        None
    ].expand(
        B,
        -1,
        -1,
    )

    yy = yy[
        None
    ].expand(
        B,
        -1,
        -1,
    )

    x = torch.as_tensor(
        x,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    y = torch.as_tensor(
        y,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    theta = torch.as_tensor(
        theta,
        device=device,
        dtype=dtype,
    ).reshape(
        B,
        1,
        1,
    )

    dx = (
        xx
        -
        x
    )

    dy = (
        yy
        -
        y
    )

    c = torch.cos(
        theta
    )

    s = torch.sin(
        theta
    )

    canonical_x = (
        CLAY_CENTER_X
        +
        c * dx
        +
        s * dy
    )

    canonical_y = (
        CLAY_CENTER_Y
        -
        s * dx
        +
        c * dy
    )

    grid = torch.stack(
        [
            world_x_to_norm(
                canonical_x
            ),
            world_y_to_norm(
                canonical_y
            ),
        ],
        dim=-1,
    )

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


# ============================================================
# EDGE MASK
# ============================================================

_EDGE_MASK_CACHE = {}


def edge_mask(
    device,
):

    key = str(
        device
    )

    if key in _EDGE_MASK_CACHE:
        return _EDGE_MASK_CACHE[
            key
        ]

    bx = max(
        1,
        int(
            MAP_W
            *
            TARGET_EDGE_FRACTION
        )
    )

    by = max(
        1,
        int(
            MAP_H
            *
            TARGET_EDGE_FRACTION
        )
    )

    mask = torch.zeros(
        (
            MAP_H,
            MAP_W,
        ),
        device=device,
        dtype=torch.bool,
    )

    mask[
        :by,
        :
    ] = True

    mask[
        -by:,
        :
    ] = True

    mask[
        :,
        :bx
    ] = True

    mask[
        :,
        -bx:
    ] = True

    _EDGE_MASK_CACHE[
        key
    ] = mask

    return mask


# ============================================================
# GPU TARGET ADJUSTMENT
#
# 논문 Eq.(3)(4)에 대응:
#
# - edge = current
# - material amount 맞춤
# - table/height constraint
# - alpha를 가능한 크게
# ============================================================

def adjust_target_to_state_gpu(
    current,
    raw_target,
    iterations=150,
):

    if current.ndim == 4:
        current = current[
            0,
            0
        ]

    elif current.ndim == 3:
        current = current[
            0
        ]

    if raw_target.ndim == 4:
        raw_target = raw_target[
            0,
            0
        ]

    elif raw_target.ndim == 3:
        raw_target = raw_target[
            0
        ]

    device = current.device

    dtype = current.dtype

    M = edge_mask(
        device
    )

    inside = (
        ~M
    )

    alpha_raw = torch.zeros(
        (),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    beta = torch.zeros(
        (),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam(
        [
            alpha_raw,
            beta,
        ],
        lr=0.03,
    )

    current_sum = current.sum().detach()

    for _ in range(
        iterations
    ):

        alpha = (
            0.1
            +
            2.9
            *
            torch.sigmoid(
                alpha_raw
            )
        )

        candidate_inside = (
            alpha
            *
            raw_target
            +
            beta
        )

        candidate = torch.where(
            M,
            current,
            candidate_inside,
        )

        material_loss = (
            (
                candidate.sum()
                -
                current_sum
            )
            /
            current.numel()
        ) ** 2

        below_loss = torch.mean(
            torch.relu(
                CLAY_BOTTOM_Z
                -
                candidate
            ) ** 2
        )

        above_loss = torch.mean(
            torch.relu(
                candidate
                -
                (
                    CLAY_TOP_Z
                    +
                    0.015
                )
            ) ** 2
        )

        sharpness_reward = (
            -1e-4
            *
            alpha
        )

        loss = (
            1000.0
            *
            material_loss
            +
            1000.0
            *
            below_loss
            +
            1000.0
            *
            above_loss
            +
            sharpness_reward
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

    with torch.no_grad():

        alpha = (
            0.1
            +
            2.9
            *
            torch.sigmoid(
                alpha_raw
            )
        )

        adjusted = (
            alpha
            *
            raw_target
            +
            beta
        )

        adjusted = torch.where(
            M,
            current,
            adjusted,
        )

        adjusted = torch.clamp(
            adjusted,
            CLAY_BOTTOM_Z,
            CLAY_TOP_Z
            +
            0.015,
        )

    return adjusted