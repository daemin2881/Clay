import os

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
)

from vs_config import (
    MAP_H,
    MAP_W,

    CLAY_BOTTOM_Z,
    CLAY_TOP_Z,

    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_SIZE_X,
    CLAY_SIZE_Y,

    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
)


# ============================================================
# TARGET WORKSPACE
#
# planner의 실제 안전 실행 범위와 맞춘다.
# ============================================================

TARGET_X_MIN = max(
    0.380,
    X_MIN,
)

TARGET_X_MAX = min(
    0.560,
    X_MAX,
)

TARGET_Y_MIN = max(
    -0.110,
    Y_MIN,
)

TARGET_Y_MAX = min(
    0.110,
    Y_MAX,
)


# ============================================================
# FONT
# ============================================================

# 논문은 정확한 폰트 이름 대신 serif font라고 명시한다.
# Fig. 2와 형태가 가까운 Times 계열 Liberation Serif Bold를
# 가장 먼저 사용한다.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def get_font(
    size,
):

    for path in FONT_CANDIDATES:

        if os.path.isfile(
            path
        ):

            return ImageFont.truetype(
                path,
                size,
            )

    return ImageFont.load_default()


# ============================================================
# MASK -> RAISED RELIEF HEIGHT
#
# mask가 높은 곳이 글자/형상 위쪽이 된다.
#
# 글자를 올린 만큼 배경을 조금 낮춰 전체 평균 높이,
# 즉 재료량이 크게 변하지 않도록 한다.
# ============================================================

def mask_to_relief_height(
    mask,
    relief_height,
    device,
    current=None,
):

    profile = torch.as_tensor(
        mask,
        device=device,
        dtype=torch.float32,
    )

    profile = torch.clamp(
        profile,
        0.0,
        1.0,
    )

    # 글자 윗면을 넓히고 경계를 부드럽게 유지
    profile = torch.pow(
        profile,
        0.70,
    )

    # ========================================================
    # RAW TARGET ONLY
    #
    # 여기서는 재료량을 맞추거나 배경을 낮추지 않는다.
    #
    # 논문식 alpha / beta target adjustment와
    # material conservation은
    # adjust_target_to_state_gpu()가 담당한다.
    # ========================================================

    target = (
        CLAY_TOP_Z
        +
        relief_height
        *
        profile
    )

    target = torch.clamp(
        target,
        min=CLAY_BOTTOM_Z,
        max=(
            CLAY_TOP_Z
            +
            0.015
        ),
    )

    return target


# ============================================================
# LEGACY INDENTED HEIGHT
#
# 필요한 경우 음각 비교 실험에 사용할 수 있다.
# 현재 알파벳, X, 하트 타겟에서는 사용하지 않는다.
# ============================================================

def mask_to_indented_height(
    mask,
    depth,
    device,
):

    mask = torch.as_tensor(
        mask,
        device=device,
        dtype=torch.float32,
    )

    return (
        CLAY_TOP_Z
        -
        depth
        *
        mask
    )


# ============================================================
# LETTER MASK
# ============================================================

def make_letter_mask(
    letter,
):

    letter = letter.upper()

    image = Image.new(
        "L",
        (
            MAP_W,
            MAP_H,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        image
    )

    # 기존 0.68 -> 0.50
    font = get_font(
        int(
            MAP_H
            *
            0.55
        )
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        letter,
        font=font,
    )

    # ========================================================
    # SAFE WORKSPACE CENTER
    # ========================================================

    target_center_x = (
        TARGET_X_MIN
        +
        TARGET_X_MAX
    ) / 2.0

    target_center_y = (
        TARGET_Y_MIN
        +
        TARGET_Y_MAX
    ) / 2.0

    clay_min_x = (
        CLAY_CENTER_X
        -
        CLAY_SIZE_X / 2.0
    )

    clay_min_y = (
        CLAY_CENTER_Y
        -
        CLAY_SIZE_Y / 2.0
    )

    center_col = (
        (
            target_center_x
            -
            clay_min_x
        )
        /
        CLAY_SIZE_X
        *
        (
            MAP_W - 1
        )
    )

    center_row = (
        (
            target_center_y
            -
            clay_min_y
        )
        /
        CLAY_SIZE_Y
        *
        (
            MAP_H - 1
        )
    )

    bbox_center_x = (
        bbox[0]
        +
        bbox[2]
    ) / 2.0

    bbox_center_y = (
        bbox[1]
        +
        bbox[3]
    ) / 2.0

    x = int(
        round(
            center_col
            -
            bbox_center_x
        )
    )

    y = int(
        round(
            center_row
            -
            bbox_center_y
        )
    )

    draw.text(
        (
            x,
            y,
        ),
        letter,
        fill=255,
        font=font,
    )

    blur_radius = max(
        2.0,
        MAP_W
        *
        0.010,
    )

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=blur_radius
        )
    )

    mask = (
        np.asarray(
            image
        )
        .astype(
            np.float32
        )
        /
        255.0
    )

    return mask


# ============================================================
# LETTER RELIEF TARGET
#
# 논문 Fig. 2의 A -> F 부조 목표
#
# 글자 내부는 높고, 글자 바깥 배경은 약간 낮다.
# ============================================================

def make_letter_target(
    letter,
    device,
    depth=0.008,
    current=None,
):

    mask = make_letter_mask(
        letter
    )

    return mask_to_relief_height(
        mask,
        relief_height=depth,
        device=device,
        current=current,
    )


# ============================================================
# LARGE X MASK
#
# 논문 Fig. 11의 큰 X relief sculpture
# ============================================================

def make_x_mask():

    image = Image.new(
        "L",
        (
            MAP_W,
            MAP_H,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        image
    )

    margin = int(
        MAP_W
        *
        0.20
    )

    line_width = max(
        2,
        int(
            MAP_W
            *
            0.08
        ),
    )

    draw.line(
        (
            margin,
            margin,
            MAP_W - margin,
            MAP_H - margin,
        ),
        fill=255,
        width=line_width,
    )

    draw.line(
        (
            MAP_W - margin,
            margin,
            margin,
            MAP_H - margin,
        ),
        fill=255,
        width=line_width,
    )

    blur_radius = max(
        2.0,
        MAP_W
        *
        0.010,
    )

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=blur_radius
        )
    )

    mask = (
        np.asarray(
            image
        )
        .astype(
            np.float32
        )
        /
        255.0
    )

    return mask


# ============================================================
# LARGE X RELIEF TARGET
# ============================================================

def make_x_target(
    device,
    depth=0.008,
    current=None,
):

    mask = make_x_mask()

    return mask_to_relief_height(
        mask,
        relief_height=depth,
        device=device,
        current=current,
    )


# ============================================================
# HEART MASK
# ============================================================

def make_heart_mask(
    device,
):

    ys = torch.linspace(
        -1.4,
        1.4,
        MAP_H,
        device=device,
        dtype=torch.float32,
    )

    xs = torch.linspace(
        -1.4,
        1.4,
        MAP_W,
        device=device,
        dtype=torch.float32,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    equation = (
        (
            xx
            *
            xx
            +
            yy
            *
            yy
            -
            1.0
        ) ** 3
        -
        xx
        *
        xx
        *
        yy
        *
        yy
        *
        yy
    )

    mask = (
        equation
        <=
        0.0
    ).float()

    # 하트 경계를 부드럽게 만든다.
    mask = mask[
        None,
        None
    ]

    for _ in range(
        4
    ):

        mask = torch.nn.functional.avg_pool2d(
            mask,
            kernel_size=5,
            stride=1,
            padding=2,
        )

    return mask[
        0,
        0
    ]


# ============================================================
# HEART RELIEF TARGET
# ============================================================

def make_heart_target(
    device,
    depth=0.008,
    current=None,
):

    mask = make_heart_mask(
        device
    )

    return mask_to_relief_height(
        mask,
        relief_height=depth,
        device=device,
        current=current,
    )


# ============================================================
# FLAT TARGET
#
# 논문 Fig. 12 smoothing experiment
# ============================================================

def make_flat_target(
    current,
):

    mean = torch.mean(
        current
    )

    return torch.full_like(
        current,
        mean,
    )


# ============================================================
# TARGET ROUTER
# ============================================================

def make_target(
    name,
    device,
    depth=0.008,
    current=None,
):

    lower = name.lower()

    if lower == "heart":

        return make_heart_target(
            device,
            depth,
            current=current,
        )

    if lower == "x":

        return make_x_target(
            device,
            depth,
            current=current,
        )

    if (
        len(name)
        ==
        1
        and
        name.upper()
        in
        "ABCDEF"
    ):

        return make_letter_target(
            name,
            device,
            depth,
            current=current,
        )

    raise ValueError(
        f"Unknown target: {name}"
    )

