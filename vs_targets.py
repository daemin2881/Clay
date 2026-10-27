import os

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

from vs_config import (
    MAP_H,
    MAP_W,
    CLAY_TOP_Z,
)


# ============================================================
# FONT
# ============================================================

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
# MASK -> HEIGHT
# ============================================================

def mask_to_height(
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
# LETTER TARGET
#
# 논문 Fig.2 A→F serif letters
# ============================================================

def make_letter_target(
    letter,
    device,
    depth=0.008,
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


    font = get_font(

        int(
            MAP_H
            *
            0.68
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


    width = (

        bbox[
            2
        ]

        -

        bbox[
            0
        ]
    )


    height = (

        bbox[
            3
        ]

        -

        bbox[
            1
        ]
    )


    x = (

        MAP_W

        -

        width

    ) // 2


    y = (

        MAP_H

        -

        height

    ) // 2 - bbox[
        1
    ]


    draw.text(

        (
            x,
            y,
        ),

        letter,

        fill=255,

        font=font,
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


    return mask_to_height(

        mask,

        depth,

        device,
    )


# ============================================================
# LARGE X
#
# Fig.11
# ============================================================

def make_x_target(
    device,
    depth=0.008,
):

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


    return mask_to_height(

        mask,

        depth,

        device,
    )


# ============================================================
# HEART
# ============================================================

def make_heart_target(
    device,
    depth=0.008,
):

    ys = torch.linspace(

        -1.4,

        1.4,

        MAP_H,

        device=device,
    )


    xs = torch.linspace(

        -1.4,

        1.4,

        MAP_W,

        device=device,
    )


    yy, xx = torch.meshgrid(

        ys,

        xs,

        indexing="ij",
    )


    equation = (

        (
            xx * xx

            +

            yy * yy

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


    return (

        CLAY_TOP_Z

        -

        depth
        *
        mask
    )


# ============================================================
# FLAT TARGET
#
# Fig.12 smoothing experiment
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
):

    lower = name.lower()


    if lower == "heart":

        return make_heart_target(
            device,
            depth,
        )


    if lower == "x":

        return make_x_target(
            device,
            depth,
        )


    if (
        len(
            name
        )
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
        )


    raise ValueError(
        f"Unknown target: {name}"
    )