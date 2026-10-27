import os
import csv

import numpy as np
import torch

from PIL import Image

from vs_representation import (
    spatial_gradient,
)


# ============================================================
# DIRECTORY
# ============================================================

def ensure_dir(
    path,
):

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path


# ============================================================
# RAW TENSOR
# ============================================================

def save_tensor(
    tensor,
    path,
):

    # INTENTIONAL GPU -> CPU
    # experiment output only

    torch.save(

        tensor
        .detach()
        .float()
        .cpu(),

        path,
    )


# ============================================================
# GRAYSCALE
# ============================================================

def save_gray_png(
    tensor,
    path,
    vmin=None,
    vmax=None,
):

    x = (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    x = np.squeeze(
        x
    )


    if vmin is None:

        vmin = float(
            np.nanmin(
                x
            )
        )


    if vmax is None:

        vmax = float(
            np.nanmax(
                x
            )
        )


    denom = max(
        vmax - vmin,
        1e-8,
    )


    image = (

        (
            x
            -
            vmin
        )

        /

        denom
    )


    image = np.clip(

        image,

        0.0,

        1.0,
    )


    image = (

        image

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
# SIGNED DELTA IMAGE
#
# red  : indentation
# blue : displaced/up material
#
# Fig.7-like visualization
# ============================================================

def save_delta_png(
    before,
    after,
    path,
):

    delta = (

        after

        -

        before
    )


    d = (
        delta
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    d = np.squeeze(
        d
    )


    scale = max(

        float(
            np.max(
                np.abs(
                    d
                )
            )
        ),

        1e-8,
    )


    rgb = np.ones(
        (
            d.shape[
                0
            ],
            d.shape[
                1
            ],
            3,
        ),
        dtype=np.float32,
    )


    # negative = indentation = red

    neg = np.clip(

        -d
        /
        scale,

        0.0,

        1.0,
    )


    # positive = displaced material = blue

    pos = np.clip(

        d
        /
        scale,

        0.0,

        1.0,
    )


    rgb[
        ...,
        1
    ] -= np.maximum(
        neg,
        pos,
    )


    rgb[
        ...,
        2
    ] -= neg


    rgb[
        ...,
        0
    ] -= pos


    rgb = np.clip(

        rgb,

        0.0,

        1.0,
    )


    Image.fromarray(

        (
            rgb
            *
            255
        ).astype(
            np.uint8
        )

    ).save(
        path
    )


# ============================================================
# GRADIENT VISUALIZATION
#
# green = |dx|
# blue  = |dy|
#
# Fig.5-like
# ============================================================

def save_gradient_png(
    height,
    path,
):

    h = height


    if h.ndim == 2:

        h = h[
            None,
            None
        ]


    elif h.ndim == 3:

        h = h[
            None
        ]


    gx, gy = spatial_gradient(
        h
    )


    gx = torch.abs(
        gx[
            0,
            0
        ]
    )


    gy = torch.abs(
        gy[
            0,
            0
        ]
    )


    scale = torch.maximum(

        gx.max(),

        gy.max(),
    )


    scale = torch.clamp(

        scale,

        min=1e-8,
    )


    g = (

        gx
        /
        scale
    )


    b = (

        gy
        /
        scale
    )


    r = torch.zeros_like(
        g
    )


    rgb = torch.stack(

        [
            r,
            g,
            b,
        ],

        dim=-1,
    )


    image = (

        rgb
        .detach()
        .cpu()
        .numpy()
    )


    Image.fromarray(

        (
            image
            *
            255
        ).astype(
            np.uint8
        )

    ).save(
        path
    )


# ============================================================
# STATE BUNDLE
# ============================================================

def save_state_bundle(
    height,
    directory,
    prefix,
):

    ensure_dir(
        directory
    )


    save_tensor(

        height,

        os.path.join(
            directory,
            f"{prefix}.pt",
        ),
    )


    save_gray_png(

        height,

        os.path.join(
            directory,
            f"{prefix}_depth.png",
        ),
    )


    save_gradient_png(

        height,

        os.path.join(
            directory,
            f"{prefix}_gradient.png",
        ),
    )