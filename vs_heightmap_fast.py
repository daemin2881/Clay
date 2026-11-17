"""입자 -> 512x512 높이맵 변환의 빠른 버전 (vs_simulator.particles_to_height_gpu 와 같은 결과).
원본은 (2r+1)^2 = 441 번의 오프셋 반복마다 GPU 동기화(.any())가 있어서 한 번 호출에 수 초가 걸린다(프로파일: 푸시 1개 재생 60초 중 26초가 이 함수).
이 파일의 함수는 원본 소스에서 이중 반복문만 묶음 계산으로 바꾼 것이다(나머지는 원본 그대로). 수식이 같고 amax 가 순서에 무관해서 결과가 같아야 하며,
height_fast_check.py 로 실제 입자 위치에서 원본과 비트 단위로 같은지, 얼마나 빠른지 검증한 뒤에만 쓴다.
"""
import math

import torch
import torch.nn.functional as F

from vs_config import (CLAY_BOTTOM_Z, CLAY_CENTER_X, CLAY_CENTER_Y, CLAY_SIZE_X, CLAY_SIZE_Y, CLAY_TOP_Z, HEIGHT_MAP_RES, MPM_PARTICLE_SIZE)

_CHUNK_ENTRIES = 2_000_000      # 한 묶음에서 다루는 (입자, 오프셋) 쌍의 수 상한(일시 메모리 약 200MB). 워커 여러 개가 동시에 부르므로 작게 잡는다


@torch.no_grad()
def particles_to_height_gpu_fast(
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


    # ---- 속도 개선(결과는 원본과 비트 단위로 같아야 함): (2r+1)^2 번의 오프셋 반복과 매 반복의 GPU 동기화(.any())를 묶음 단위 계산으로 바꾼다.
    #      각 (입자, 오프셋) 쌍의 수식은 원본과 완전히 같고, 마지막 scatter_reduce(amax)는 순서에 무관하다.
    offsets = [(oy_, ox_) for oy_ in range(-radius_px, radius_px + 1) for ox_ in range(-radius_px, radius_px + 1)]
    n_part = int(x.numel())
    chunk = max(1, min(len(offsets), int(_CHUNK_ENTRIES // max(n_part, 1))))
    for c0 in range(0, len(offsets), chunk):
        sub = offsets[c0:c0 + chunk]
        oy_t = torch.tensor([o[0] for o in sub], device=device, dtype=base_y.dtype).view(-1, 1)
        ox_t = torch.tensor([o[1] for o in sub], device=device, dtype=base_x.dtype).view(-1, 1)
        iy = base_y[None, :] + oy_t
        ix = base_x[None, :] + ox_t
        valid_xy = (iy >= 0) & (iy < HEIGHT_MAP_RES) & (ix >= 0) & (ix < HEIGHT_MAP_RES)
        px = x0 + ix.to(dtype) * pixel_size_x
        py = y0 + iy.to(dtype) * pixel_size_y
        dx = px - x[None, :]
        dy = py - y[None, :]
        dist_sq = dx * dx + dy * dy
        inside = valid_xy & (dist_sq <= support_radius_sq)
        ix_i = ix[inside]
        iy_i = iy[inside]
        z_i = z_center[None, :].expand_as(inside)[inside]
        d2_i = dist_sq[inside]
        normalized_d2 = (d2_i / support_radius_sq).clamp(0.0, 1.0)
        candidate_height = z_i + particle_top_offset - radial_drop * normalized_d2
        rim_fraction = ((support_radius - torch.sqrt(d2_i.clamp_min(0.0))) / pixel_size).clamp(0.0, 1.0)
        rim_weight = rim_fraction.square() * (3.0 - 2.0 * rim_fraction)
        candidate_height = candidate_height + radial_drop * torch.log(rim_weight)
        flat_idx = iy_i * HEIGHT_MAP_RES + ix_i
        height_flat.scatter_reduce_(0, flat_idx, candidate_height, reduce="amax", include_self=True)
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




def apply_fast_height(sim_module):
    """vs_simulator 모듈의 particles_to_height_gpu 를 빠른 버전으로 바꾼다(실행 중에만, 파일은 안 바꿈: 물리 서명은 파일 해시라 영향 없음).
    VS_FAST_HEIGHT=0 이면 바꾸지 않는다. 결과는 원본과 비트 단위로 같다(height_fast_check.py 로 검증)."""
    import os
    if os.environ.get("VS_FAST_HEIGHT", "1") != "1":
        return False
    sim_module.particles_to_height_gpu = particles_to_height_gpu_fast
    return True
