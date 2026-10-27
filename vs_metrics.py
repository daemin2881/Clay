import torch

from vs_representation import (
    ensure_bchw,
    spatial_gradient,
)

from vs_config import (
    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_WIDTH,
    CLAY_LENGTH,
)


def depth_mse(
    pred,
    target,
):

    pred = ensure_bchw(
        pred
    )

    target = ensure_bchw(
        target
    )

    return torch.mean(
        (
            pred
            -
            target
        ) ** 2
    )


def visual_mse(
    pred,
    target,
):

    pred = ensure_bchw(
        pred
    )

    target = ensure_bchw(
        target
    )

    pgx, pgy = spatial_gradient(
        pred
    )

    tgx, tgy = spatial_gradient(
        target
    )

    return torch.mean(
        (
            pgx
            -
            tgx
        ) ** 2
        +
        (
            pgy
            -
            tgy
        ) ** 2
    )


def total_loss(
    pred,
    target,
    w3d=1.0,
    wviz=1.0,
):

    l3d = depth_mse(
        pred,
        target
    )

    lviz = visual_mse(
        pred,
        target
    )

    total = (
        w3d
        *
        l3d
        +
        wviz
        *
        lviz
    )

    return (
        total,
        l3d,
        lviz,
    )


# ============================================================
# POINT CLOUD
# ============================================================

def height_to_pointcloud(
    height,
    samples=24,
):

    height = ensure_bchw(
        height
    )

    B, _, H, W = height.shape

    device = height.device

    dtype = height.dtype

    rows = torch.linspace(
        0,
        H - 1,
        samples,
        device=device,
    ).long()

    cols = torch.linspace(
        0,
        W - 1,
        samples,
        device=device,
    ).long()

    rr, cc = torch.meshgrid(
        rows,
        cols,
        indexing="ij",
    )

    x = (
        CLAY_CENTER_X
        -
        CLAY_WIDTH / 2.0
        +
        cc.to(
            dtype
        )
        /
        (W - 1)
        *
        CLAY_WIDTH
    )

    y = (
        CLAY_CENTER_Y
        -
        CLAY_LENGTH / 2.0
        +
        rr.to(
            dtype
        )
        /
        (H - 1)
        *
        CLAY_LENGTH
    )

    x = x.reshape(
        1,
        -1
    ).expand(
        B,
        -1
    )

    y = y.reshape(
        1,
        -1
    ).expand(
        B,
        -1
    )

    z = height[
        :,
        0,
        rr,
        cc
    ].reshape(
        B,
        -1
    )

    return torch.stack(
        [
            x,
            y,
            z,
        ],
        dim=-1,
    )


def chamfer_distance(
    pred,
    target,
    samples=24,
):

    P = height_to_pointcloud(
        pred,
        samples
    )

    Q = height_to_pointcloud(
        target,
        samples
    )

    D = torch.cdist(
        P,
        Q
    )

    p_to_q = D.min(
        dim=2
    ).values.mean(
        dim=1
    )

    q_to_p = D.min(
        dim=1
    ).values.mean(
        dim=1
    )

    return (
        (
            p_to_q
            +
            q_to_p
        )
        /
        2.0
    ).mean()


# ============================================================
# APPROXIMATE GPU EMD
#
# Sinkhorn approximation.
# Evaluation only.
# ============================================================

def sinkhorn_emd(
    pred,
    target,
    samples=20,
    epsilon=0.01,
    iterations=30,
):

    P = height_to_pointcloud(
        pred,
        samples
    )

    Q = height_to_pointcloud(
        target,
        samples
    )

    C = torch.cdist(
        P,
        Q
    ) ** 2

    B, N, M = C.shape

    mu = torch.full(
        (
            B,
            N,
        ),
        1.0 / N,
        device=C.device,
        dtype=C.dtype,
    )

    nu = torch.full(
        (
            B,
            M,
        ),
        1.0 / M,
        device=C.device,
        dtype=C.dtype,
    )

    K = torch.exp(
        -C
        /
        epsilon
    )

    u = torch.ones_like(
        mu
    )

    v = torch.ones_like(
        nu
    )

    for _ in range(
        iterations
    ):

        Kv = torch.bmm(
            K,
            v.unsqueeze(
                -1
            )
        ).squeeze(
            -1
        )

        u = (
            mu
            /
            torch.clamp(
                Kv,
                min=1e-8,
            )
        )

        KTu = torch.bmm(
            K.transpose(
                1,
                2
            ),
            u.unsqueeze(
                -1
            )
        ).squeeze(
            -1
        )

        v = (
            nu
            /
            torch.clamp(
                KTu,
                min=1e-8,
            )
        )

    transport = (
        u.unsqueeze(
            2
        )
        *
        K
        *
        v.unsqueeze(
            1
        )
    )

    cost = (
        transport
        *
        C
    ).sum(
        dim=(1, 2)
    )

    return cost.mean()
    