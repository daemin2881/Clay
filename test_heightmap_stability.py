import os

os.environ[
    "VS_SHOW_VIEWER"
] = "1"

os.environ[
    "VS_MATERIAL"
] = "medium"


import torch
import matplotlib.pyplot as plt

import vs_simulator as sim


def main():

    sim.initialize()


    print()
    print(
        "==============================================="
    )
    print(
        "HEIGHT MAP STABILITY TEST"
    )
    print(
        "==============================================="
    )


    # --------------------------------------------------------
    # STATE 0
    # --------------------------------------------------------

    h0 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    # --------------------------------------------------------
    # NO ACTION
    #
    # Let simulation advance while robot simply holds.
    # --------------------------------------------------------

    sim.hold(
        20
    )


    # --------------------------------------------------------
    # STATE 1
    # --------------------------------------------------------

    h1 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    diff = (
        h1
        -
        h0
    )


    mae = torch.mean(
        torch.abs(
            diff
        )
    )


    mse = torch.mean(
        diff
        *
        diff
    )


    max_error = torch.max(
        torch.abs(
            diff
        )
    )


    values = torch.stack(
        [
            mae,
            mse,
            max_error,
            h0.min(),
            h0.max(),
            h1.min(),
            h1.max(),
        ]
    ).detach().cpu().tolist()


    (
        mae_v,
        mse_v,
        max_v,
        h0_min,
        h0_max,
        h1_min,
        h1_max,
    ) = values


    print()
    print(
        f"h0 min/max = "
        f"{h0_min * 1000.0:.3f} / "
        f"{h0_max * 1000.0:.3f} mm"
    )


    print(
        f"h1 min/max = "
        f"{h1_min * 1000.0:.3f} / "
        f"{h1_max * 1000.0:.3f} mm"
    )


    print()
    print(
        f"NO-ACTION MAE = "
        f"{mae_v * 1000.0:.6f} mm"
    )


    print(
        f"NO-ACTION MSE = "
        f"{mse_v:.9e}"
    )


    print(
        f"NO-ACTION MAX = "
        f"{max_v * 1000.0:.6f} mm"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    h0_np = (
        h0
        .cpu()
        .numpy()
    )


    h1_np = (
        h1
        .cpu()
        .numpy()
    )


    diff_np = (
        diff
        .cpu()
        .numpy()
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            5,
        ),
    )


    state_min = min(
        h0_np.min(),
        h1_np.min(),
    )


    state_max = max(
        h0_np.max(),
        h1_np.max(),
    )


    im0 = axes[
        0
    ].imshow(
        h0_np,
        vmin=state_min,
        vmax=state_max,
    )


    axes[
        0
    ].set_title(
        "Height Before Hold"
    )


    plt.colorbar(
        im0,
        ax=axes[
            0
        ],
    )


    im1 = axes[
        1
    ].imshow(
        h1_np,
        vmin=state_min,
        vmax=state_max,
    )


    axes[
        1
    ].set_title(
        "Height After Hold"
    )


    plt.colorbar(
        im1,
        ax=axes[
            1
        ],
    )


    abs_max = max(
        abs(
            diff_np
        ).max(),
        1e-8,
    )


    im2 = axes[
        2
    ].imshow(
        diff_np,
        vmin=-abs_max,
        vmax=abs_max,
    )


    axes[
        2
    ].set_title(
        "No-action Difference"
    )


    plt.colorbar(
        im2,
        ax=axes[
            2
        ],
    )


    for ax in axes:

        ax.axis(
            "off"
        )


    plt.tight_layout()


    output = (
        "results/"
        "heightmap_stability.png"
    )


    os.makedirs(
        "results",
        exist_ok=True,
    )


    plt.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print()
    print(
        "saved:",
        output
    )

    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()