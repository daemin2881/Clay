import argparse
import os

import torch
import matplotlib.pyplot as plt

from vs_config import (
    DATASET_DIR,
)

from vs_dataset import (
    GPUDataset,
)

from vs_representation import (
    actual_to_canonical,
    canonical_to_actual,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def mean_abs(
    x,
):

    return torch.mean(
        torch.abs(
            x
        )
    )


def max_abs(
    x,
):

    return torch.max(
        torch.abs(
            x
        )
    )


def to_image(
    tensor,
):

    arr = (
        tensor
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )

    if arr.ndim != 2:

        raise RuntimeError(
            f"Expected 2D image, got shape {arr.shape}"
        )

    return arr


# ============================================================
# VISUALIZATION
# ============================================================

def save_figure(
    save_path,
    world_state,
    world_next,
    world_delta,
    can_state,
    can_next,
    can_delta,
    restored_state,
    restored_next,
    restored_delta,
    title,
):

    world_state = to_image(
        world_state
    )

    world_next = to_image(
        world_next
    )

    world_delta = to_image(
        world_delta
    )

    can_state = to_image(
        can_state
    )

    can_next = to_image(
        can_next
    )

    can_delta = to_image(
        can_delta
    )

    restored_state = to_image(
        restored_state
    )

    restored_next = to_image(
        restored_next
    )

    restored_delta = to_image(
        restored_delta
    )


    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            15,
            14,
        ),
    )

    fig.suptitle(
        title,
        fontsize=12,
    )


    # --------------------------------------------------------
    # DELTA SHARED SCALE
    # --------------------------------------------------------

    delta_max = max(
        float(
            abs(
                world_delta.min()
            )
        ),
        float(
            abs(
                world_delta.max()
            )
        ),
        float(
            abs(
                can_delta.min()
            )
        ),
        float(
            abs(
                can_delta.max()
            )
        ),
        float(
            abs(
                restored_delta.min()
            )
        ),
        float(
            abs(
                restored_delta.max()
            )
        ),
    )


    if delta_max < 1e-9:

        delta_max = 1e-6


    # ========================================================
    # ROW 1: WORLD
    # ========================================================

    im00 = axes[
        0,
        0
    ].imshow(
        world_state
    )

    axes[
        0,
        0
    ].set_title(
        "World Current"
    )

    plt.colorbar(
        im00,
        ax=axes[
            0,
            0
        ],
    )


    im01 = axes[
        0,
        1
    ].imshow(
        world_next
    )

    axes[
        0,
        1
    ].set_title(
        "World Next"
    )

    plt.colorbar(
        im01,
        ax=axes[
            0,
            1
        ],
    )


    im02 = axes[
        0,
        2
    ].imshow(
        world_delta,
        vmin=-delta_max,
        vmax=delta_max,
    )

    axes[
        0,
        2
    ].set_title(
        "World Delta"
    )

    plt.colorbar(
        im02,
        ax=axes[
            0,
            2
        ],
    )


    # ========================================================
    # ROW 2: CANONICAL
    # ========================================================

    im10 = axes[
        1,
        0
    ].imshow(
        can_state
    )

    axes[
        1,
        0
    ].set_title(
        "Canonical Current"
    )

    plt.colorbar(
        im10,
        ax=axes[
            1,
            0
        ],
    )


    im11 = axes[
        1,
        1
    ].imshow(
        can_next
    )

    axes[
        1,
        1
    ].set_title(
        "Canonical Next"
    )

    plt.colorbar(
        im11,
        ax=axes[
            1,
            1
        ],
    )


    im12 = axes[
        1,
        2
    ].imshow(
        can_delta,
        vmin=-delta_max,
        vmax=delta_max,
    )

    axes[
        1,
        2
    ].set_title(
        "Canonical Delta"
    )

    plt.colorbar(
        im12,
        ax=axes[
            1,
            2
        ],
    )


    # ========================================================
    # ROW 3: ROUND TRIP
    # ========================================================

    im20 = axes[
        2,
        0
    ].imshow(
        restored_state
    )

    axes[
        2,
        0
    ].set_title(
        "Restored Current"
    )

    plt.colorbar(
        im20,
        ax=axes[
            2,
            0
        ],
    )


    im21 = axes[
        2,
        1
    ].imshow(
        restored_next
    )

    axes[
        2,
        1
    ].set_title(
        "Restored Next"
    )

    plt.colorbar(
        im21,
        ax=axes[
            2,
            1
        ],
    )


    im22 = axes[
        2,
        2
    ].imshow(
        restored_delta,
        vmin=-delta_max,
        vmax=delta_max,
    )

    axes[
        2,
        2
    ].set_title(
        "Restored Delta"
    )

    plt.colorbar(
        im22,
        ax=axes[
            2,
            2
        ],
    )


    for ax in axes.flat:

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )


    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.96,
        ]
    )


    plt.savefig(
        save_path,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--material",
        default="medium",
    )


    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--output-dir",
        default="results/canonical_test",
    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )


    device = torch.device(
        "cuda"
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = GPUDataset(

        os.path.join(
            DATASET_DIR,
            args.material,
        ),

        device,
    )


    n = len(
        dataset
    )


    num_samples = min(
        args.num_samples,
        n,
    )


    if num_samples <= 0:

        raise RuntimeError(
            "No samples available."
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = os.path.join(
        args.output_dir,
        args.material,
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    # ========================================================
    # SAMPLE INDICES
    #
    # Spread samples over the whole dataset.
    # ========================================================

    indices = torch.linspace(
        0,
        n - 1,
        steps=num_samples,
        device=device,
    ).long()


    # ========================================================
    # GLOBAL STATS
    # ========================================================

    world_mae_sum = torch.zeros(
        (),
        device=device,
    )

    can_mae_sum = torch.zeros(
        (),
        device=device,
    )

    world_max_sum = torch.zeros(
        (),
        device=device,
    )

    can_max_sum = torch.zeros(
        (),
        device=device,
    )

    roundtrip_state_mae_sum = torch.zeros(
        (),
        device=device,
    )

    roundtrip_next_mae_sum = torch.zeros(
        (),
        device=device,
    )

    roundtrip_delta_mae_sum = torch.zeros(
        (),
        device=device,
    )

    count = 0


    print()
    print(
        "==============================================="
    )

    print(
        "CANONICAL TRANSFORM SANITY TEST"
    )

    print(
        "==============================================="
    )

    print(
        "material =",
        args.material
    )

    print(
        "samples  =",
        num_samples
    )

    print(
        "==============================================="
    )


    # ========================================================
    # TEST
    # ========================================================

    with torch.no_grad():

        for i in range(
            num_samples
        ):

            idx = indices[
                i:
                i
                +
                1
            ]


            raw = dataset.get_raw_batch(
                idx
            )


            state = raw[
                "state"
            ]


            action = raw[
                "action"
            ]


            next_state = raw[
                "next_state"
            ]


            x = action[
                :,
                0
            ]

            y = action[
                :,
                1
            ]

            theta = action[
                :,
                2
            ]

            length = action[
                :,
                3
            ]

            depth = action[
                :,
                4
            ]


            # ------------------------------------------------
            # WORLD DELTA
            # ------------------------------------------------

            world_delta = (
                next_state
                -
                state
            )


            # ------------------------------------------------
            # ACTUAL -> CANONICAL
            # ------------------------------------------------

            can_state = actual_to_canonical(
                state,
                x,
                y,
                theta,
            )


            can_next = actual_to_canonical(
                next_state,
                x,
                y,
                theta,
            )


            can_delta = (
                can_next
                -
                can_state
            )


            # ------------------------------------------------
            # CANONICAL -> ACTUAL ROUND TRIP
            # ------------------------------------------------

            restored_state = canonical_to_actual(
                can_state,
                x,
                y,
                theta,
            )


            restored_next = canonical_to_actual(
                can_next,
                x,
                y,
                theta,
            )


            restored_delta = (
                restored_next
                -
                restored_state
            )


            # ------------------------------------------------
            # MAGNITUDES
            # ------------------------------------------------

            world_mae = mean_abs(
                world_delta
            )


            can_mae = mean_abs(
                can_delta
            )


            world_max = max_abs(
                world_delta
            )


            can_max = max_abs(
                can_delta
            )


            mae_ratio = (
                can_mae
                /
                world_mae.clamp_min(
                    1e-12
                )
            )


            max_ratio = (
                can_max
                /
                world_max.clamp_min(
                    1e-12
                )
            )


            # ------------------------------------------------
            # ROUND TRIP ERRORS
            # ------------------------------------------------

            state_roundtrip_mae = mean_abs(
                restored_state
                -
                state
            )


            next_roundtrip_mae = mean_abs(
                restored_next
                -
                next_state
            )


            delta_roundtrip_mae = mean_abs(
                restored_delta
                -
                world_delta
            )


            # ------------------------------------------------
            # ACCUMULATE
            # ------------------------------------------------

            world_mae_sum += world_mae

            can_mae_sum += can_mae

            world_max_sum += world_max

            can_max_sum += can_max

            roundtrip_state_mae_sum += (
                state_roundtrip_mae
            )

            roundtrip_next_mae_sum += (
                next_roundtrip_mae
            )

            roundtrip_delta_mae_sum += (
                delta_roundtrip_mae
            )


            count += 1


            # ------------------------------------------------
            # CPU VALUES
            # ------------------------------------------------

            values = torch.stack(
                [
                    world_mae,
                    can_mae,
                    world_max,
                    can_max,
                    mae_ratio,
                    max_ratio,
                    state_roundtrip_mae,
                    next_roundtrip_mae,
                    delta_roundtrip_mae,
                ]
            ).detach().cpu().tolist()


            (
                world_mae_v,
                can_mae_v,
                world_max_v,
                can_max_v,
                mae_ratio_v,
                max_ratio_v,
                state_rt_v,
                next_rt_v,
                delta_rt_v,

            ) = values


            sample_id = int(
                idx[
                    0
                ].item()
            )


            action_values = (
                action[
                    0
                ]
                .detach()
                .cpu()
                .tolist()
            )


            (
                x_v,
                y_v,
                theta_v,
                length_v,
                depth_v,

            ) = action_values


            print(
                f"[{i + 1:02d}/{num_samples}] "
                f"sample={sample_id:04d} | "
                f"world MAE={world_mae_v * 1000.0:.4f}mm | "
                f"can MAE={can_mae_v * 1000.0:.4f}mm | "
                f"ratio={mae_ratio_v:.3f} | "
                f"world MAX={world_max_v * 1000.0:.3f}mm | "
                f"can MAX={can_max_v * 1000.0:.3f}mm | "
                f"max ratio={max_ratio_v:.3f} | "
                f"RT delta MAE={delta_rt_v * 1000.0:.4f}mm"
            )


            # ------------------------------------------------
            # SAVE FIGURE
            # ------------------------------------------------

            title = (
                f"sample={sample_id} | "
                f"x={x_v:.3f}, y={y_v:.3f}, "
                f"theta={theta_v:.2f}, "
                f"l={length_v * 1000.0:.1f}mm, "
                f"z={depth_v * 1000.0:.1f}mm\n"
                f"canonical/world MAE ratio="
                f"{mae_ratio_v:.3f} | "
                f"MAX ratio={max_ratio_v:.3f} | "
                f"round-trip delta error="
                f"{delta_rt_v * 1000.0:.4f}mm"
            )


            save_path = os.path.join(

                output_dir,

                (
                    f"canonical_"
                    f"{i:02d}_"
                    f"sample_"
                    f"{sample_id:04d}.png"
                ),
            )


            save_figure(

                save_path=save_path,

                world_state=state,

                world_next=next_state,

                world_delta=world_delta,

                can_state=can_state,

                can_next=can_next,

                can_delta=can_delta,

                restored_state=restored_state,

                restored_next=restored_next,

                restored_delta=restored_delta,

                title=title,
            )


    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    world_mae_mean = (
        world_mae_sum
        /
        count
    )


    can_mae_mean = (
        can_mae_sum
        /
        count
    )


    world_max_mean = (
        world_max_sum
        /
        count
    )


    can_max_mean = (
        can_max_sum
        /
        count
    )


    state_rt_mean = (
        roundtrip_state_mae_sum
        /
        count
    )


    next_rt_mean = (
        roundtrip_next_mae_sum
        /
        count
    )


    delta_rt_mean = (
        roundtrip_delta_mae_sum
        /
        count
    )


    global_mae_ratio = (
        can_mae_mean
        /
        world_mae_mean.clamp_min(
            1e-12
        )
    )


    global_max_ratio = (
        can_max_mean
        /
        world_max_mean.clamp_min(
            1e-12
        )
    )


    summary = torch.stack(
        [
            world_mae_mean,
            can_mae_mean,
            world_max_mean,
            can_max_mean,
            global_mae_ratio,
            global_max_ratio,
            state_rt_mean,
            next_rt_mean,
            delta_rt_mean,
        ]
    ).detach().cpu().tolist()


    (
        world_mae_v,
        can_mae_v,
        world_max_v,
        can_max_v,
        mae_ratio_v,
        max_ratio_v,
        state_rt_v,
        next_rt_v,
        delta_rt_v,

    ) = summary


    print()
    print(
        "==============================================="
    )

    print(
        "GLOBAL SUMMARY"
    )

    print(
        "==============================================="
    )


    print(
        f"World delta MAE         = "
        f"{world_mae_v * 1000.0:.6f} mm"
    )


    print(
        f"Canonical delta MAE     = "
        f"{can_mae_v * 1000.0:.6f} mm"
    )


    print(
        f"Canonical / World MAE   = "
        f"{mae_ratio_v:.4f}"
    )


    print()


    print(
        f"World delta MAX         = "
        f"{world_max_v * 1000.0:.6f} mm"
    )


    print(
        f"Canonical delta MAX     = "
        f"{can_max_v * 1000.0:.6f} mm"
    )


    print(
        f"Canonical / World MAX   = "
        f"{max_ratio_v:.4f}"
    )


    print()


    print(
        f"Round-trip state MAE    = "
        f"{state_rt_v * 1000.0:.6f} mm"
    )


    print(
        f"Round-trip next MAE     = "
        f"{next_rt_v * 1000.0:.6f} mm"
    )


    print(
        f"Round-trip delta MAE    = "
        f"{delta_rt_v * 1000.0:.6f} mm"
    )


    print(
        "==============================================="
    )


    print(
        "Interpretation:"
    )


    print(
        "  MAE ratio ~ 1.0  -> canonical transform "
        "preserves average deformation."
    )


    print(
        "  MAX ratio ~ 1.0  -> deformation peaks are "
        "preserved."
    )


    print(
        "  Low round-trip delta MAE -> transform pair "
        "is internally consistent."
    )


    print(
        "==============================================="
    )


    print(
        "output dir =",
        output_dir
    )


if __name__ == "__main__":

    main()