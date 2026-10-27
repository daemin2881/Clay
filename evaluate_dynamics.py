import argparse
import os

import torch
import matplotlib.pyplot as plt

from vs_config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
)

from vs_dataset import (
    GPUDataset,
)

from vs_dynamics import (
    Param2Deform,
)

from vs_metrics import (
    depth_mse,
    visual_mse,
)


# ============================================================
# BASIC METRICS
# ============================================================

def tensor_mae(
    pred,
    gt,
):
    return torch.mean(
        torch.abs(
            pred - gt
        )
    )


def tensor_max_abs(
    pred,
    gt,
):
    return torch.max(
        torch.abs(
            pred - gt
        )
    )


def delta_magnitude(
    delta,
):
    return torch.mean(
        torch.abs(
            delta
        )
    )


# ============================================================
# SAFE IMAGE CONVERSION
# ============================================================

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
            f"Expected 2D image after squeeze, "
            f"got shape {arr.shape}"
        )

    return arr


# ============================================================
# VISUALIZATION
# ============================================================

def save_prediction_figure(
    save_path,
    current,
    gt_next,
    pred_next,
    gt_delta,
    pred_delta,
    error_map,
    title,
):

    current = to_image(
        current
    )

    gt_next = to_image(
        gt_next
    )

    pred_next = to_image(
        pred_next
    )

    gt_delta = to_image(
        gt_delta
    )

    pred_delta = to_image(
        pred_delta
    )

    error_map = to_image(
        error_map
    )


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )


    fig.suptitle(
        title,
        fontsize=12,
    )


    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    im0 = axes[
        0,
        0
    ].imshow(
        current,
    )

    axes[
        0,
        0
    ].set_title(
        "Current State"
    )

    plt.colorbar(
        im0,
        ax=axes[
            0,
            0
        ],
    )


    # --------------------------------------------------------
    # GT NEXT
    # --------------------------------------------------------

    im1 = axes[
        0,
        1
    ].imshow(
        gt_next,
    )

    axes[
        0,
        1
    ].set_title(
        "GT Next State"
    )

    plt.colorbar(
        im1,
        ax=axes[
            0,
            1
        ],
    )


    # --------------------------------------------------------
    # PRED NEXT
    # --------------------------------------------------------

    im2 = axes[
        0,
        2
    ].imshow(
        pred_next,
    )

    axes[
        0,
        2
    ].set_title(
        "Predicted Next State"
    )

    plt.colorbar(
        im2,
        ax=axes[
            0,
            2
        ],
    )


    # --------------------------------------------------------
    # DELTA SCALE
    # --------------------------------------------------------

    delta_max = max(
        float(
            abs(
                gt_delta.min()
            )
        ),
        float(
            abs(
                gt_delta.max()
            )
        ),
        float(
            abs(
                pred_delta.min()
            )
        ),
        float(
            abs(
                pred_delta.max()
            )
        ),
    )


    if delta_max < 1e-9:

        delta_max = 1e-6


    # --------------------------------------------------------
    # GT DELTA
    # --------------------------------------------------------

    im3 = axes[
        1,
        0
    ].imshow(
        gt_delta,
        vmin=-delta_max,
        vmax=delta_max,
    )

    axes[
        1,
        0
    ].set_title(
        "GT Delta"
    )

    plt.colorbar(
        im3,
        ax=axes[
            1,
            0
        ],
    )


    # --------------------------------------------------------
    # PRED DELTA
    # --------------------------------------------------------

    im4 = axes[
        1,
        1
    ].imshow(
        pred_delta,
        vmin=-delta_max,
        vmax=delta_max,
    )

    axes[
        1,
        1
    ].set_title(
        "Predicted Delta"
    )

    plt.colorbar(
        im4,
        ax=axes[
            1,
            1
        ],
    )


    # --------------------------------------------------------
    # ERROR MAP
    # --------------------------------------------------------

    im5 = axes[
        1,
        2
    ].imshow(
        error_map,
    )

    axes[
        1,
        2
    ].set_title(
        "Absolute Error"
    )

    plt.colorbar(
        im5,
        ax=axes[
            1,
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
            0.95,
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
        "--checkpoint",
        default=None,
    )


    parser.add_argument(
        "--num-vis",
        type=int,
        default=12,
    )


    parser.add_argument(
        "--output-dir",
        default="results/dynamics_eval",
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
    # CHECKPOINT PATH
    # ========================================================

    if args.checkpoint is None:

        checkpoint_path = os.path.join(

            CHECKPOINT_DIR,

            (
                f"param2deform_"
                f"{args.material}_"
                f"paper_"
                f"paper.pt"
            ),
        )

    else:

        checkpoint_path = args.checkpoint


    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "PARAM2DEFORM EVALUATION"
    )

    print(
        "==============================================="
    )

    print(
        "device     =",
        device
    )

    print(
        "material   =",
        args.material
    )

    print(
        "checkpoint =",
        checkpoint_path
    )

    print(
        "==============================================="
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


    if "val_indices" in checkpoint:

        val_indices = checkpoint[
            "val_indices"
        ].to(
            device
        )

    elif "test_indices" in checkpoint:

        val_indices = checkpoint[
            "test_indices"
        ].to(
            device
        )

    else:

        raise RuntimeError(
            "Checkpoint contains neither "
            "'val_indices' nor 'test_indices'."
        )


    print(
        "validation samples =",
        val_indices.numel()
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = Param2Deform().to(
        device
    )


    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )


    model.eval()


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    checkpoint_name = os.path.splitext(
        os.path.basename(
            checkpoint_path
        )
    )[0]


    output_dir = os.path.join(
        args.output_dir,
        args.material,
        checkpoint_name,
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    # ========================================================
    # GLOBAL ACCUMULATORS
    #
    # Important:
    # metrics are weighted by batch sample count.
    # ========================================================

    total_samples = 0


    model_depth_mse_sum = torch.zeros(
        (),
        device=device,
    )

    baseline_depth_mse_sum = torch.zeros(
        (),
        device=device,
    )


    model_visual_mse_sum = torch.zeros(
        (),
        device=device,
    )

    baseline_visual_mse_sum = torch.zeros(
        (),
        device=device,
    )


    model_mae_sum = torch.zeros(
        (),
        device=device,
    )

    baseline_mae_sum = torch.zeros(
        (),
        device=device,
    )


    model_max_sum = torch.zeros(
        (),
        device=device,
    )


    delta_error_sum = torch.zeros(
        (),
        device=device,
    )


    gt_delta_mag_sum = torch.zeros(
        (),
        device=device,
    )

    pred_delta_mag_sum = torch.zeros(
        (),
        device=device,
    )


    # ========================================================
    # FULL VALIDATION
    # ========================================================

    with torch.no_grad():

        for start in range(
            0,
            val_indices.shape[
                0
            ],
            BATCH_SIZE,
        ):

            idx = val_indices[
                start:
                start
                +
                BATCH_SIZE
            ]


            batch = dataset.get_batch(
                idx
            )


            batch_n = idx.numel()


            pred_delta = model(

                batch[
                    "state_rep"
                ],

                batch[
                    "params"
                ],
            )


            current = batch[
                "state"
            ]


            gt_next = batch[
                "next_state"
            ]


            gt_delta = batch[
                "delta"
            ]


            pred_next = (
                current
                +
                pred_delta
            )


            baseline_next = current


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model_depth_mse_sum += (
                depth_mse(
                    pred_next,
                    gt_next,
                )
                *
                batch_n
            )


            model_visual_mse_sum += (
                visual_mse(
                    pred_next,
                    gt_next,
                )
                *
                batch_n
            )


            model_mae_sum += (
                tensor_mae(
                    pred_next,
                    gt_next,
                )
                *
                batch_n
            )


            model_max_sum += (
                tensor_max_abs(
                    pred_next,
                    gt_next,
                )
                *
                batch_n
            )


            delta_error_sum += (
                tensor_mae(
                    pred_delta,
                    gt_delta,
                )
                *
                batch_n
            )


            gt_delta_mag_sum += (
                delta_magnitude(
                    gt_delta
                )
                *
                batch_n
            )


            pred_delta_mag_sum += (
                delta_magnitude(
                    pred_delta
                )
                *
                batch_n
            )


            # ------------------------------------------------
            # ZERO-DELTA BASELINE
            # ------------------------------------------------

            baseline_depth_mse_sum += (
                depth_mse(
                    baseline_next,
                    gt_next,
                )
                *
                batch_n
            )


            baseline_visual_mse_sum += (
                visual_mse(
                    baseline_next,
                    gt_next,
                )
                *
                batch_n
            )


            baseline_mae_sum += (
                tensor_mae(
                    baseline_next,
                    gt_next,
                )
                *
                batch_n
            )


            total_samples += batch_n


    if total_samples == 0:

        raise RuntimeError(
            "Validation set contains zero samples."
        )


    # ========================================================
    # GLOBAL AVERAGES
    # ========================================================

    model_depth_mse = (
        model_depth_mse_sum
        /
        total_samples
    )


    baseline_depth_mse = (
        baseline_depth_mse_sum
        /
        total_samples
    )


    model_visual_mse = (
        model_visual_mse_sum
        /
        total_samples
    )


    baseline_visual_mse = (
        baseline_visual_mse_sum
        /
        total_samples
    )


    model_mae_val = (
        model_mae_sum
        /
        total_samples
    )


    baseline_mae_val = (
        baseline_mae_sum
        /
        total_samples
    )


    model_max_val = (
        model_max_sum
        /
        total_samples
    )


    delta_error_val = (
        delta_error_sum
        /
        total_samples
    )


    gt_delta_mag = (
        gt_delta_mag_sum
        /
        total_samples
    )


    pred_delta_mag = (
        pred_delta_mag_sum
        /
        total_samples
    )


    depth_improvement = (
        1.0
        -
        model_depth_mse
        /
        baseline_depth_mse.clamp_min(
            1e-12
        )
    )


    visual_improvement = (
        1.0
        -
        model_visual_mse
        /
        baseline_visual_mse.clamp_min(
            1e-12
        )
    )


    mae_improvement = (
        1.0
        -
        model_mae_val
        /
        baseline_mae_val.clamp_min(
            1e-12
        )
    )


    delta_ratio = (
        pred_delta_mag
        /
        gt_delta_mag.clamp_min(
            1e-12
        )
    )


    values = torch.stack(
        [
            model_depth_mse,
            baseline_depth_mse,

            model_visual_mse,
            baseline_visual_mse,

            model_mae_val,
            baseline_mae_val,

            model_max_val,

            delta_error_val,

            gt_delta_mag,
            pred_delta_mag,

            depth_improvement,
            visual_improvement,
            mae_improvement,

            delta_ratio,
        ]
    ).detach().cpu().tolist()


    (
        model_depth_mse_v,
        baseline_depth_mse_v,

        model_visual_mse_v,
        baseline_visual_mse_v,

        model_mae_v,
        baseline_mae_v,

        model_max_v,

        delta_error_v,

        gt_delta_mag_v,
        pred_delta_mag_v,

        depth_improvement_v,
        visual_improvement_v,
        mae_improvement_v,

        delta_ratio_v,

    ) = values


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "==============================================="
    )


    print(
        f"Model depth MSE        = "
        f"{model_depth_mse_v:.9e}"
    )

    print(
        f"Zero-delta depth MSE   = "
        f"{baseline_depth_mse_v:.9e}"
    )

    print(
        f"Depth improvement      = "
        f"{depth_improvement_v * 100.0:.2f}%"
    )


    print()


    print(
        f"Model visual MSE       = "
        f"{model_visual_mse_v:.9e}"
    )

    print(
        f"Zero-delta visual MSE  = "
        f"{baseline_visual_mse_v:.9e}"
    )

    print(
        f"Visual improvement     = "
        f"{visual_improvement_v * 100.0:.2f}%"
    )


    print()


    print(
        f"Model next-state MAE   = "
        f"{model_mae_v * 1000.0:.6f} mm"
    )

    print(
        f"Zero-delta MAE         = "
        f"{baseline_mae_v * 1000.0:.6f} mm"
    )

    print(
        f"MAE improvement        = "
        f"{mae_improvement_v * 100.0:.2f}%"
    )


    print()


    print(
        f"Mean model MAX error   = "
        f"{model_max_v * 1000.0:.6f} mm"
    )

    print(
        f"Delta prediction MAE   = "
        f"{delta_error_v * 1000.0:.6f} mm"
    )


    print()


    print(
        f"GT delta magnitude     = "
        f"{gt_delta_mag_v * 1000.0:.6f} mm"
    )

    print(
        f"Pred delta magnitude   = "
        f"{pred_delta_mag_v * 1000.0:.6f} mm"
    )

    print(
        f"Pred / GT delta ratio  = "
        f"{delta_ratio_v:.4f}"
    )


    print(
        "==============================================="
    )


    # ========================================================
    # VISUALIZATION SELECTION
    # ========================================================

    num_vis = min(
        args.num_vis,
        val_indices.numel(),
    )


    positions = torch.linspace(
        0,
        val_indices.numel() - 1,
        steps=num_vis,
        device=device,
    ).long()


    selected = val_indices[
        positions
    ]


    print()
    print(
        "Saving prediction visualizations..."
    )


    # ========================================================
    # SAMPLE VISUALIZATION
    # ========================================================

    with torch.no_grad():

        for vis_i in range(
            num_vis
        ):

            idx = selected[
                vis_i:
                vis_i
                +
                1
            ]


            batch = dataset.get_batch(
                idx
            )


            pred_delta = model(

                batch[
                    "state_rep"
                ],

                batch[
                    "params"
                ],
            )


            current = batch[
                "state"
            ]


            gt_next = batch[
                "next_state"
            ]


            gt_delta = batch[
                "delta"
            ]


            pred_next = (
                current
                +
                pred_delta
            )


            baseline_next = current


            error_map = torch.abs(
                pred_next
                -
                gt_next
            )


            sample_model_mae = tensor_mae(
                pred_next,
                gt_next,
            )


            sample_baseline_mae = tensor_mae(
                baseline_next,
                gt_next,
            )


            sample_gt_delta_mag = delta_magnitude(
                gt_delta
            )


            sample_pred_delta_mag = delta_magnitude(
                pred_delta
            )


            sample_ratio = (
                sample_pred_delta_mag
                /
                sample_gt_delta_mag.clamp_min(
                    1e-12
                )
            )


            sample_values = torch.stack(
                [
                    sample_model_mae,
                    sample_baseline_mae,
                    sample_gt_delta_mag,
                    sample_pred_delta_mag,
                    sample_ratio,
                ]
            ).detach().cpu().tolist()


            (
                sample_model_mae_v,
                sample_baseline_mae_v,
                sample_gt_delta_mag_v,
                sample_pred_delta_mag_v,
                sample_ratio_v,

            ) = sample_values


            sample_id = int(
                idx[
                    0
                ].item()
            )


            action = (
                batch[
                    "action"
                ][
                    0
                ]
                .detach()
                .cpu()
                .tolist()
            )


            x, y, theta, length, depth = action


            title = (
                f"sample={sample_id} | "
                f"x={x:.3f}, y={y:.3f}, "
                f"theta={theta:.2f}, "
                f"l={length * 1000.0:.1f}mm, "
                f"z={depth * 1000.0:.1f}mm\n"
                f"model MAE={sample_model_mae_v * 1000.0:.3f}mm | "
                f"zero MAE={sample_baseline_mae_v * 1000.0:.3f}mm | "
                f"delta ratio={sample_ratio_v:.3f}"
            )


            save_path = os.path.join(

                output_dir,

                (
                    f"prediction_"
                    f"{vis_i:02d}_"
                    f"sample_"
                    f"{sample_id:04d}.png"
                ),
            )


            save_prediction_figure(

                save_path=save_path,

                current=current,

                gt_next=gt_next,

                pred_next=pred_next,

                gt_delta=gt_delta,

                pred_delta=pred_delta,

                error_map=error_map,

                title=title,
            )


            print(
                f"[{vis_i + 1:02d}/{num_vis}] "
                f"sample={sample_id:04d} | "
                f"model MAE="
                f"{sample_model_mae_v * 1000.0:.4f}mm | "
                f"zero="
                f"{sample_baseline_mae_v * 1000.0:.4f}mm | "
                f"delta ratio="
                f"{sample_ratio_v:.3f}"
            )


            print(
                "saved:",
                save_path
            )


    print()
    print(
        "==============================================="
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "==============================================="
    )

    print(
        "output dir =",
        output_dir
    )

    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()