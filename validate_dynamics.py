import argparse
import os
import math

import torch
import matplotlib.pyplot as plt

from vs_config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SEED,
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

def mae(
    a,
    b,
):
    return torch.mean(
        torch.abs(
            a - b
        )
    )


def max_abs_error(
    a,
    b,
):
    return torch.max(
        torch.abs(
            a - b
        )
    )


# ============================================================
# SAVE VISUAL COMPARISON
# ============================================================

def save_visualization(
    state,
    gt,
    pred,
    action,
    sample_index,
    output_dir,
):

    state = (
        state
        .detach()
        .float()
        .cpu()
        .squeeze()
        .numpy()
    )

    gt = (
        gt
        .detach()
        .float()
        .cpu()
        .squeeze()
        .numpy()
    )

    pred = (
        pred
        .detach()
        .float()
        .cpu()
        .squeeze()
        .numpy()
    )


    actual_delta = (
        gt - state
    )

    predicted_delta = (
        pred - state
    )

    error = (
        pred - gt
    )


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )


    # --------------------------------------------------------
    # Common state range
    # --------------------------------------------------------

    state_min = min(
        state.min(),
        gt.min(),
        pred.min(),
    )

    state_max = max(
        state.max(),
        gt.max(),
        pred.max(),
    )


    im0 = axes[
        0,
        0
    ].imshow(
        state,
        vmin=state_min,
        vmax=state_max,
    )

    axes[
        0,
        0
    ].set_title(
        "Before"
    )

    fig.colorbar(
        im0,
        ax=axes[
            0,
            0
        ],
    )


    im1 = axes[
        0,
        1
    ].imshow(
        gt,
        vmin=state_min,
        vmax=state_max,
    )

    axes[
        0,
        1
    ].set_title(
        "Actual After"
    )

    fig.colorbar(
        im1,
        ax=axes[
            0,
            1
        ],
    )


    im2 = axes[
        0,
        2
    ].imshow(
        pred,
        vmin=state_min,
        vmax=state_max,
    )

    axes[
        0,
        2
    ].set_title(
        "Predicted After"
    )

    fig.colorbar(
        im2,
        ax=axes[
            0,
            2
        ],
    )


    # --------------------------------------------------------
    # Common deformation range
    # --------------------------------------------------------

    delta_abs_max = max(
        abs(
            actual_delta
        ).max(),
        abs(
            predicted_delta
        ).max(),
        1e-8,
    )


    im3 = axes[
        1,
        0
    ].imshow(
        actual_delta,
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    axes[
        1,
        0
    ].set_title(
        "Actual Deformation"
    )

    fig.colorbar(
        im3,
        ax=axes[
            1,
            0
        ],
    )


    im4 = axes[
        1,
        1
    ].imshow(
        predicted_delta,
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    axes[
        1,
        1
    ].set_title(
        "Predicted Deformation"
    )

    fig.colorbar(
        im4,
        ax=axes[
            1,
            1
        ],
    )


    error_abs_max = max(
        abs(
            error
        ).max(),
        1e-8,
    )


    im5 = axes[
        1,
        2
    ].imshow(
        error,
        vmin=-error_abs_max,
        vmax=error_abs_max,
    )

    axes[
        1,
        2
    ].set_title(
        "Prediction Error"
    )

    fig.colorbar(
        im5,
        ax=axes[
            1,
            2
        ],
    )


    for ax in axes.flat:
        ax.axis(
            "off"
        )


    action_cpu = (
        action
        .detach()
        .cpu()
        .flatten()
        .tolist()
    )


    x = action_cpu[
        0
    ]

    y = action_cpu[
        1
    ]

    theta = action_cpu[
        2
    ]

    length = action_cpu[
        3
    ]

    depth = action_cpu[
        4
    ]


    fig.suptitle(
        (
            f"sample={sample_index} | "
            f"x={x:.3f}, "
            f"y={y:.3f}, "
            f"theta={theta:+.2f}, "
            f"length={length * 1000.0:.1f} mm, "
            f"depth={depth * 1000.0:.1f} mm"
        ),
        fontsize=12,
    )


    plt.tight_layout()


    output_path = os.path.join(
        output_dir,
        f"sample_{sample_index:04d}.png",
    )


    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    return output_path


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
        "--tag",
        default="full",
    )


    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )


    args = parser.parse_args()


    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )


    device = torch.device(
        "cuda"
    )


    torch.manual_seed(
        args.seed
    )


    torch.cuda.manual_seed_all(
        args.seed
    )


    # ========================================================
    # PATHS
    # ========================================================

    dataset_path = os.path.join(
        DATASET_DIR,
        args.material,
    )


    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        (
            f"param2deform_"
            f"{args.material}_"
            f"{args.tag}.pt"
        ),
    )


    output_dir = os.path.join(
        RESULTS_DIR,
        "dynamics_validation",
        args.material,
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )


    print()
    print(
        "==============================================="
    )

    print(
        "DYNAMICS VALIDATION"
    )

    print(
        "==============================================="
    )

    print(
        "dataset    =",
        dataset_path
    )

    print(
        "checkpoint =",
        checkpoint_path
    )

    print(
        "samples    =",
        args.samples
    )

    print(
        "output     =",
        output_dir
    )

    print(
        "==============================================="
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = GPUDataset(
        dataset_path,
        device,
        limit=None,
    )


    n = len(
        dataset
    )


    print(
        "dataset size =",
        n
    )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    model = Param2Deform().to(
        device
    )


    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )


    model.eval()


    print(
        "checkpoint loaded"
    )


    # ========================================================
    # IMPORTANT:
    # use the exact held-out validation indices saved by
    # train_dynamics.py
    # ========================================================

    if (
        "test_indices"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain test_indices. "
            "Use the checkpoint generated by the new "
            "train_dynamics.py."
        )


    test_indices = (
        checkpoint[
            "test_indices"
        ]
        .to(
            device=device,
            dtype=torch.long,
        )
    )


    num_samples = min(
        args.samples,
        test_indices.shape[
            0
        ],
    )


    # randomly select validation samples
    generator = torch.Generator(
        device=device
    )


    generator.manual_seed(
        args.seed + 12345
    )


    order = torch.randperm(
        test_indices.shape[
            0
        ],
        device=device,
        generator=generator,
    )


    selected_indices = test_indices[
        order[
            :num_samples
        ]
    ]


    # ========================================================
    # STATISTICS
    # ========================================================

    model_mse_values = []

    baseline_mse_values = []

    model_mae_values = []

    baseline_mae_values = []

    visual_values = []

    baseline_visual_values = []

    max_error_values = []


    # ========================================================
    # VALIDATION
    # ========================================================

    with torch.no_grad():

        for i in range(
            num_samples
        ):

            index = selected_indices[
                i:
                i + 1
            ]


            batch = dataset.get_batch(
                index
            )


            state = batch[
                "state"
            ]


            gt = (
                batch[
                    "state"
                ]
                +
                batch[
                    "delta"
                ]
            )


            action = batch[
                "action"
            ] if (
                "action"
                in batch
            ) else None


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            pred_delta = model(
                batch[
                    "state_rep"
                ],
                batch[
                    "params"
                ],
            )


            pred = (
                state
                +
                pred_delta
            )


            # ------------------------------------------------
            # ZERO-DEFORMATION BASELINE
            # ------------------------------------------------

            baseline_pred = state


            # ------------------------------------------------
            # MODEL METRICS
            # ------------------------------------------------

            model_l3d = depth_mse(
                pred,
                gt,
            )


            model_lviz = visual_mse(
                pred,
                gt,
            )


            model_mae = mae(
                pred,
                gt,
            )


            model_max_error = (
                max_abs_error(
                    pred,
                    gt,
                )
            )


            # ------------------------------------------------
            # BASELINE METRICS
            # ------------------------------------------------

            baseline_l3d = depth_mse(
                baseline_pred,
                gt,
            )


            baseline_lviz = visual_mse(
                baseline_pred,
                gt,
            )


            baseline_mae = mae(
                baseline_pred,
                gt,
            )


            model_mse_values.append(
                model_l3d
            )

            baseline_mse_values.append(
                baseline_l3d
            )

            model_mae_values.append(
                model_mae
            )

            baseline_mae_values.append(
                baseline_mae
            )

            visual_values.append(
                model_lviz
            )

            baseline_visual_values.append(
                baseline_lviz
            )

            max_error_values.append(
                model_max_error
            )


            # ------------------------------------------------
            # LOG
            # ------------------------------------------------

            values = torch.stack(
                [
                    model_l3d,
                    baseline_l3d,
                    model_mae,
                    baseline_mae,
                    model_lviz,
                    baseline_lviz,
                    model_max_error,
                ]
            ).detach().cpu().tolist()


            (
                model_l3d_cpu,
                baseline_l3d_cpu,
                model_mae_cpu,
                baseline_mae_cpu,
                model_lviz_cpu,
                baseline_lviz_cpu,
                max_error_cpu,

            ) = values


            dataset_index = int(
                index.item()
            )


            print()
            print(
                "-----------------------------------------------"
            )

            print(
                f"SAMPLE {i + 1}/{num_samples} "
                f"(dataset index={dataset_index})"
            )

            print(
                "-----------------------------------------------"
            )


            print(
                f"MSE       model={model_l3d_cpu:.9e} "
                f"baseline={baseline_l3d_cpu:.9e}"
            )


            print(
                f"MAE       model={model_mae_cpu:.9e} "
                f"baseline={baseline_mae_cpu:.9e}"
            )


            print(
                f"Visual    model={model_lviz_cpu:.9e} "
                f"baseline={baseline_lviz_cpu:.9e}"
            )


            print(
                f"Max error = "
                f"{max_error_cpu * 1000.0:.3f} mm"
            )


            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            if action is not None:

                path = save_visualization(

                    state=state,

                    gt=gt,

                    pred=pred,

                    action=action,

                    sample_index=dataset_index,

                    output_dir=output_dir,
                )


                print(
                    "saved image:",
                    path
                )


            else:

                print(
                    "WARNING: batch has no 'action' key; "
                    "visualization skipped."
                )


    # ========================================================
    # FINAL MEAN
    # ========================================================

    model_mse = torch.stack(
        model_mse_values
    ).mean()


    baseline_mse = torch.stack(
        baseline_mse_values
    ).mean()


    model_mae = torch.stack(
        model_mae_values
    ).mean()


    baseline_mae = torch.stack(
        baseline_mae_values
    ).mean()


    model_visual = torch.stack(
        visual_values
    ).mean()


    baseline_visual = torch.stack(
        baseline_visual_values
    ).mean()


    mean_max_error = torch.stack(
        max_error_values
    ).mean()


    mse_improvement = (
        1.0
        -
        model_mse
        /
        baseline_mse.clamp_min(
            1e-12
        )
    ) * 100.0


    mae_improvement = (
        1.0
        -
        model_mae
        /
        baseline_mae.clamp_min(
            1e-12
        )
    ) * 100.0


    visual_improvement = (
        1.0
        -
        model_visual
        /
        baseline_visual.clamp_min(
            1e-12
        )
    ) * 100.0


    final_values = torch.stack(
        [
            model_mse,
            baseline_mse,

            model_mae,
            baseline_mae,

            model_visual,
            baseline_visual,

            mean_max_error,

            mse_improvement,
            mae_improvement,
            visual_improvement,
        ]
    ).detach().cpu().tolist()


    (
        model_mse_cpu,
        baseline_mse_cpu,

        model_mae_cpu,
        baseline_mae_cpu,

        model_visual_cpu,
        baseline_visual_cpu,

        mean_max_error_cpu,

        mse_improvement_cpu,
        mae_improvement_cpu,
        visual_improvement_cpu,

    ) = final_values


    print()
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
        f"MSE:"
    )

    print(
        f"  baseline = "
        f"{baseline_mse_cpu:.9e}"
    )

    print(
        f"  model    = "
        f"{model_mse_cpu:.9e}"
    )

    print(
        f"  improve  = "
        f"{mse_improvement_cpu:.2f}%"
    )


    print()
    print(
        "MAE:"
    )

    print(
        f"  baseline = "
        f"{baseline_mae_cpu * 1000.0:.4f} mm"
    )

    print(
        f"  model    = "
        f"{model_mae_cpu * 1000.0:.4f} mm"
    )

    print(
        f"  improve  = "
        f"{mae_improvement_cpu:.2f}%"
    )


    print()
    print(
        "VISUAL:"
    )

    print(
        f"  baseline = "
        f"{baseline_visual_cpu:.9e}"
    )

    print(
        f"  model    = "
        f"{model_visual_cpu:.9e}"
    )

    print(
        f"  improve  = "
        f"{visual_improvement_cpu:.2f}%"
    )


    print()
    print(
        f"mean max pixel error = "
        f"{mean_max_error_cpu * 1000.0:.3f} mm"
    )


    print()
    print(
        "images saved to:"
    )

    print(
        output_dir
    )


    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()