import argparse
import os
import random

import torch

from vs_config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    TRAIN_RATIO,
    SEED,
    LOSS_EPS,
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
# MEMORY-EFFICIENT MODEL FORWARD
# ============================================================

def forward_model_amp(
    model,
    state_representation,
    parameters,
):
    if state_representation.device.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            return model(
                state_representation,
                parameters,
            )

    return model(
        state_representation,
        parameters,
    )


# ============================================================
# PAPER LOSS
#
# Paper-style objective:
#
#   L = w_3d * L3D + w_viz * Lviz
#
# ============================================================

def paper_loss(
    l3d,
    lviz,
    w_3d,
    w_viz,
):

    loss = (
        w_3d
        *
        l3d

        +

        w_viz
        *
        lviz
    )


    return (
        loss,
        l3d,
        lviz,
    )


# ============================================================
# DATASET-DERIVED LOSS SCALE
#
# Used only in normalized mode.
#
# Baseline:
#
#   prediction = current state
#
# i.e.
#
#   predicted deformation = 0
#
# ============================================================

@torch.no_grad()
def compute_dataset_loss_scales(
    dataset,
    indices,
    batch_size,
    device,
):

    l3d_sum = torch.zeros(
        (),
        device=device,
    )


    lviz_sum = torch.zeros(
        (),
        device=device,
    )


    batches = 0


    for start in range(
        0,
        indices.shape[0],
        batch_size,
    ):

        idx = indices[
            start:
            start
            +
            batch_size
        ]


        batch = dataset.get_batch(
            idx
        )


        current = batch[
            "state"
        ]


        gt = batch[
            "next_state"
        ]


        # ----------------------------------------------------
        # ZERO-DEFORMATION BASELINE
        # ----------------------------------------------------

        baseline_prediction = current


        l3d = depth_mse(
            baseline_prediction,
            gt,
        )


        lviz = visual_mse(
            baseline_prediction,
            gt,
        )


        l3d_sum += l3d

        lviz_sum += lviz


        batches += 1


    if batches == 0:

        raise RuntimeError(
            "Cannot compute loss scales: "
            "training set contains zero batches."
        )


    scale_l3d = (
        l3d_sum
        /
        batches
    ).clamp_min(
        LOSS_EPS
    )


    scale_lviz = (
        lviz_sum
        /
        batches
    ).clamp_min(
        LOSS_EPS
    )


    return (
        scale_l3d,
        scale_lviz,
    )


# ============================================================
# NORMALIZED LOSS
#
# Adaptation used to keep L3D and Lviz on comparable scales.
#
# This is NOT the strict paper objective.
#
# ============================================================

def normalized_loss(
    l3d,
    lviz,
    scale_l3d,
    scale_lviz,
    lambda_viz,
):

    norm_l3d = (
        l3d
        /
        scale_l3d
    )


    norm_lviz = (
        lviz
        /
        scale_lviz
    )


    loss = (
        norm_l3d

        +

        lambda_viz
        *
        norm_lviz
    )


    return (
        loss,
        norm_l3d,
        norm_lviz,
    )


# ============================================================
# ROLLOUT-LEVEL SPLIT
#
# Important:
#
# samples from the same rollout must not be split across
# train and validation.
#
# ============================================================

def make_rollout_split(
    dataset,
    train_ratio,
    seed,
    device,
):

    if not dataset.has_rollout_metadata:

        raise RuntimeError(
            "Rollout metadata is required for rollout-level "
            "train/validation split."
        )


    rollout_ids = dataset.get_rollout_ids()


    unique_rollouts = torch.unique(
        rollout_ids
    )


    num_rollouts = unique_rollouts.numel()


    if num_rollouts < 2:

        raise RuntimeError(
            f"Need at least 2 rollouts for train/val split, "
            f"got {num_rollouts}."
        )


    generator = torch.Generator(
        device=device
    )


    generator.manual_seed(
        seed
    )


    perm = torch.randperm(
        num_rollouts,
        generator=generator,
        device=device,
    )


    shuffled_rollouts = unique_rollouts[
        perm
    ]


    train_rollout_n = int(
        num_rollouts
        *
        train_ratio
    )


    train_rollout_n = max(
        1,
        min(
            train_rollout_n,
            num_rollouts - 1,
        ),
    )


    train_rollouts = shuffled_rollouts[
        :train_rollout_n
    ]


    val_rollouts = shuffled_rollouts[
        train_rollout_n:
    ]


    # --------------------------------------------------------
    # BUILD SAMPLE MASKS
    # --------------------------------------------------------

    train_mask = torch.zeros(
        rollout_ids.shape,
        dtype=torch.bool,
        device=device,
    )


    val_mask = torch.zeros(
        rollout_ids.shape,
        dtype=torch.bool,
        device=device,
    )


    for rid in train_rollouts:

        train_mask |= (
            rollout_ids
            ==
            rid
        )


    for rid in val_rollouts:

        val_mask |= (
            rollout_ids
            ==
            rid
        )


    train_indices = torch.nonzero(
        train_mask,
        as_tuple=False,
    ).squeeze(
        1
    )


    val_indices = torch.nonzero(
        val_mask,
        as_tuple=False,
    ).squeeze(
        1
    )


    if train_indices.numel() == 0:

        raise RuntimeError(
            "Training split contains zero samples."
        )


    if val_indices.numel() == 0:

        raise RuntimeError(
            "Validation split contains zero samples."
        )


    return (
        train_indices,
        val_indices,
        train_rollouts,
        val_rollouts,
    )


# ============================================================
# FALLBACK SAMPLE-LEVEL SPLIT
#
# Only for legacy datasets without rollout metadata.
#
# ============================================================

def make_sample_split(
    n,
    train_ratio,
    seed,
    device,
):

    generator = torch.Generator(
        device=device
    )


    generator.manual_seed(
        seed
    )


    permutation = torch.randperm(
        n,
        device=device,
        generator=generator,
    )


    train_n = int(
        n
        *
        train_ratio
    )


    train_n = max(
        1,
        min(
            train_n,
            n - 1,
        ),
    )


    train_indices = permutation[
        :train_n
    ]


    val_indices = permutation[
        train_n:
    ]


    return (
        train_indices,
        val_indices,
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
        "--limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory. Default: dataset/<material>",
    )


    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Model checkpoint used to initialize fine-tuning.",
    )


    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )


    # --------------------------------------------------------
    # LOSS MODE
    #
    # paper:
    #   w3d * L3D + wviz * Lviz
    #
    # normalized:
    #   L3D / baseline + lambda * Lviz / baseline
    # --------------------------------------------------------

    parser.add_argument(
        "--loss-mode",
        type=str,
        default="paper",
        choices=[
            "paper",
            "normalized",
        ],
    )


    parser.add_argument(
        "--w-3d",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--w-viz",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--lambda-viz",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )


    parser.add_argument(
        "--tag",
        default="full",
    )


    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )


    args = parser.parse_args()


    # ========================================================
    # RANDOM SEED
    # ========================================================

    torch.manual_seed(
        SEED
    )


    random.seed(
        SEED
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    # ========================================================
    # DEVICE
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for this training script."
        )


    device = torch.device(
        "cuda"
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "PARAM2DEFORM TRAINING"
    )

    print(
        "==============================================="
    )


    print(
        "device        =",
        device
    )


    print(
        "material      =",
        args.material
    )


    print(
        "epochs        =",
        args.epochs
    )


    print(
        "batch size    =",
        BATCH_SIZE
    )


    print(
        "learning rate =",
        args.learning_rate
    )


    print(
        "loss mode     =",
        args.loss_mode
    )


    if args.loss_mode == "paper":

        print(
            "w_3d          =",
            args.w_3d
        )


        print(
            "w_viz         =",
            args.w_viz
        )


    else:

        print(
            "lambda_viz    =",
            args.lambda_viz
        )


    print(
        "grad clip     =",
        args.grad_clip
    )


    print(
        "==============================================="
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset_root = (
        args.dataset_dir
        if args.dataset_dir is not None
        else os.path.join(
            DATASET_DIR,
            args.material,
        )
    )


    print(
        "dataset root  =",
        dataset_root,
    )


    dataset = GPUDataset(

        dataset_root,

        device,

        limit=args.limit,
    )


    n = len(
        dataset
    )


    if n < 2:

        raise RuntimeError(
            f"Dataset is too small: {n}"
        )


    print()
    print(
        "dataset samples =",
        n
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    if dataset.has_rollout_metadata:

        (
            train_indices,
            val_indices,
            train_rollouts,
            val_rollouts,

        ) = make_rollout_split(

            dataset=dataset,

            train_ratio=TRAIN_RATIO,

            seed=SEED,

            device=device,
        )


        print()
        print(
            "==============================================="
        )

        print(
            "ROLLOUT-LEVEL SPLIT"
        )

        print(
            "==============================================="
        )


        print(
            "train rollouts =",
            train_rollouts.numel()
        )


        print(
            "val rollouts   =",
            val_rollouts.numel()
        )


        print(
            "train samples  =",
            train_indices.numel()
        )


        print(
            "val samples    =",
            val_indices.numel()
        )


        print(
            "==============================================="
        )


    else:

        print()
        print(
            "WARNING:"
        )

        print(
            "Rollout metadata not found."
        )

        print(
            "Using sample-level random split."
        )


        (
            train_indices,
            val_indices,

        ) = make_sample_split(

            n=n,

            train_ratio=TRAIN_RATIO,

            seed=SEED,

            device=device,
        )


        train_rollouts = None

        val_rollouts = None


        print(
            "train samples =",
            train_indices.numel()
        )


        print(
            "val samples   =",
            val_indices.numel()
        )


    # ========================================================
    # NORMALIZED LOSS SCALES
    # ========================================================

    scale_l3d = None

    scale_lviz = None


    if args.loss_mode == "normalized":

        print()
        print(
            "Computing dataset-derived loss scales..."
        )


        (
            scale_l3d,
            scale_lviz,

        ) = compute_dataset_loss_scales(

            dataset=dataset,

            indices=train_indices,

            batch_size=BATCH_SIZE,

            device=device,
        )


        scale_values = torch.stack(
            [
                scale_l3d,
                scale_lviz,
            ]
        ).detach().cpu().tolist()


        print()
        print(
            "==============================================="
        )

        print(
            "FIXED LOSS SCALES"
        )

        print(
            "==============================================="
        )


        print(
            f"L3D baseline  = "
            f"{scale_values[0]:.9e}"
        )


        print(
            f"Lviz baseline = "
            f"{scale_values[1]:.9e}"
        )


        print(
            "==============================================="
        )


    # ========================================================
    # MODEL
    # ========================================================

    model = Param2Deform().to(
        device
    )


    if args.init_checkpoint is not None:

        if not os.path.isfile(
            args.init_checkpoint
        ):

            raise FileNotFoundError(
                args.init_checkpoint
            )


        initial_checkpoint = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=False,
        )


        if (
            isinstance(
                initial_checkpoint,
                dict,
            )
            and
            "model" in initial_checkpoint
        ):

            initial_state_dict = (
                initial_checkpoint[
                    "model"
                ]
            )

        else:

            initial_state_dict = (
                initial_checkpoint
            )


        model.load_state_dict(
            initial_state_dict,
            strict=True,
        )


        print()
        print(
            "loaded initial model =",
            args.init_checkpoint,
        )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=args.learning_rate,
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        (
            f"param2deform_"
            f"{args.material}_"
            f"{args.loss_mode}_"
            f"{args.tag}.pt"
        ),
    )


    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )


    best = float(
        "inf"
    )


    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(
        args.epochs
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        shuffled = train_indices[

            torch.randperm(
                train_indices.shape[
                    0
                ],
                device=device,
            )
        ]


        train_loss_sum = torch.zeros(
            (),
            device=device,
        )


        train_l3d_sum = torch.zeros(
            (),
            device=device,
        )


        train_viz_sum = torch.zeros(
            (),
            device=device,
        )


        train_component_3d_sum = torch.zeros(
            (),
            device=device,
        )


        train_component_viz_sum = torch.zeros(
            (),
            device=device,
        )


        train_batches = 0


        # ====================================================
        # TRAIN BATCHES
        # ====================================================

        for start in range(

            0,

            shuffled.shape[
                0
            ],

            BATCH_SIZE,
        ):

            idx = shuffled[

                start:
                start
                +
                BATCH_SIZE
            ]


            batch = dataset.get_batch(
                idx
            )


            # ------------------------------------------------
            # PARAM2DEFORM
            #
            # INPUT:
            #
            #   state_rep:
            #       canonical state
            #       + spatial gradients
            #
            #   params:
            #       normalized [length, depth]
            #
            # OUTPUT:
            #
            #   canonical deformation
            # ------------------------------------------------

            pred_delta = forward_model_amp(
                model,
                batch["state_rep"],
                batch["params"],
            )


            # ------------------------------------------------
            # PREDICTED NEXT STATE
            # ------------------------------------------------

            pred = (

                batch[
                    "state"
                ]

                +

                pred_delta
            )


            # ------------------------------------------------
            # GROUND TRUTH NEXT STATE
            # ------------------------------------------------

            gt = batch[
                "next_state"
            ]


            # ------------------------------------------------
            # RAW PAPER METRICS
            # ------------------------------------------------

            l3d = depth_mse(
                pred,
                gt,
            )


            lviz = visual_mse(
                pred,
                gt,
            )


            # ------------------------------------------------
            # OBJECTIVE
            # ------------------------------------------------

            if args.loss_mode == "paper":

                (
                    loss,
                    comp_3d,
                    comp_viz,

                ) = paper_loss(

                    l3d=l3d,

                    lviz=lviz,

                    w_3d=args.w_3d,

                    w_viz=args.w_viz,
                )


            else:

                (
                    loss,
                    comp_3d,
                    comp_viz,

                ) = normalized_loss(

                    l3d=l3d,

                    lviz=lviz,

                    scale_l3d=scale_l3d,

                    scale_lviz=scale_lviz,

                    lambda_viz=args.lambda_viz,
                )


            # ------------------------------------------------
            # OPTIMIZE
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            if (
                args.grad_clip is not None
                and
                args.grad_clip > 0.0
            ):

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=args.grad_clip,
                )


            optimizer.step()


            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            train_loss_sum += (
                loss.detach()
            )


            train_l3d_sum += (
                l3d.detach()
            )


            train_viz_sum += (
                lviz.detach()
            )


            train_component_3d_sum += (
                comp_3d.detach()
            )


            train_component_viz_sum += (
                comp_viz.detach()
            )


            train_batches += 1


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        val_loss_sum = torch.zeros(
            (),
            device=device,
        )


        val_l3d_sum = torch.zeros(
            (),
            device=device,
        )


        val_viz_sum = torch.zeros(
            (),
            device=device,
        )


        val_component_3d_sum = torch.zeros(
            (),
            device=device,
        )


        val_component_viz_sum = torch.zeros(
            (),
            device=device,
        )


        val_batches = 0


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


                pred_delta = forward_model_amp(
                    model,
                    batch["state_rep"],
                    batch["params"],
                )


                pred = (

                    batch[
                        "state"
                    ]

                    +

                    pred_delta
                )


                gt = batch[
                    "next_state"
                ]


                l3d = depth_mse(
                    pred,
                    gt,
                )


                lviz = visual_mse(
                    pred,
                    gt,
                )


                if args.loss_mode == "paper":

                    (
                        loss,
                        comp_3d,
                        comp_viz,

                    ) = paper_loss(

                        l3d=l3d,

                        lviz=lviz,

                        w_3d=args.w_3d,

                        w_viz=args.w_viz,
                    )


                else:

                    (
                        loss,
                        comp_3d,
                        comp_viz,

                    ) = normalized_loss(

                        l3d=l3d,

                        lviz=lviz,

                        scale_l3d=scale_l3d,

                        scale_lviz=scale_lviz,

                        lambda_viz=args.lambda_viz,
                    )


                val_loss_sum += (
                    loss.detach()
                )


                val_l3d_sum += (
                    l3d.detach()
                )


                val_viz_sum += (
                    lviz.detach()
                )


                val_component_3d_sum += (
                    comp_3d.detach()
                )


                val_component_viz_sum += (
                    comp_viz.detach()
                )


                val_batches += 1


        # ====================================================
        # EPOCH AVERAGES
        # ====================================================

        train_loss_tensor = (
            train_loss_sum
            /
            train_batches
        )


        train_l3d_tensor = (
            train_l3d_sum
            /
            train_batches
        )


        train_viz_tensor = (
            train_viz_sum
            /
            train_batches
        )


        train_component_3d_tensor = (
            train_component_3d_sum
            /
            train_batches
        )


        train_component_viz_tensor = (
            train_component_viz_sum
            /
            train_batches
        )


        val_loss_tensor = (
            val_loss_sum
            /
            val_batches
        )


        val_l3d_tensor = (
            val_l3d_sum
            /
            val_batches
        )


        val_viz_tensor = (
            val_viz_sum
            /
            val_batches
        )


        val_component_3d_tensor = (
            val_component_3d_sum
            /
            val_batches
        )


        val_component_viz_tensor = (
            val_component_viz_sum
            /
            val_batches
        )


        # ====================================================
        # ONE GPU -> CPU SYNC PER EPOCH
        # ====================================================

        values = torch.stack(
            [
                train_loss_tensor,

                train_l3d_tensor,

                train_viz_tensor,

                train_component_3d_tensor,

                train_component_viz_tensor,

                val_loss_tensor,

                val_l3d_tensor,

                val_viz_tensor,

                val_component_3d_tensor,

                val_component_viz_tensor,
            ]
        )


        values_cpu = (
            values
            .detach()
            .cpu()
            .tolist()
        )


        (
            train_loss,
            train_l3d,
            train_viz,
            train_comp_3d,
            train_comp_viz,

            val_loss,
            val_l3d,
            val_viz,
            val_comp_3d,
            val_comp_viz,

        ) = values_cpu


        # ====================================================
        # LOG
        # ====================================================

        if args.loss_mode == "paper":

            print(

                f"[{epoch + 1:03d}/{args.epochs}] "

                f"train={train_loss:.6e} "

                f"val={val_loss:.6e} "

                f"| L3D={val_l3d:.6e} "

                f"Lviz={val_viz:.6e} "

                f"| weighted "
                f"3D={args.w_3d * val_l3d:.6e} "
                f"viz={args.w_viz * val_viz:.6e}"
            )


        else:

            print(

                f"[{epoch + 1:03d}/{args.epochs}] "

                f"train={train_loss:.6e} "

                f"val={val_loss:.6e} "

                f"| raw "
                f"L3D={val_l3d:.6e} "
                f"Lviz={val_viz:.6e} "

                f"| norm "
                f"L3D={val_comp_3d:.4f} "
                f"Lviz={val_comp_viz:.4f}"
            )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_loss < best:

            best = val_loss


            checkpoint = {

                "model":
                model.state_dict(),

                "material":
                args.material,

                "dataset_root":
                dataset_root,

                "learning_rate":
                args.learning_rate,

                "init_checkpoint":
                args.init_checkpoint,

                "loss_mode":
                args.loss_mode,

                "limit":
                args.limit,

                "epochs":
                args.epochs,

                "best_val_loss":
                best,

                "train_indices":
                train_indices.detach().cpu(),

                "val_indices":
                val_indices.detach().cpu(),

                "train_ratio":
                TRAIN_RATIO,

                "seed":
                SEED,
            }


            # ------------------------------------------------
            # PAPER LOSS METADATA
            # ------------------------------------------------

            if args.loss_mode == "paper":

                checkpoint[
                    "w_3d"
                ] = args.w_3d


                checkpoint[
                    "w_viz"
                ] = args.w_viz


            # ------------------------------------------------
            # NORMALIZED LOSS METADATA
            # ------------------------------------------------

            else:

                checkpoint[
                    "lambda_viz"
                ] = args.lambda_viz


                checkpoint[
                    "loss_scale_l3d"
                ] = scale_l3d.detach().cpu()


                checkpoint[
                    "loss_scale_lviz"
                ] = scale_lviz.detach().cpu()


            # ------------------------------------------------
            # ROLLOUT SPLIT METADATA
            # ------------------------------------------------

            if train_rollouts is not None:

                checkpoint[
                    "train_rollouts"
                ] = train_rollouts.detach().cpu()


            if val_rollouts is not None:

                checkpoint[
                    "val_rollouts"
                ] = val_rollouts.detach().cpu()


            torch.save(
                checkpoint,
                checkpoint_path,
            )


            print(
                "saved:",
                checkpoint_path
            )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "==============================================="
    )


    print(
        "best validation loss =",
        best
    )


    print(
        "checkpoint =",
        checkpoint_path
    )


    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()