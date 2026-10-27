import argparse
import csv
import os

import torch

from vs_config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    RESULT_DIR,
    SAMPLE_COUNTS,
    NOISE_LEVELS,
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
    chamfer_distance,
)

from vs_planner import (
    predict_next_batch,
    add_normalized_action_noise,
)


# ============================================================
# CROSS MATERIAL
# Fig. 6
# ============================================================

def experiment_cross_material(
    checkpoints,
):

    device = torch.device(
        "cuda"
    )


    materials = [
        "soft",
        "medium",
        "hard",
    ]


    output = os.path.join(
        RESULT_DIR,
        "exp03_cross_material.csv",
    )


    with open(
        output,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "train_material",
                "test_material",
                "L3D",
            ]
        )


        for train_material in materials:

            checkpoint_path = checkpoints[
                train_material
            ]


            model = Param2Deform().to(
                device
            )


            ckpt = torch.load(
                checkpoint_path,
                map_location=device,
            )


            model.load_state_dict(
                ckpt[
                    "model"
                ]
            )


            model.eval()


            for test_material in materials:

                dataset = GPUDataset(
                    os.path.join(
                        DATASET_DIR,
                        test_material,
                    ),
                    device,
                )


                n = min(
                    100,
                    len(
                        dataset
                    ),
                )


                indices = torch.arange(
                    n,
                    device=device,
                )


                batch = dataset.get_batch(
                    indices
                )


                with torch.no_grad():

                    pred = (
                        batch[
                            "state"
                        ]
                        +
                        model(
                            batch[
                                "state_rep"
                            ],
                            batch[
                                "params"
                            ],
                        )
                    )


                    target = (
                        batch[
                            "state"
                        ]
                        +
                        batch[
                            "delta"
                        ]
                    )


                    loss = depth_mse(
                        pred,
                        target
                    )


                value = float(
                    loss.item()
                )


                writer.writerow(
                    [
                        train_material,
                        test_material,
                        value,
                    ]
                )


                print(
                    train_material,
                    "->",
                    test_material,
                    value,
                )


    print(
        "saved:",
        output
    )


# ============================================================
# NOISE
# Fig. 14
# ============================================================

def experiment_noise(
    checkpoint,
    material,
):

    device = torch.device(
        "cuda"
    )


    dataset = GPUDataset(
        os.path.join(
            DATASET_DIR,
            material,
        ),
        device,
    )


    model = Param2Deform().to(
        device
    )


    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )


    model.load_state_dict(
        ckpt[
            "model"
        ]
    )


    model.eval()


    n = min(
        64,
        len(
            dataset
        ),
    )


    indices = torch.arange(
        n,
        device=device,
    )


    batch = dataset.get_batch(
        indices
    )


    states = batch[
        "state"
    ]


    actions = batch[
        "action"
    ]


    ground_truth = (
        batch[
            "state"
        ]
        +
        batch[
            "delta"
        ]
    )


    output = os.path.join(
        RESULT_DIR,
        "exp09_noise.csv",
    )


    with open(
        output,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "sigma",
                "L3D",
                "Lviz",
                "CD",
            ]
        )


        for sigma in NOISE_LEVELS:

            noisy_actions = (
                add_normalized_action_noise(
                    actions,
                    sigma,
                )
            )


            with torch.no_grad():

                pred = predict_next_batch(
                    model,
                    states,
                    noisy_actions,
                )


                l3d = depth_mse(
                    pred,
                    ground_truth,
                )

                lviz = visual_mse(
                    pred,
                    ground_truth,
                )

                cd = chamfer_distance(
                    pred,
                    ground_truth,
                )


            values = torch.stack(
                [
                    l3d,
                    lviz,
                    cd,
                ]
            ).detach().cpu().tolist()


            writer.writerow(
                [
                    sigma,
                    *values,
                ]
            )


            print(
                sigma,
                values
            )


    print(
        "saved:",
        output
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--exp",
        required=True,
        choices=[
            "cross_material",
            "noise",
        ],
    )


    parser.add_argument(
        "--checkpoint",
        default=None,
    )


    parser.add_argument(
        "--material",
        default="medium",
    )


    parser.add_argument(
        "--soft_checkpoint",
        default=None,
    )

    parser.add_argument(
        "--medium_checkpoint",
        default=None,
    )

    parser.add_argument(
        "--hard_checkpoint",
        default=None,
    )


    args = parser.parse_args()


    if args.exp == "noise":

        if args.checkpoint is None:

            raise ValueError(
                "--checkpoint required"
            )


        experiment_noise(
            args.checkpoint,
            args.material,
        )


    elif args.exp == "cross_material":

        checkpoints = {

            "soft":
            args.soft_checkpoint,

            "medium":
            args.medium_checkpoint,

            "hard":
            args.hard_checkpoint,
        }


        experiment_cross_material(
            checkpoints
        )


if __name__ == "__main__":

    main()