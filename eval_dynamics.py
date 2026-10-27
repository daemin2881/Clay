import argparse
import os

import torch

from vs_config import (
    DATASET_DIR,
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
    sinkhorn_emd,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--material",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()


    device = torch.device(
        "cuda"
    )


    dataset = GPUDataset(
        os.path.join(
            DATASET_DIR,
            args.material,
        ),
        device,
    )


    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )


    model = Param2Deform().to(
        device
    )


    model.load_state_dict(
        ckpt[
            "model"
        ]
    )


    model.eval()


    if (
        "test_indices"
        in ckpt
    ):

        indices = ckpt[
            "test_indices"
        ].to(
            device
        )

    else:

        indices = torch.arange(
            len(
                dataset
            ),
            device=device,
        )


    l3d_sum = torch.zeros(
        (),
        device=device,
    )

    lviz_sum = torch.zeros(
        (),
        device=device,
    )

    cd_sum = torch.zeros(
        (),
        device=device,
    )

    emd_sum = torch.zeros(
        (),
        device=device,
    )

    count = 0


    with torch.no_grad():

        for idx in indices:

            batch = dataset.get_batch(
                idx[
                    None
                ]
            )


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


            l3d_sum += depth_mse(
                pred,
                target
            )

            lviz_sum += visual_mse(
                pred,
                target
            )

            cd_sum += chamfer_distance(
                pred,
                target
            )

            emd_sum += sinkhorn_emd(
                pred,
                target
            )

            count += 1


    results = torch.stack(
        [
            l3d_sum / count,
            lviz_sum / count,
            cd_sum / count,
            emd_sum / count,
        ]
    )


    # INTENTIONAL final sync
    values = (
        results
        .detach()
        .cpu()
        .tolist()
    )


    print()

    print(
        "==============================================="
    )

    print(
        "DYNAMICS EVALUATION"
    )

    print(
        "==============================================="
    )

    print(
        "L3D :",
        values[0]
    )

    print(
        "Lviz:",
        values[1]
    )

    print(
        "CD  :",
        values[2]
    )

    print(
        "EMD :",
        values[3]
    )

    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()