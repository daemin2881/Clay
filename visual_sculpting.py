import argparse
import csv
import os

import torch

from vs_config import (
    RESULT_DIR,
    PLAN_HORIZON,
    MPC_EXECUTE,
    TARGET_MEAN_ERROR,
)

from vs_dynamics import (
    Param2Deform,
)

from vs_targets import (
    make_target,
)

from vs_representation import (
    adjust_target_to_state_gpu,
)

from vs_planner import (
    greedy_plan_gpu,
    refine_plan_gpu,
)

from vs_metrics import (
    total_loss,
    chamfer_distance,
)

from vs_io import (
    ensure_dir,
    save_state_bundle,
)


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True,
    )


    parser.add_argument(
        "--target",
        default="X",
    )


    parser.add_argument(
        "--w3d",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--wviz",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--max_actions",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--viewer",
        action="store_true",
    )


    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
    )


    args = parser.parse_args()


    # ========================================================
    # IMPORT SIMULATOR AFTER VIEWER ENV
    # ========================================================

    os.environ[
        "VS_SHOW_VIEWER"
    ] = (
        "1"
        if args.viewer
        else
        "0"
    )


    import vs_simulator as sim


    device = sim.DEVICE


    # ========================================================
    # OUTPUT
    # ========================================================

    experiment_dir = ensure_dir(

        os.path.join(

            RESULT_DIR,

            (
                f"sculpt_"
                f"{args.target}_"
                f"w3d{args.w3d}_"
                f"wviz{args.wviz}"
            ),
        )
    )


    state_dir = ensure_dir(

        os.path.join(

            experiment_dir,

            "states",
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    checkpoint = torch.load(

        args.checkpoint,

        map_location=device,
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


    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )


    # ========================================================
    # INIT
    # ========================================================

    sim.initialize()


    current = (

        sim
        .get_height_gpu()
        [
            None,
            None
        ]
    )


    save_state_bundle(

        current,

        state_dir,

        "initial",
    )


    # ========================================================
    # RAW TARGET
    # ========================================================

    raw_target = make_target(

        args.target,

        device,
    )


    save_state_bundle(

        raw_target,

        state_dir,

        "raw_target",
    )


    # ========================================================
    # ADJUST TARGET
    # ========================================================

    adjusted = adjust_target_to_state_gpu(

        current[
            0,
            0
        ],

        raw_target,
    )


    target = adjusted[
        None,
        None
    ]


    save_state_bundle(

        target,

        state_dir,

        "adjusted_target",
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_path = os.path.join(

        experiment_dir,

        "actions.csv",
    )


    action_count = 0


    with open(

        csv_path,

        "w",

        newline="",

    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "action",
                "x",
                "y",
                "theta",
                "length",
                "depth",
                "mean_abs_error_m",
                "L3D",
                "Lviz",
                "Chamfer",
            ]
        )


        # ====================================================
        # MPC LOOP
        # ====================================================

        while (
            action_count
            <
            args.max_actions
        ):

            current = (

                sim
                .get_height_gpu()
                [
                    None,
                    None
                ]
            )


            mean_error = torch.mean(

                torch.abs(

                    current

                    -

                    target
                )
            )


            # only once per MPC iteration
            mean_value = float(
                mean_error.item()
            )


            print()

            print(
                "==============================================="
            )

            print(
                "MPC REPLAN"
            )

            print(
                "==============================================="
            )

            print(
                "actions:",
                action_count,
            )

            print(
                "mean error:",
                mean_value
                *
                1000.0,
                "mm",
            )


            if (
                mean_value
                <
                TARGET_MEAN_ERROR
            ):

                print(
                    "TARGET CONVERGED"
                )

                break


            # =================================================
            # GREEDY INITIALIZATION
            #
            # GPU batched candidates
            # >=40 actions
            # =================================================

            initial_plan = greedy_plan_gpu(

                model,

                current,

                target,

                horizon=PLAN_HORIZON,

                w3d=args.w3d,

                wviz=args.wviz,
            )


            # =================================================
            # DIFFERENTIABLE REFINEMENT
            # =================================================

            refined_plan = refine_plan_gpu(

                model,

                current,

                target,

                initial_plan,

                w3d=args.w3d,

                wviz=args.wviz,
            )


            execute_n = min(

                MPC_EXECUTE,

                refined_plan.shape[
                    0
                ],
            )


            # =================================================
            # EXECUTE FIRST FEW ACTIONS
            # =================================================

            for i in range(
                execute_n
            ):

                action = refined_plan[
                    i
                ]


                # physical execution
                sim.linear_push(

                    action,

                    action_index=(
                        action_count
                    ),
                )


                action_count += 1


                after = (

                    sim
                    .get_height_gpu()
                    [
                        None,
                        None
                    ]
                )


                total, l3d, lviz = total_loss(

                    after,

                    target,

                    args.w3d,

                    args.wviz,
                )


                cd = chamfer_distance(

                    after,

                    target,
                )


                mae = torch.mean(

                    torch.abs(

                        after

                        -

                        target
                    )
                )


                # INTENTIONAL GPU -> CPU
                # one record per physical action
                record = torch.cat(

                    [
                        action,

                        torch.stack(
                            [
                                mae,
                                l3d,
                                lviz,
                                cd,
                            ]
                        ),
                    ]

                ).detach().cpu().tolist()


                writer.writerow(
                    [
                        action_count,
                        *record,
                    ]
                )


                file.flush()


                print(
                    f"[ACTION {action_count:03d}] "
                    f"MAE={record[5]*1000:.3f}mm "
                    f"L3D={record[6]:.6e} "
                    f"Lviz={record[7]:.6e} "
                    f"CD={record[8]:.6e}"
                )


                # =================================================
                # PERIODIC PAPER-LIKE OUTPUT
                # =================================================

                if (
                    action_count
                    %
                    args.save_every
                    ==
                    0
                ):

                    save_state_bundle(

                        after,

                        state_dir,

                        f"action_{action_count:04d}",
                    )


                if (
                    action_count
                    >=
                    args.max_actions
                ):

                    break


    # ========================================================
    # FINAL
    # ========================================================

    final_state = (

        sim
        .get_height_gpu()
        [
            None,
            None
        ]
    )


    save_state_bundle(

        final_state,

        state_dir,

        "final",
    )


    print()

    print(
        "==============================================="
    )

    print(
        "SCULPTING COMPLETE"
    )

    print(
        "==============================================="
    )

    print(
        "target:",
        args.target,
    )

    print(
        "actions:",
        action_count,
    )

    print(
        "output:",
        experiment_dir,
    )


if __name__ == "__main__":

    main()