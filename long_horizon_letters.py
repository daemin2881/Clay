import argparse
import csv
import os

import torch

from vs_config import (
    RESULT_DIR,
    PLAN_HORIZON,
    MPC_EXECUTE,
)

from vs_dynamics import (
    Param2Deform,
)

from vs_targets import (
    make_letter_target,
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


# ============================================================
# PAPER FIG.2
#
# cumulative actions:
#
# A 190
# B 250
# C 310
# D 370
# E 430
# F 490
#
# We preserve these cumulative milestones.
# ============================================================

CUMULATIVE_ACTION_TARGETS = {

    "A":
    190,

    "B":
    250,

    "C":
    310,

    "D":
    370,

    "E":
    430,

    "F":
    490,
}


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True,
    )


    parser.add_argument(
        "--viewer",
        action="store_true",
    )


    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
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


    args = parser.parse_args()


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


    for p in model.parameters():

        p.requires_grad_(
            False
        )


    # ========================================================
    # ONE SCENE ONLY
    # ========================================================

    sim.initialize()


    root_dir = ensure_dir(

        os.path.join(

            RESULT_DIR,

            "exp05_long_horizon_A_to_F",
        )
    )


    global_csv = os.path.join(

        root_dir,

        "long_horizon_actions.csv",
    )


    cumulative_action = 0


    with open(

        global_csv,

        "w",

        newline="",

    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "letter",
                "global_action",
                "x",
                "y",
                "theta",
                "length",
                "depth",
                "L3D",
                "Lviz",
                "Chamfer",
            ]
        )


        # ====================================================
        # A -> F
        # ====================================================

        for letter in [

            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]:

            print()

            print(
                "###############################################"
            )

            print(
                f"# TARGET {letter}"
            )

            print(
                "###############################################"
            )


            letter_dir = ensure_dir(

                os.path.join(

                    root_dir,

                    letter,
                )
            )


            state_dir = ensure_dir(

                os.path.join(

                    letter_dir,

                    "states",
                )
            )


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

                "start",
            )


            # =================================================
            # RAW LETTER
            # =================================================

            raw_target = make_letter_target(

                letter,

                device,
            )


            save_state_bundle(

                raw_target,

                state_dir,

                "raw_target",
            )


            # =================================================
            # TARGET ADJUSTMENT AGAINST CURRENT STATE
            #
            # IMPORTANT:
            # current = previous letter final state
            # =================================================

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


            target_cumulative = (
                CUMULATIVE_ACTION_TARGETS[
                    letter
                ]
            )


            # =================================================
            # CONTINUE WITHOUT RESET
            # =================================================

            while (
                cumulative_action
                <
                target_cumulative
            ):

                current = (

                    sim
                    .get_height_gpu()
                    [
                        None,
                        None
                    ]
                )


                initial = greedy_plan_gpu(

                    model,

                    current,

                    target,

                    horizon=PLAN_HORIZON,

                    w3d=args.w3d,

                    wviz=args.wviz,
                )


                plan = refine_plan_gpu(

                    model,

                    current,

                    target,

                    initial,

                    w3d=args.w3d,

                    wviz=args.wviz,
                )


                execute_n = min(

                    MPC_EXECUTE,

                    target_cumulative
                    -
                    cumulative_action,
                )


                for i in range(
                    execute_n
                ):

                    action = plan[
                        i
                    ]


                    sim.linear_push(

                        action,

                        action_index=(
                            cumulative_action
                        ),
                    )


                    cumulative_action += 1


                    after = (

                        sim
                        .get_height_gpu()
                        [
                            None,
                            None
                        ]
                    )


                    _, l3d, lviz = total_loss(

                        after,

                        target,

                        args.w3d,

                        args.wviz,
                    )


                    cd = chamfer_distance(

                        after,

                        target,
                    )


                    values = torch.cat(

                        [
                            action,

                            torch.stack(
                                [
                                    l3d,
                                    lviz,
                                    cd,
                                ]
                            ),
                        ]

                    ).detach().cpu().tolist()


                    writer.writerow(
                        [
                            letter,
                            cumulative_action,
                            *values,
                        ]
                    )


                    file.flush()


                    print(
                        f"{letter} "
                        f"action={cumulative_action:03d} "
                        f"L3D={values[5]:.6e} "
                        f"Lviz={values[6]:.6e} "
                        f"CD={values[7]:.6e}"
                    )


                    if (
                        cumulative_action
                        %
                        args.save_every
                        ==
                        0
                    ):

                        save_state_bundle(

                            after,

                            state_dir,

                            (
                                f"global_action_"
                                f"{cumulative_action:04d}"
                            ),
                        )


            # =================================================
            # FINAL LETTER STATE
            # =================================================

            final_letter = (

                sim
                .get_height_gpu()
                [
                    None,
                    None
                ]
            )


            save_state_bundle(

                final_letter,

                state_dir,

                f"{letter}_final",
            )


            print()

            print(
                f"{letter} complete @ "
                f"{cumulative_action} actions"
            )


    print()

    print(
        "==============================================="
    )

    print(
        "A -> F LONG HORIZON COMPLETE"
    )

    print(
        "==============================================="
    )

    print(
        "total actions:",
        cumulative_action,
    )

    print(
        "output:",
        root_dir,
    )


if __name__ == "__main__":

    main()