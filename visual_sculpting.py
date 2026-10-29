import argparse
import csv
import gc
import os

import torch

from vs_config import (
    RESULT_DIR,
    MPC_EXECUTE,
    TARGET_MEAN_ERROR,
    RECENT_ACTION_HISTORY,
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

    # ========================================================
    # BASIC
    # ========================================================

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
        "--max-actions",
        "--max_actions",
        dest="max_actions",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    parser.add_argument(
        "--save-every",
        "--save_every",
        dest="save_every",
        type=int,
        default=5,
    )

    # ========================================================
    # LOW-MEMORY PLANNER
    # ========================================================

    parser.add_argument(
        "--plan-horizon",
        type=int,
        default=5,
        help=(
            "MPC planning horizon. "
            "Use 5 for the 7.5 GB GPU smoke test."
        ),
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help=(
            "Number of candidate actions per greedy step."
        ),
    )

    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help=(
            "Number of candidate predictions evaluated "
            "simultaneously. Use 1 on an 8 GB GPU."
        ),
    )

    parser.add_argument(
        "--refine-actions",
        type=int,
        default=5,
        help=(
            "Maximum number of leading actions refined."
        ),
    )

    parser.add_argument(
        "--refine-steps",
        type=int,
        default=0,
        help=(
            "Differentiable refinement iterations. "
            "Use 0 for the initial low-memory test."
        ),
    )

    args = parser.parse_args()

    if args.max_actions <= 0:
        raise ValueError(
            "--max-actions must be positive."
        )

    if args.save_every <= 0:
        raise ValueError(
            "--save-every must be positive."
        )

    if args.plan_horizon <= 0:
        raise ValueError(
            "--plan-horizon must be positive."
        )

    if args.candidates <= 0:
        raise ValueError(
            "--candidates must be positive."
        )

    if args.candidate_batch_size <= 0:
        raise ValueError(
            "--candidate-batch-size must be positive."
        )

    if args.refine_actions < 0:
        raise ValueError(
            "--refine-actions cannot be negative."
        )

    if args.refine_steps < 0:
        raise ValueError(
            "--refine-steps cannot be negative."
        )

    # ========================================================
    # VIEWER ENVIRONMENT
    # ========================================================

    os.environ["VS_SHOW_VIEWER"] = (
        "1"
        if args.viewer
        else "0"
    )

    # Simulator import 이후 Genesis가 초기화된다.
    import vs_simulator as sim

    # sim.DEVICE는 initialize 이전에는 None일 수 있다.
    sim.initialize()

    initial_height = sim.get_height_gpu()

    device = initial_height.device
    dtype = initial_height.dtype

    print()
    print(
        "==============================================="
    )
    print(
        "SCULPTING DEVICE"
    )
    print(
        "==============================================="
    )
    print(
        "device               =",
        device,
    )
    print(
        "dtype                =",
        dtype,
    )
    print(
        "plan horizon         =",
        args.plan_horizon,
    )
    print(
        "candidates           =",
        args.candidates,
    )
    print(
        "candidate batch size =",
        args.candidate_batch_size,
    )
    print(
        "refine actions       =",
        args.refine_actions,
    )
    print(
        "refine steps         =",
        args.refine_steps,
    )
    print(
        "==============================================="
    )

    # ========================================================
    # HEIGHT STATE HELPER
    # ========================================================

    def get_current_state():

        height = torch.as_tensor(
            sim.get_height_gpu(),
            device=device,
            dtype=dtype,
        )

        if height.ndim == 4:
            height = height[
                0,
                0
            ]

        elif height.ndim == 3:
            height = height[
                0
            ]

        if height.ndim != 2:
            raise RuntimeError(
                "Expected a 2D height map, "
                f"but got shape={tuple(height.shape)}"
            )

        return height[
            None,
            None
        ]

    # ========================================================
    # OUTPUT
    # ========================================================

    experiment_name = (
        f"sculpt_"
        f"{args.target}_"
        f"w3d{args.w3d}_"
        f"wviz{args.wviz}"
    )

    experiment_dir = ensure_dir(
        os.path.join(
            RESULT_DIR,
            experiment_name,
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
    #
    # 체크포인트 전체를 GPU에 올리지 않는다.
    # optimizer state가 포함되어 있으면 VRAM을 낭비하기 때문이다.
    # ========================================================

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model = Param2Deform()

    model.load_state_dict(
        checkpoint["model"]
    )

    del checkpoint

    gc.collect()

    model = model.to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # INITIAL STATE
    # ========================================================

    current = get_current_state()

    save_state_bundle(
        current,
        state_dir,
        "initial",
    )

    # initial_height는 더 이상 필요하지 않다.
    del initial_height

    # ========================================================
    # RAW TARGET
    # ========================================================

    raw_target = make_target(
        args.target,
        device,
    )

    raw_target = torch.as_tensor(
        raw_target,
        device=device,
        dtype=dtype,
    )

    if raw_target.ndim == 4:
        raw_target = raw_target[
            0,
            0
        ]

    elif raw_target.ndim == 3:
        raw_target = raw_target[
            0
        ]

    if raw_target.ndim != 2:
        raise RuntimeError(
            "Expected a 2D target height map, "
            f"but got shape={tuple(raw_target.shape)}"
        )

    print()
    print(
        "==============================================="
    )
    print(
        "TARGET DEVICE CHECK"
    )
    print(
        "==============================================="
    )
    print(
        "current device =",
        current.device,
    )
    print(
        "target device  =",
        raw_target.device,
    )
    print(
        "current dtype  =",
        current.dtype,
    )
    print(
        "target dtype   =",
        raw_target.dtype,
    )
    print(
        "target shape   =",
        tuple(raw_target.shape),
    )
    print(
        "==============================================="
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

    adjusted = adjusted.to(
        device=device,
        dtype=dtype,
    )

    target = adjusted[
        None,
        None
    ].detach()

    save_state_bundle(
        target,
        state_dir,
        "adjusted_target",
    )

    # target adjustment 임시 텐서 정리
    del raw_target
    del adjusted

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # INITIAL ERROR
    # ========================================================

    initial_mae = torch.mean(
        torch.abs(
            current
            -
            target
        )
    )

    initial_mae_value = float(
        initial_mae.item()
    )

    print()
    print(
        "initial target MAE =",
        initial_mae_value
        *
        1000.0,
        "mm",
    )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = os.path.join(
        experiment_dir,
        "actions.csv",
    )

    action_count = 0

    recent_action_list = []

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

            current = get_current_state()

            mean_error = torch.mean(
                torch.abs(
                    current
                    -
                    target
                )
            )

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

            remaining_actions = (
                args.max_actions
                -
                action_count
            )

            planning_horizon = min(
                args.plan_horizon,
                remaining_actions,
            )

            if recent_action_list:
                recent_actions = torch.stack(
                    recent_action_list,
                    dim=0,
                )
            else:
                recent_actions = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================
            # GREEDY INITIALIZATION
            #
            # 후보 전체를 한꺼번에 예측하지 않는다.
            # candidate_batch_size 단위로 나누어 계산한다.
            # =================================================

            print()
            print(
                "[PLANNER] greedy initialization"
            )
            print(
                "[PLANNER] horizon =",
                planning_horizon,
            )
            print(
                "[PLANNER] candidates =",
                args.candidates,
            )
            print(
                "[PLANNER] candidate batch =",
                args.candidate_batch_size,
            )

            initial_plan = greedy_plan_gpu(
                model,
                current,
                target,
                horizon=planning_horizon,
                candidates=args.candidates,
                w3d=args.w3d,
                wviz=args.wviz,
                recent_actions=recent_actions,
                candidate_batch_size=(
                    args.candidate_batch_size
                ),
            )

            initial_plan = initial_plan.to(
                device=device,
                dtype=dtype,
            )

            # =================================================
            # DIFFERENTIABLE REFINEMENT
            #
            # refine_steps=0이면 greedy plan을 그대로 사용한다.
            # =================================================

            if (
                args.refine_steps
                >
                0
                and
                args.refine_actions
                >
                0
            ):

                print()
                print(
                    "[PLANNER] differentiable refinement"
                )
                print(
                    "[PLANNER] actions =",
                    min(
                        args.refine_actions,
                        initial_plan.shape[0],
                    ),
                )
                print(
                    "[PLANNER] steps =",
                    args.refine_steps,
                )

                refined_plan = refine_plan_gpu(
                    model,
                    current,
                    target,
                    initial_plan,
                    w3d=args.w3d,
                    wviz=args.wviz,
                    recent_actions=recent_actions,
                    max_refine_actions=(
                        args.refine_actions
                    ),
                    refine_steps=(
                        args.refine_steps
                    ),
                )

            else:

                print()
                print(
                    "[PLANNER] refinement skipped"
                )

                refined_plan = initial_plan

            refined_plan = refined_plan.to(
                device=device,
                dtype=dtype,
            )

            execute_n = min(
                MPC_EXECUTE,
                refined_plan.shape[0],
                remaining_actions,
            )

            # =================================================
            # EXECUTE FIRST FEW ACTIONS
            # =================================================

            for i in range(
                execute_n
            ):

                action = refined_plan[
                    i
                ].detach()

                print()
                print(
                    f"[EXECUTE {action_count + 1:03d}] "
                    f"x={float(action[0].item()):.4f} "
                    f"y={float(action[1].item()):.4f} "
                    f"theta={float(action[2].item()):+.4f} "
                    f"length={float(action[3].item()) * 1000.0:.2f}mm "
                    f"depth={float(action[4].item()) * 1000.0:.2f}mm"
                )

                # 실제 Genesis 물리 실행
                sim.linear_push(
                    action,
                    action_index=action_count,
                )

                action_count += 1

                recent_action_list.append(
                    action.detach().clone()
                )

                if (
                    len(recent_action_list)
                    >
                    RECENT_ACTION_HISTORY
                ):

                    recent_action_list = (
                        recent_action_list[
                            -RECENT_ACTION_HISTORY:
                        ]
                    )

                after = get_current_state()

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

                record_tensor = torch.cat(
                    [
                        action.reshape(
                            -1
                        ),
                        torch.stack(
                            [
                                mae,
                                l3d,
                                lviz,
                                cd,
                            ]
                        ),
                    ]
                )

                record = (
                    record_tensor
                    .detach()
                    .cpu()
                    .tolist()
                )

                writer.writerow(
                    [
                        action_count,
                        *record,
                    ]
                )

                file.flush()

                print(
                    f"[ACTION {action_count:03d}] "
                    f"MAE={record[5] * 1000.0:.3f}mm "
                    f"L3D={record[6]:.6e} "
                    f"Lviz={record[7]:.6e} "
                    f"CD={record[8]:.6e}"
                )

                # =============================================
                # PERIODIC OUTPUT
                # =============================================

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

            del initial_plan
            del refined_plan

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # FINAL
    # ========================================================

    final_state = get_current_state()

    final_mae = torch.mean(
        torch.abs(
            final_state
            -
            target
        )
    )

    final_mae_value = float(
        final_mae.item()
    )

    save_state_bundle(
        final_state,
        state_dir,
        "final",
    )

    improvement = (
        initial_mae_value
        -
        final_mae_value
    )

    if initial_mae_value > 0.0:
        improvement_ratio = (
            improvement
            /
            initial_mae_value
            *
            100.0
        )
    else:
        improvement_ratio = 0.0

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
        "initial MAE:",
        initial_mae_value
        *
        1000.0,
        "mm",
    )
    print(
        "final MAE:",
        final_mae_value
        *
        1000.0,
        "mm",
    )
    print(
        "improvement:",
        improvement
        *
        1000.0,
        "mm",
    )
    print(
        "improvement ratio:",
        improvement_ratio,
        "%",
    )
    print(
        "output:",
        experiment_dir,
    )
    print(
        "==============================================="
    )


if __name__ == "__main__":
    main()