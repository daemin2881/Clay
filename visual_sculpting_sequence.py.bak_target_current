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
from vs_dynamics import Param2Deform
from vs_targets import make_target
from vs_representation import adjust_target_to_state_gpu
from vs_planner import (
    greedy_plan_gpu,
    refine_plan_gpu,
    predict_next,
)
from vs_metrics import (
    total_loss,
    chamfer_distance,
)
from vs_io import (
    ensure_dir,
    save_state_bundle,
)


def parse_targets(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def parse_budgets(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--letters",
        default="A,B,C,D,E,F",
    )
    parser.add_argument(
        "--letter-action-budgets",
        default="190,60,60,60,60,60",
    )
    parser.add_argument(
        "--completion-mode",
        choices=["paper", "auto"],
        default="paper",
    )
    parser.add_argument(
        "--convergence-checks",
        type=int,
        default=3,
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
        "--plan-horizon",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--refine-actions",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    parser.add_argument(
        "--mpc-execute",
        type=int,
        default=5,
        help=(
            "Number of planned actions executed "
            "before observing and replanning."
        ),
    )

    args = parser.parse_args()

    if args.mpc_execute <= 0:
        raise ValueError(
            "--mpc-execute must be positive."
        )

    targets = parse_targets(
        args.letters
    )
    budgets = parse_budgets(
        args.letter_action_budgets
    )

    if not targets:
        raise ValueError(
            "--letters contains no targets."
        )

    if len(budgets) == 1:
        budgets = budgets * len(targets)

    if len(budgets) != len(targets):
        raise ValueError(
            "Action budgets must contain one value or "
            "one value for each target."
        )

    if any(value <= 0 for value in budgets):
        raise ValueError(
            "Every action budget must be positive."
        )

    if args.convergence_checks <= 0:
        raise ValueError(
            "--convergence-checks must be positive."
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

    if args.save_every <= 0:
        raise ValueError(
            "--save-every must be positive."
        )

    os.environ["VS_SHOW_VIEWER"] = (
        "1" if args.viewer else "0"
    )

    # 전체 A→F 실험에서 단 한 번만 초기화한다.
    import vs_simulator as sim

    sim.initialize()

    initial_height = sim.get_height_gpu()
    device = initial_height.device
    dtype = initial_height.dtype

    def get_current_state():
        height = torch.as_tensor(
            sim.get_height_gpu(),
            device=device,
            dtype=dtype,
        )

        if height.ndim == 4:
            height = height[0, 0]
        elif height.ndim == 3:
            height = height[0]

        if height.ndim != 2:
            raise RuntimeError(
                f"Invalid height shape: {tuple(height.shape)}"
            )

        return height[None, None]

    print()
    print("===============================================")
    print("ALPHABET SEQUENCE")
    print("===============================================")
    print("targets             =", targets)
    print("action budgets      =", budgets)
    print("completion mode     =", args.completion_mode)
    print(
        "MAE threshold       =",
        TARGET_MEAN_ERROR * 1000.0,
        "mm",
    )
    print("convergence checks  =", args.convergence_checks)
    print("plan horizon        =", args.plan_horizon)
    print("MPC execute         =", args.mpc_execute)
    print("===============================================")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = Param2Deform()
    model.load_state_dict(
        checkpoint["model"]
    )

    del checkpoint

    model = model.to(
        device=device,
        dtype=dtype,
    )
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sequence_name = "".join(targets)

    experiment_dir = ensure_dir(
        os.path.join(
            RESULT_DIR,
            (
                f"sculpt_sequence_{sequence_name}_"
                f"w3d{args.w3d}_wviz{args.wviz}"
            ),
        )
    )

    state_root = ensure_dir(
        os.path.join(
            experiment_dir,
            "states",
        )
    )

    save_state_bundle(
        get_current_state(),
        state_root,
        "sequence_initial",
    )

    actions_csv_path = os.path.join(
        experiment_dir,
        "actions.csv",
    )
    summary_csv_path = os.path.join(
        experiment_dir,
        "sequence_summary.csv",
    )

    cumulative_actions = 0
    recent_action_list = []

    with (
        open(
            actions_csv_path,
            "w",
            newline="",
        ) as action_file,
        open(
            summary_csv_path,
            "w",
            newline="",
        ) as summary_file,
    ):
        action_writer = csv.writer(
            action_file
        )
        summary_writer = csv.writer(
            summary_file
        )

        action_writer.writerow(
            [
                "target",
                "target_action",
                "cumulative_action",
                "x",
                "y",
                "theta",
                "length",
                "depth",
                "before_mae_m",
                "predicted_mae_m",
                "actual_mae_m",
                "prediction_actual_mae_m",
                "L3D",
                "Lviz",
                "Chamfer",
            ]
        )

        summary_writer.writerow(
            [
                "target",
                "target_actions",
                "cumulative_actions",
                "initial_mae_mm",
                "final_mae_mm",
                "improvement_mm",
                "improvement_percent",
                "L3D",
                "Lviz",
                "Chamfer",
                "status",
            ]
        )

        for target_index, (
            target_name,
            action_budget,
        ) in enumerate(
            zip(targets, budgets)
        ):
            # 이전 글자의 실제 최종 상태를 그대로 사용한다.
            start_state = get_current_state()

            target_dir = ensure_dir(
                os.path.join(
                    state_root,
                    f"{target_index + 1:02d}_{target_name}",
                )
            )

            save_state_bundle(
                start_state,
                target_dir,
                "start_state",
            )

            raw_target = make_target(
                target_name,
                device,
            )
            raw_target = torch.as_tensor(
                raw_target,
                device=device,
                dtype=dtype,
            )

            if raw_target.ndim == 4:
                raw_target = raw_target[0, 0]
            elif raw_target.ndim == 3:
                raw_target = raw_target[0]

            if raw_target.ndim != 2:
                raise RuntimeError(
                    f"Invalid target shape: "
                    f"{tuple(raw_target.shape)}"
                )

            save_state_bundle(
                raw_target,
                target_dir,
                "raw_target",
            )

            adjusted_target = adjust_target_to_state_gpu(
                start_state[0, 0],
                raw_target,
            ).to(
                device=device,
                dtype=dtype,
            )

            target = adjusted_target[
                None,
                None,
            ].detach()

            save_state_bundle(
                target,
                target_dir,
                "adjusted_target",
            )

            initial_mae_value = float(
                torch.mean(
                    torch.abs(
                        start_state - target
                    )
                ).item()
            )

            target_actions = 0
            consecutive_converged = 0

            if args.completion_mode == "paper":
                status = "paper_action_snapshot"
            else:
                status = "incomplete_action_budget"

            print()
            print("===============================================")
            print("TARGET START")
            print("===============================================")
            print("target             :", target_name)
            print("action budget      :", action_budget)
            print("cumulative actions :", cumulative_actions)
            print("initial MAE        :", initial_mae_value * 1000.0)
            print("clay reset         : NO")
            print("===============================================")

            while target_actions < action_budget:
                current = get_current_state()

                mean_value = float(
                    torch.mean(
                        torch.abs(
                            current - target
                        )
                    ).item()
                )

                if args.completion_mode == "auto":
                    if mean_value < TARGET_MEAN_ERROR:
                        consecutive_converged += 1
                    else:
                        consecutive_converged = 0

                    print(
                        "[CONVERGENCE]",
                        consecutive_converged,
                        "/",
                        args.convergence_checks,
                    )

                    if (
                        consecutive_converged
                        >= args.convergence_checks
                    ):
                        status = "target_converged"
                        break

                remaining = (
                    action_budget
                    -
                    target_actions
                )

                horizon = min(
                    args.plan_horizon,
                    remaining,
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

                print()
                print(
                    f"[MPC] target={target_name} "
                    f"target_actions={target_actions} "
                    f"cumulative={cumulative_actions} "
                    f"MAE={mean_value * 1000.0:.3f}mm"
                )

                initial_plan = greedy_plan_gpu(
                    model,
                    current,
                    target,
                    horizon=horizon,
                    candidates=args.candidates,
                    w3d=args.w3d,
                    wviz=args.wviz,
                    recent_actions=recent_actions,
                    candidate_batch_size=(
                        args.candidate_batch_size
                    ),
                )

                if (
                    args.refine_steps > 0
                    and args.refine_actions > 0
                ):
                    plan = refine_plan_gpu(
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
                        refine_steps=args.refine_steps,
                    )
                else:
                    plan = initial_plan

                execute_n = min(
                    args.mpc_execute,
                    plan.shape[0],
                    remaining,
                )

                for plan_index in range(execute_n):
                    action = plan[
                        plan_index
                    ].detach()

                    before = get_current_state()

                    with torch.no_grad():
                        predicted_after = predict_next(
                            model,
                            before,
                            action,
                        ).detach()

                    before_mae = torch.mean(
                        torch.abs(
                            before - target
                        )
                    )
                    predicted_mae = torch.mean(
                        torch.abs(
                            predicted_after - target
                        )
                    )

                    print(
                        f"[EXECUTE "
                        f"{cumulative_actions + 1:04d}] "
                        f"target={target_name} "
                        f"target_action={target_actions + 1} "
                        f"length={float(action[3]) * 1000.0:.2f}mm "
                        f"depth={float(action[4]) * 1000.0:.2f}mm"
                    )

                    sim.linear_push(
                        action,
                        action_index=cumulative_actions,
                    )

                    target_actions += 1
                    cumulative_actions += 1

                    recent_action_list.append(
                        action.detach().clone()
                    )

                    if (
                        len(recent_action_list)
                        > RECENT_ACTION_HISTORY
                    ):
                        recent_action_list = recent_action_list[
                            -RECENT_ACTION_HISTORY:
                        ]

                    actual_after = get_current_state()

                    _, l3d, lviz = total_loss(
                        actual_after,
                        target,
                        args.w3d,
                        args.wviz,
                    )

                    cd = chamfer_distance(
                        actual_after,
                        target,
                    )

                    actual_mae = torch.mean(
                        torch.abs(
                            actual_after - target
                        )
                    )
                    model_gap = torch.mean(
                        torch.abs(
                            predicted_after - actual_after
                        )
                    )

                    metric_values = torch.stack(
                        [
                            before_mae,
                            predicted_mae,
                            actual_mae,
                            model_gap,
                            l3d,
                            lviz,
                            cd,
                        ]
                    ).detach().cpu().tolist()

                    action_values = (
                        action.detach().cpu().tolist()
                    )

                    action_writer.writerow(
                        [
                            target_name,
                            target_actions,
                            cumulative_actions,
                            *action_values,
                            *metric_values,
                        ]
                    )
                    action_file.flush()

                    print(
                        f"[ACTION {cumulative_actions:04d}] "
                        f"before={metric_values[0] * 1000.0:.3f}mm "
                        f"predicted={metric_values[1] * 1000.0:.3f}mm "
                        f"actual={metric_values[2] * 1000.0:.3f}mm "
                        f"model_gap={metric_values[3] * 1000.0:.3f}mm"
                    )

                    if (
                        target_actions
                        % args.save_every
                        == 0
                    ):
                        save_state_bundle(
                            actual_after,
                            target_dir,
                            (
                                f"action_{target_actions:04d}_"
                                f"global_{cumulative_actions:04d}"
                            ),
                        )

                    if target_actions >= action_budget:
                        break

                del initial_plan
                del plan

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            final_state = get_current_state()

            _, final_l3d, final_lviz = total_loss(
                final_state,
                target,
                args.w3d,
                args.wviz,
            )

            final_cd = chamfer_distance(
                final_state,
                target,
            )

            final_mae_value = float(
                torch.mean(
                    torch.abs(
                        final_state - target
                    )
                ).item()
            )

            metric_values = torch.stack(
                [
                    final_l3d,
                    final_lviz,
                    final_cd,
                ]
            ).detach().cpu().tolist()

            improvement = (
                initial_mae_value
                -
                final_mae_value
            )

            if initial_mae_value > 0.0:
                improvement_percent = (
                    improvement
                    /
                    initial_mae_value
                    *
                    100.0
                )
            else:
                improvement_percent = 0.0

            save_state_bundle(
                final_state,
                target_dir,
                "final_state",
            )

            summary_writer.writerow(
                [
                    target_name,
                    target_actions,
                    cumulative_actions,
                    initial_mae_value * 1000.0,
                    final_mae_value * 1000.0,
                    improvement * 1000.0,
                    improvement_percent,
                    *metric_values,
                    status,
                ]
            )
            summary_file.flush()

            print()
            print("===============================================")
            print("TARGET END")
            print("===============================================")
            print("target             :", target_name)
            print("target actions     :", target_actions)
            print("cumulative actions :", cumulative_actions)
            print("final MAE          :", final_mae_value * 1000.0)
            print("status             :", status)
            print("clay reset         : NO")
            print("===============================================")

            del start_state
            del raw_target
            del adjusted_target
            del target
            del final_state

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_state_bundle(
        get_current_state(),
        state_root,
        "sequence_final",
    )

    print()
    print("===============================================")
    print("ALPHABET SEQUENCE COMPLETE")
    print("===============================================")
    print("targets            :", targets)
    print("cumulative actions :", cumulative_actions)
    print("clay resets        : 0")
    print("actions CSV        :", actions_csv_path)
    print("summary CSV        :", summary_csv_path)
    print("output             :", experiment_dir)
    print("===============================================")


if __name__ == "__main__":
    main()
