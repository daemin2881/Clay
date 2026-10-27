import argparse
import math
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Visual Sculpting transitions "
            "(S_t, [x,y,theta,length,depth], S_t+1)."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--material",
        type=str,
        default="medium",
        choices=[
            "soft",
            "medium",
            "hard",
        ],
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--rollout-length",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--settle-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-resample",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--clear",
        action="store_true",
    )

    parser.add_argument(
        "--append",
        action="store_true",
    )

    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError(
            "--n must be >= 1"
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be >= 1"
        )

    if args.rollout_length <= 0:
        raise ValueError(
            "--rollout-length must be >= 1"
        )

    if args.settle_steps < 0:
        raise ValueError(
            "--settle-steps must be >= 0"
        )

    if args.max_resample <= 0:
        raise ValueError(
            "--max-resample must be >= 1"
        )

    if args.clear and args.append:
        raise ValueError(
            "--clear and --append cannot be used together"
        )

    return args


ARGS = parse_args()


# ============================================================
# ENVIRONMENT
#
# vs_simulator가 import될 때 읽기 때문에
# import보다 먼저 설정해야 한다.
# ============================================================

os.environ["VS_MATERIAL"] = ARGS.material

os.environ["VS_SHOW_VIEWER"] = (
    "1"
    if ARGS.viewer
    else "0"
)


# ============================================================
# IMPORTS
# ============================================================

import torch

import vs_simulator as sim

from vs_config import (
    DATASET_DIR,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    LENGTH_MIN,
    LENGTH_MAX,
    DEPTH_MIN,
    DEPTH_MAX,
)


# ============================================================
# CONSTANTS
# ============================================================

ACTION_NAMES = [
    "x",
    "y",
    "theta",
    "length",
    "depth",
]

ACTION_TYPE = (
    "visual_sculpting_linear_push"
)

SAMPLER_NAME = (
    "uniform_feasible_5d"
)


# ============================================================
# UNIFORM SCALAR SAMPLING
# ============================================================

def uniform_scalar(
    low,
    high,
    device,
    dtype,
):
    if high < low:
        raise ValueError(
            f"Invalid range: [{low}, {high}]"
        )

    return torch.empty(
        (),
        device=device,
        dtype=dtype,
    ).uniform_(
        low,
        high,
    )


# ============================================================
# ACTION ENDPOINT
# ============================================================

def endpoint(
    action,
):
    x, y, theta, length, _ = (
        action.unbind()
    )

    return torch.stack(
        [
            x
            +
            length
            *
            torch.cos(
                theta
            ),

            y
            +
            length
            *
            torch.sin(
                theta
            ),
        ]
    )


# ============================================================
# ACTION VALIDATION
# ============================================================

def action_is_valid(
    action,
):
    if action.shape != (5,):
        return False

    if not bool(
        torch.isfinite(
            action
        ).all()
    ):
        return False

    (
        x,
        y,
        theta,
        length,
        depth,
    ) = action.unbind()

    end_xy = endpoint(
        action
    )

    return bool(
        (x >= X_MIN)
        &
        (x <= X_MAX)
        &
        (y >= Y_MIN)
        &
        (y <= Y_MAX)
        &
        (theta >= -math.pi)
        &
        (theta <= math.pi)
        &
        (length >= LENGTH_MIN)
        &
        (length <= LENGTH_MAX)
        &
        (depth >= DEPTH_MIN)
        &
        (depth <= DEPTH_MAX)
        &
        (end_xy[0] >= X_MIN)
        &
        (end_xy[0] <= X_MAX)
        &
        (end_xy[1] >= Y_MIN)
        &
        (end_xy[1] <= Y_MAX)
    )


# ============================================================
# VISUAL SCULPTING UNIFORM ACTION SAMPLER
#
# a = [x, y, theta, length, depth]
#
# 각 파라미터를 설정 범위에서 균등 표본 추출한다.
# push 시작점과 끝점이 작업영역 안에 있는 action만
# rejection sampling으로 선택한다.
#
# 기존 코드에 있던 다음 편향은 제거한다.
#
# - shallow / medium / deep 비율
# - short / medium / long 비율
# - edge inward 방향 강제
# - 최근 시작점 회피
# ============================================================

class UniformFeasibleActionSampler:

    def __init__(
        self,
        device,
        dtype,
        max_resample,
    ):
        self.device = device
        self.dtype = dtype
        self.max_resample = int(
            max_resample
        )

    def random_unit(
        self,
    ):
        return torch.rand(
            (),
            device=self.device,
            dtype=self.dtype,
        )

    def sample(
        self,
    ):
        # ----------------------------------------------------
        # 1. 방향 균등 표본 추출
        # ----------------------------------------------------

        theta = (
            -math.pi
            +
            self.random_unit()
            *
            (
                2.0
                *
                math.pi
            )
        )

        # ----------------------------------------------------
        # 2. 길이 균등 표본 추출
        #
        # 5mm ~ 324mm가 동일한 확률밀도를 갖는다.
        # ----------------------------------------------------

        length = (
            LENGTH_MIN
            +
            self.random_unit()
            *
            (
                LENGTH_MAX
                -
                LENGTH_MIN
            )
        )

        # ----------------------------------------------------
        # 3. 깊이 균등 표본 추출
        #
        # 1mm ~ 50mm가 동일한 확률밀도를 갖는다.
        # ----------------------------------------------------

        depth = (
            DEPTH_MIN
            +
            self.random_unit()
            *
            (
                DEPTH_MAX
                -
                DEPTH_MIN
            )
        )

        # ----------------------------------------------------
        # 4. 이동 벡터
        # ----------------------------------------------------

        dx = (
            length
            *
            torch.cos(
                theta
            )
        )

        dy = (
            length
            *
            torch.sin(
                theta
            )
        )

        # ----------------------------------------------------
        # 5. 주어진 방향과 길이에서 가능한 시작점 범위
        #
        # 시작점과 끝점이 모두 workspace 내부가 되도록 한다.
        # 긴 action도 잘리지 않고 그대로 실행된다.
        # ----------------------------------------------------

        x_low = (
            X_MIN
            +
            torch.clamp(
                -dx,
                min=0.0,
            )
        )

        x_high = (
            X_MAX
            -
            torch.clamp(
                dx,
                min=0.0,
            )
        )

        y_low = (
            Y_MIN
            +
            torch.clamp(
                -dy,
                min=0.0,
            )
        )

        y_high = (
            Y_MAX
            -
            torch.clamp(
                dy,
                min=0.0,
            )
        )

        if bool(
            (x_high < x_low)
            |
            (y_high < y_low)
        ):
            raise RuntimeError(
                "Configured ACTION_LENGTH_MAX is larger "
                "than the feasible workspace."
            )

        # ----------------------------------------------------
        # 6. 가능한 영역에서 시작점 표본 추출
        # ----------------------------------------------------

        x = (
            x_low
            +
            self.random_unit()
            *
            (
                x_high
                -
                x_low
            )
        )

        y = (
            y_low
            +
            self.random_unit()
            *
            (
                y_high
                -
                y_low
            )
        )

        # ----------------------------------------------------
        # 7. Visual Sculpting 5D action
        # ----------------------------------------------------

        action = torch.stack(
            [
                x,
                y,
                theta,
                length,
                depth,
            ]
        )

        if not action_is_valid(
            action
        ):
            raise RuntimeError(
                "Generated action failed workspace validation."
            )

        return (
            action,
            1,
        )

# ============================================================
# DATASET WRITER
# ============================================================

class DatasetWriter:

    def __init__(
        self,
        output_dir,
        args,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.args = args

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []

        self.rollout_id_buffer = []
        self.rollout_step_buffer = []

        existing = sorted(
            self.output_dir.glob(
                "chunk_*.pt"
            )
        )

        if args.clear:
            for path in existing:
                path.unlink()

            print(
                f"[dataset] removed "
                f"{len(existing)} old chunks"
            )

            existing = []

        elif (
            existing
            and
            not args.append
        ):
            raise RuntimeError(
                f"{self.output_dir} already contains "
                f"{len(existing)} chunks. "
                "Use a new --output-dir, move the old "
                "directory, or explicitly use --append."
            )

        self.next_chunk_id = (
            self._next_chunk_id(
                existing
            )
        )

        self.rollout_id_offset = (
            self._next_rollout_id(
                existing
            )
        )

    @staticmethod
    def _next_chunk_id(
        paths,
    ):
        ids = []

        for path in paths:
            try:
                ids.append(
                    int(
                        path.stem.split(
                            "_"
                        )[-1]
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return (
            max(ids) + 1
            if ids
            else 0
        )

    @staticmethod
    def _next_rollout_id(
        paths,
    ):
        maximum = -1

        for path in paths:
            chunk = torch.load(
                path,
                map_location="cpu",
            )

            rollout_ids = chunk.get(
                "rollout_id"
            )

            if (
                rollout_ids is not None
                and
                rollout_ids.numel() > 0
            ):
                maximum = max(
                    maximum,
                    int(
                        rollout_ids.max()
                    ),
                )

        return maximum + 1

    def add(
        self,
        state,
        action,
        next_state,
        rollout_id,
        rollout_step,
    ):
        self.state_buffer.append(
            state.detach().clone()
        )

        self.action_buffer.append(
            action.detach().clone()
        )

        self.next_state_buffer.append(
            next_state.detach().clone()
        )

        self.rollout_id_buffer.append(
            int(
                rollout_id
            )
        )

        self.rollout_step_buffer.append(
            int(
                rollout_step
            )
        )

        if (
            len(
                self.state_buffer
            )
            >=
            self.args.chunk_size
        ):
            self.flush()

    def flush(
        self,
    ):
        if not self.state_buffer:
            return

        states = torch.stack(
            self.state_buffer,
            dim=0,
        ).cpu()

        actions = torch.stack(
            self.action_buffer,
            dim=0,
        ).cpu()

        next_states = torch.stack(
            self.next_state_buffer,
            dim=0,
        ).cpu()

        rollout_ids = torch.tensor(
            self.rollout_id_buffer,
            dtype=torch.long,
        )

        rollout_steps = torch.tensor(
            self.rollout_step_buffer,
            dtype=torch.long,
        )

        if states.ndim != 3:
            raise RuntimeError(
                "State batch must be [N,H,W], "
                f"got {states.shape}"
            )

        if next_states.shape != states.shape:
            raise RuntimeError(
                f"next_state shape {next_states.shape} "
                f"!= state shape {states.shape}"
            )

        if actions.shape != (
            states.shape[0],
            5,
        ):
            raise RuntimeError(
                "Action batch must be [N,5], "
                f"got {actions.shape}"
            )

        if not bool(
            torch.isfinite(
                states
            ).all()
        ):
            raise RuntimeError(
                "State batch contains NaN or Inf"
            )

        if not bool(
            torch.isfinite(
                actions
            ).all()
        ):
            raise RuntimeError(
                "Action batch contains NaN or Inf"
            )

        if not bool(
            torch.isfinite(
                next_states
            ).all()
        ):
            raise RuntimeError(
                "next_state batch contains NaN or Inf"
            )

        path = (
            self.output_dir
            /
            f"chunk_{self.next_chunk_id:05d}.pt"
        )

        torch.save(
            {
                # -------------------------------------------
                # Main transition data
                # -------------------------------------------

                "state":
                states,

                "action":
                actions,

                "next_state":
                next_states,

                # -------------------------------------------
                # Compatibility aliases
                # -------------------------------------------

                "states":
                states,

                "actions":
                actions,

                "next_states":
                next_states,

                # -------------------------------------------
                # Rollout metadata
                # -------------------------------------------

                "rollout_id":
                rollout_ids,

                "rollout_step":
                rollout_steps,

                "rollout_length":
                int(
                    self.args.rollout_length
                ),

                # -------------------------------------------
                # Action metadata
                # -------------------------------------------

                "action_type":
                ACTION_TYPE,

                "action_dim":
                5,

                "action_definition":
                ACTION_NAMES,

                "action_sampler":
                SAMPLER_NAME,

                "action_ranges":
                {
                    "x":
                    (
                        float(
                            X_MIN
                        ),
                        float(
                            X_MAX
                        ),
                    ),

                    "y":
                    (
                        float(
                            Y_MIN
                        ),
                        float(
                            Y_MAX
                        ),
                    ),

                    "theta":
                    (
                        -math.pi,
                        math.pi,
                    ),

                    "length":
                    (
                        float(
                            LENGTH_MIN
                        ),
                        float(
                            LENGTH_MAX
                        ),
                    ),

                    "depth":
                    (
                        float(
                            DEPTH_MIN
                        ),
                        float(
                            DEPTH_MAX
                        ),
                    ),
                },

                "material":
                self.args.material,

                "seed":
                int(
                    self.args.seed
                ),

                "settle_steps":
                int(
                    self.args.settle_steps
                ),
            },
            path,
        )

        print(
            f"[SAVE] {path} | "
            f"state={tuple(states.shape)} | "
            f"action={tuple(actions.shape)} | "
            f"next_state={tuple(next_states.shape)}"
        )

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.next_state_buffer.clear()

        self.rollout_id_buffer.clear()
        self.rollout_step_buffer.clear()

        self.next_chunk_id += 1


# ============================================================
# HEIGHT-MAP SNAPSHOT
# ============================================================

def clone_height_state():
    state = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )

    if state.ndim != 2:
        raise RuntimeError(
            "Height map must be [H,W], "
            f"got {state.shape}"
        )

    if not bool(
        torch.isfinite(
            state
        ).all()
    ):
        raise RuntimeError(
            "Height map contains NaN or Inf"
        )

    return state


# ============================================================
# HEADER
# ============================================================

def print_header(
    output_dir,
):
    num_rollouts = math.ceil(
        ARGS.n
        /
        ARGS.rollout_length
    )

    print()
    print(
        "==============================================="
    )
    print(
        "VISUAL SCULPTING TRANSITION DATA COLLECTION"
    )
    print(
        "==============================================="
    )
    print(
        f"samples          : {ARGS.n}"
    )
    print(
        f"rollouts         : {num_rollouts}"
    )
    print(
        f"rollout length   : {ARGS.rollout_length}"
    )
    print(
        f"material         : {ARGS.material}"
    )
    print(
        f"viewer           : {ARGS.viewer}"
    )
    print(
        f"settle steps     : {ARGS.settle_steps}"
    )
    print(
        f"sampler          : {SAMPLER_NAME}"
    )
    print(
        "action           : "
        "[x, y, theta, length, depth]"
    )
    print(
        f"x range          : "
        f"{X_MIN:.6f} ~ {X_MAX:.6f} m"
    )
    print(
        f"y range          : "
        f"{Y_MIN:.6f} ~ {Y_MAX:.6f} m"
    )
    print(
        f"theta range      : "
        f"{-math.pi:.6f} ~ {math.pi:.6f} rad"
    )
    print(
        f"length range     : "
        f"{LENGTH_MIN * 1000.0:.1f} ~ "
        f"{LENGTH_MAX * 1000.0:.1f} mm"
    )
    print(
        f"depth range      : "
        f"{DEPTH_MIN * 1000.0:.1f} ~ "
        f"{DEPTH_MAX * 1000.0:.1f} mm"
    )
    print(
        f"output           : {output_dir}"
    )
    print(
        "==============================================="
    )
    print()


# ============================================================
# MAIN
# ============================================================

@torch.no_grad()
def main():
    torch.manual_seed(
        ARGS.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            ARGS.seed
        )

    output_dir = (
        Path(
            ARGS.output_dir
        )
        if ARGS.output_dir is not None
        else
        Path(
            DATASET_DIR
        )
        /
        ARGS.material
    )

    writer = DatasetWriter(
        output_dir,
        ARGS,
    )

    print_header(
        output_dir
    )

    # ========================================================
    # SIMULATOR INITIALIZATION
    # ========================================================

    simulator = sim.initialize()

    if (
        sim.DEVICE is None
        or
        sim.FLOAT_DTYPE is None
    ):
        raise RuntimeError(
            "Simulator device/dtype was not initialized"
        )

    # ========================================================
    # INITIAL STABILIZATION
    #
    # 초기 중력 침하가 학습 delta에 포함되지 않게 한다.
    # ========================================================

    if ARGS.settle_steps > 0:
        print(
            f"Initial stabilization: "
            f"{ARGS.settle_steps} steps"
        )

        simulator.hold(
            ARGS.settle_steps
        )

    # ========================================================
    # CLEAN SCENE SNAPSHOT
    # ========================================================

    initial_scene_state = (
        simulator.scene.get_state()
    )

    initial_last_q = (
        simulator.last_q
        .detach()
        .clone()
    )

    initial_height = (
        clone_height_state()
    )

    # ========================================================
    # SAMPLER
    # ========================================================

    sampler = (
        UniformFeasibleActionSampler(
            device=sim.DEVICE,
            dtype=sim.FLOAT_DTYPE,
            max_resample=ARGS.max_resample,
        )
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    x_values = []
    y_values = []
    theta_values = []
    length_values = []
    depth_values = []

    delta_mae_values = []
    delta_max_values = []

    total_attempts = 0
    reset_count = 0

    # ========================================================
    # COLLECTION
    # ========================================================

    try:
        for sample_idx in range(
            ARGS.n
        ):
            local_rollout_id = (
                sample_idx
                //
                ARGS.rollout_length
            )

            rollout_id = (
                writer.rollout_id_offset
                +
                local_rollout_id
            )

            rollout_step = (
                sample_idx
                %
                ARGS.rollout_length
            )

            # ================================================
            # RESET AT ROLLOUT BOUNDARY
            # ================================================

            if (
                sample_idx > 0
                and
                rollout_step == 0
            ):
                simulator.scene.reset(
                    state=initial_scene_state
                )

                simulator.last_q = (
                    initial_last_q
                    .detach()
                    .clone()
                )

                simulator.initialized = True

                reset_height = (
                    clone_height_state()
                )

                reset_error_mm = float(
                    torch.mean(
                        torch.abs(
                            reset_height
                            -
                            initial_height
                        )
                    )
                    *
                    1000.0
                )

                if reset_error_mm > 0.25:
                    raise RuntimeError(
                        "Scene reset did not recover "
                        "the clean state: "
                        f"MAE={reset_error_mm:.6f} mm"
                    )

                reset_count += 1

                print()
                print(
                    f"[ROLLOUT RESET] "
                    f"id={rollout_id:04d} "
                    f"clean_state_mae="
                    f"{reset_error_mm:.6f}mm"
                )

            # ================================================
            # STATE BEFORE ACTION
            # ================================================

            state_before = (
                clone_height_state()
            )

            # ================================================
            # UNIFORM 5D ACTION
            # ================================================

            (
                action,
                attempts,
            ) = sampler.sample()

            total_attempts += attempts

            # ================================================
            # EXECUTE VISUAL SCULPTING PUSH
            # ================================================

            sim.linear_push(
                action,
                action_index=sample_idx,
            )

            # ================================================
            # STATE AFTER ACTION
            # ================================================

            state_after = (
                clone_height_state()
            )

            # ================================================
            # DEFORMATION STATISTICS
            # ================================================

            delta_abs = torch.abs(
                state_after
                -
                state_before
            )

            delta_mae_mm = float(
                delta_abs.mean()
                *
                1000.0
            )

            delta_max_mm = float(
                delta_abs.max()
                *
                1000.0
            )

            # ================================================
            # STORE TRANSITION
            # ================================================

            writer.add(
                state_before,
                action,
                state_after,
                rollout_id,
                rollout_step,
            )

            (
                x,
                y,
                theta,
                length,
                depth,
            ) = (
                action
                .detach()
                .cpu()
                .tolist()
            )

            end_x = (
                x
                +
                length
                *
                math.cos(
                    theta
                )
            )

            end_y = (
                y
                +
                length
                *
                math.sin(
                    theta
                )
            )

            x_values.append(
                x
            )

            y_values.append(
                y
            )

            theta_values.append(
                theta
            )

            length_values.append(
                length
            )

            depth_values.append(
                depth
            )

            delta_mae_values.append(
                delta_mae_mm
            )

            delta_max_values.append(
                delta_max_mm
            )

            print(
                f"[{sample_idx + 1:04d}/"
                f"{ARGS.n:04d}] "

                f"rollout={rollout_id:04d} "

                f"step={rollout_step + 1:02d}/"
                f"{ARGS.rollout_length:02d} | "

                f"start=({x:.4f},{y:.4f}) "

                f"end=({end_x:.4f},{end_y:.4f}) | "

                f"theta={theta:+.3f} "

                f"length={length * 1000.0:.2f}mm "

                f"depth={depth * 1000.0:.2f}mm | "

                f"delta_MAE={delta_mae_mm:.4f}mm "

                f"delta_MAX={delta_max_mm:.4f}mm | "

                f"sample_attempts={attempts}"
            )

    finally:
        # 실행 중 중단되더라도 이미 완성된 transition은 저장한다.
        writer.flush()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "==============================================="
    )
    print(
        "COLLECTION COMPLETE"
    )
    print(
        "==============================================="
    )
    print(
        f"samples             : {ARGS.n}"
    )
    print(
        f"scene resets        : {reset_count}"
    )
    print(
        f"sampler             : {SAMPLER_NAME}"
    )
    print(
        f"mean sample attempts: "
        f"{total_attempts / ARGS.n:.3f}"
    )
    print(
        f"x min/max           : "
        f"{min(x_values):.6f} / "
        f"{max(x_values):.6f} m"
    )
    print(
        f"y min/max           : "
        f"{min(y_values):.6f} / "
        f"{max(y_values):.6f} m"
    )
    print(
        f"theta min/max       : "
        f"{min(theta_values):.6f} / "
        f"{max(theta_values):.6f} rad"
    )
    print(
        f"length min/max      : "
        f"{min(length_values) * 1000.0:.3f} / "
        f"{max(length_values) * 1000.0:.3f} mm"
    )
    print(
        f"depth min/max       : "
        f"{min(depth_values) * 1000.0:.3f} / "
        f"{max(depth_values) * 1000.0:.3f} mm"
    )
    print(
        f"delta MAE mean      : "
        f"{sum(delta_mae_values) / len(delta_mae_values):.6f} mm"
    )
    print(
        f"delta MAX mean      : "
        f"{sum(delta_max_values) / len(delta_max_values):.6f} mm"
    )
    print(
        f"saved to            : {output_dir}"
    )
    print(
        "==============================================="
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()