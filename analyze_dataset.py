import argparse
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


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
    "--dataset-dir",
    type=str,
    default=None,
)


parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
)


parser.add_argument(
    "--no-action-mae-mm",
    type=float,
    default=0.036730,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

if args.dataset_dir is None:

    DATASET_DIR = Path(
        "dataset"
    ) / args.material

else:

    DATASET_DIR = Path(
        args.dataset_dir
    )


if args.output_dir is None:

    OUTPUT_DIR = (
        Path(
            "results"
        )
        /
        "dataset_analysis"
        /
        args.material
    )

else:

    OUTPUT_DIR = Path(
        args.output_dir
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


NO_ACTION_MAE_MM = (
    float(
        args.no_action_mae_mm
    )
)


# ============================================================
# OPTIONAL CONFIG IMPORT
#
# We use current collection ranges whenever available.
# ============================================================

try:

    from vs_config import (
        COLLECTION_DEPTH_SHALLOW_MIN,
        COLLECTION_DEPTH_SHALLOW_MAX,
        COLLECTION_DEPTH_MEDIUM_MIN,
        COLLECTION_DEPTH_MEDIUM_MAX,
        COLLECTION_DEPTH_DEEP_MIN,
        COLLECTION_DEPTH_DEEP_MAX,

        COLLECTION_LENGTH_SHORT_MIN,
        COLLECTION_LENGTH_SHORT_MAX,
        COLLECTION_LENGTH_MEDIUM_MIN,
        COLLECTION_LENGTH_MEDIUM_MAX,
        COLLECTION_LENGTH_LONG_MIN,
        COLLECTION_LENGTH_LONG_MAX,
    )


    HAVE_COLLECTION_CONFIG = True


except Exception:

    HAVE_COLLECTION_CONFIG = False


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    chunk_files = sorted(
        DATASET_DIR.glob(
            "chunk_*.pt"
        )
    )


    if len(
        chunk_files
    ) == 0:

        raise FileNotFoundError(
            f"No chunk files found in {DATASET_DIR}"
        )


    states = []

    actions = []

    next_states = []


    rollout_ids = []

    rollout_steps = []


    has_rollout_metadata = True


    action_types = set()

    action_dims = set()


    print()
    print(
        "==============================================="
    )

    print(
        "LOADING DATASET"
    )

    print(
        "==============================================="
    )


    print(
        "dataset dir =",
        DATASET_DIR
    )


    print(
        "chunks      =",
        len(
            chunk_files
        )
    )


    print()


    for chunk_file in chunk_files:

        data = torch.load(
            chunk_file,
            map_location="cpu",
        )


        # ====================================================
        # STATE
        # ====================================================

        if "state" in data:

            state = data[
                "state"
            ]


        elif "states" in data:

            state = data[
                "states"
            ]


        else:

            raise KeyError(
                f"No state/states key in {chunk_file}"
            )


        # ====================================================
        # ACTION
        # ====================================================

        if "action" in data:

            action = data[
                "action"
            ]


        elif "actions" in data:

            action = data[
                "actions"
            ]


        else:

            raise KeyError(
                f"No action/actions key in {chunk_file}"
            )


        # ====================================================
        # NEXT STATE
        # ====================================================

        if "next_state" in data:

            next_state = data[
                "next_state"
            ]


        elif "next_states" in data:

            next_state = data[
                "next_states"
            ]


        else:

            raise KeyError(
                f"No next_state/next_states key in "
                f"{chunk_file}"
            )


        # ====================================================
        # SHAPE CHECK
        # ====================================================

        if state.ndim != 3:

            raise ValueError(
                f"{chunk_file}: state must have shape "
                f"[N,H,W], got {tuple(state.shape)}"
            )


        if next_state.shape != state.shape:

            raise ValueError(
                f"{chunk_file}: next_state shape "
                f"{tuple(next_state.shape)} != "
                f"state shape {tuple(state.shape)}"
            )


        if action.ndim != 2:

            raise ValueError(
                f"{chunk_file}: action must have shape "
                f"[N,5], got {tuple(action.shape)}"
            )


        if action.shape[
            1
        ] != 5:

            raise ValueError(
                f"{chunk_file}: expected Visual Sculpting "
                f"5D action [x,y,theta,length,depth], "
                f"got shape {tuple(action.shape)}"
            )


        if action.shape[
            0
        ] != state.shape[
            0
        ]:

            raise ValueError(
                f"{chunk_file}: number of actions "
                f"does not match states."
            )


        states.append(
            state.float()
        )


        actions.append(
            action.float()
        )


        next_states.append(
            next_state.float()
        )


        # ====================================================
        # METADATA
        # ====================================================

        if "action_type" in data:

            action_types.add(
                str(
                    data[
                        "action_type"
                    ]
                )
            )


        if "action_dim" in data:

            action_dims.add(
                int(
                    data[
                        "action_dim"
                    ]
                )
            )


        # ====================================================
        # ROLLOUT METADATA
        # ====================================================

        if (
            "rollout_id" in data
            and
            "rollout_step" in data
        ):

            rid = data[
                "rollout_id"
            ].long()


            rstep = data[
                "rollout_step"
            ].long()


            if rid.shape[
                0
            ] != state.shape[
                0
            ]:

                raise ValueError(
                    f"{chunk_file}: rollout_id length "
                    f"does not match samples."
                )


            if rstep.shape[
                0
            ] != state.shape[
                0
            ]:

                raise ValueError(
                    f"{chunk_file}: rollout_step length "
                    f"does not match samples."
                )


            rollout_ids.append(
                rid
            )


            rollout_steps.append(
                rstep
            )


        else:

            has_rollout_metadata = False


        print(
            f"[loaded] {chunk_file.name} | "
            f"N={state.shape[0]} | "
            f"state={tuple(state.shape)} | "
            f"action={tuple(action.shape)}"
        )


    # ========================================================
    # CONCATENATE
    # ========================================================

    states = torch.cat(
        states,
        dim=0,
    )


    actions = torch.cat(
        actions,
        dim=0,
    )


    next_states = torch.cat(
        next_states,
        dim=0,
    )


    if (
        has_rollout_metadata
        and
        len(
            rollout_ids
        )
        ==
        len(
            chunk_files
        )
    ):

        rollout_ids = torch.cat(
            rollout_ids,
            dim=0,
        )


        rollout_steps = torch.cat(
            rollout_steps,
            dim=0,
        )


    else:

        rollout_ids = None

        rollout_steps = None


    print()
    print(
        "==============================================="
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "==============================================="
    )


    print(
        "samples          =",
        states.shape[
            0
        ]
    )


    print(
        "state shape      =",
        tuple(
            states.shape
        )
    )


    print(
        "action shape     =",
        tuple(
            actions.shape
        )
    )


    print(
        "expected action  = "
        "[x, y, theta, length, depth]"
    )


    print(
        "action metadata  =",
        sorted(
            action_types
        )
    )


    print(
        "action dims      =",
        sorted(
            action_dims
        )
    )


    print(
        "rollout metadata =",
        (
            "YES"
            if rollout_steps is not None
            else
            "NO"
        )
    )


    print(
        "==============================================="
    )


    return (
        states,
        actions,
        next_states,
        rollout_ids,
        rollout_steps,
    )


# ============================================================
# PER-SAMPLE METRICS
# ============================================================

def compute_metrics(
    states,
    next_states,
):

    delta = (
        next_states
        -
        states
    )


    abs_delta = torch.abs(
        delta
    )


    mae = torch.mean(
        abs_delta,
        dim=(
            1,
            2,
        ),
    )


    mse = torch.mean(
        delta
        *
        delta,
        dim=(
            1,
            2,
        ),
    )


    max_error = torch.amax(
        abs_delta,
        dim=(
            1,
            2,
        ),
    )


    changed_01 = torch.mean(
        (
            abs_delta
            >
            0.0001
        ).float(),
        dim=(
            1,
            2,
        ),
    )


    changed_05 = torch.mean(
        (
            abs_delta
            >
            0.0005
        ).float(),
        dim=(
            1,
            2,
        ),
    )


    changed_10 = torch.mean(
        (
            abs_delta
            >
            0.0010
        ).float(),
        dim=(
            1,
            2,
        ),
    )


    return (
        delta,
        mae,
        mse,
        max_error,
        changed_01,
        changed_05,
        changed_10,
    )


# ============================================================
# SAFE CORRELATION
# ============================================================

def correlation(
    x,
    y,
):

    x = x.float()

    y = y.float()


    if x.numel() < 2:

        return float(
            "nan"
        )


    x_centered = (
        x
        -
        x.mean()
    )


    y_centered = (
        y
        -
        y.mean()
    )


    denominator = torch.sqrt(
        torch.sum(
            x_centered
            *
            x_centered
        )
        *
        torch.sum(
            y_centered
            *
            y_centered
        )
    )


    if float(
        denominator
    ) <= 1e-12:

        return float(
            "nan"
        )


    value = (
        torch.sum(
            x_centered
            *
            y_centered
        )
        /
        denominator
    )


    return float(
        value
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

def print_group_summary(
    title,
    groups,
    mae,
    max_error,
    changed_01,
    changed_05,
    changed_10,
    strong,
):

    print()
    print(
        "==============================================="
    )

    print(
        title
    )

    print(
        "==============================================="
    )


    for name, mask in groups.items():

        mask = mask.bool()


        n = int(
            mask.sum().item()
        )


        if n == 0:

            print(
                f"{name:20s} | n=0"
            )

            continue


        print(
            f"{name:20s} | "
            f"n={n:4d} | "
            f"MAE={mae[mask].mean().item() * 1000.0:.4f} mm | "
            f"median={mae[mask].median().item() * 1000.0:.4f} mm | "
            f"MAX={max_error[mask].mean().item() * 1000.0:.3f} mm | "
            f">0.1mm={changed_01[mask].mean().item() * 100.0:.2f}% | "
            f">0.5mm={changed_05[mask].mean().item() * 100.0:.2f}% | "
            f">1mm={changed_10[mask].mean().item() * 100.0:.2f}% | "
            f">=2x drift={strong[mask].float().mean().item() * 100.0:.1f}%"
        )


    print(
        "==============================================="
    )


# ============================================================
# SAMPLE DESCRIPTION
# ============================================================

def describe_sample(
    idx,
    actions,
    mae,
    max_error,
    changed_05,
    rollout_ids,
    rollout_steps,
):

    action = actions[
        idx
    ]


    x = float(
        action[
            0
        ]
    )


    y = float(
        action[
            1
        ]
    )


    theta = float(
        action[
            2
        ]
    )


    length = float(
        action[
            3
        ]
    )


    depth = float(
        action[
            4
        ]
    )


    if (
        rollout_ids is not None
        and
        rollout_steps is not None
    ):

        rollout_text = (
            f"rollout="
            f"{int(rollout_ids[idx].item()):03d} "
            f"step="
            f"{int(rollout_steps[idx].item()) + 1:02d}"
        )


    else:

        rollout_text = (
            "rollout=? step=?"
        )


    print(
        f"#{idx + 1:04d} | "
        f"{rollout_text} | "
        f"MAE={mae[idx].item() * 1000.0:.4f}mm | "
        f"MAX={max_error[idx].item() * 1000.0:.3f}mm | "
        f">0.5mm={changed_05[idx].item() * 100.0:.2f}% | "
        f"x={x:.3f} "
        f"y={y:.3f} | "
        f"theta={theta:+.2f}rad "
        f"({math.degrees(theta):+.1f}deg) | "
        f"length={length * 1000.0:.2f}mm | "
        f"depth={depth * 1000.0:.2f}mm"
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze(
    states,
    actions,
    next_states,
    rollout_ids,
    rollout_steps,
):

    (
        delta,
        mae,
        mse,
        max_error,
        changed_01,
        changed_05,
        changed_10,

    ) = compute_metrics(
        states,
        next_states,
    )


    n = states.shape[
        0
    ]


    # ========================================================
    # ACTION PARAMETERS
    # ========================================================

    x = actions[
        :,
        0
    ]


    y = actions[
        :,
        1
    ]


    theta = actions[
        :,
        2
    ]


    length = actions[
        :,
        3
    ]


    depth = actions[
        :,
        4
    ]


    length_mm = (
        length
        *
        1000.0
    )


    depth_mm = (
        depth
        *
        1000.0
    )


    mae_mm = (
        mae
        *
        1000.0
    )


    max_mm = (
        max_error
        *
        1000.0
    )


    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "GLOBAL DEFORMATION SUMMARY"
    )

    print(
        "==============================================="
    )


    print(
        f"samples          = {n}"
    )


    print()


    print(
        f"mean MAE         = "
        f"{mae.mean().item() * 1000.0:.6f} mm"
    )


    print(
        f"median MAE       = "
        f"{mae.median().item() * 1000.0:.6f} mm"
    )


    print(
        f"min MAE          = "
        f"{mae.min().item() * 1000.0:.6f} mm"
    )


    print(
        f"max MAE          = "
        f"{mae.max().item() * 1000.0:.6f} mm"
    )


    print()


    print(
        f"mean MSE         = "
        f"{mse.mean().item():.9e}"
    )


    print()


    print(
        f"mean MAX         = "
        f"{max_error.mean().item() * 1000.0:.6f} mm"
    )


    print(
        f"max MAX          = "
        f"{max_error.max().item() * 1000.0:.6f} mm"
    )


    print()


    print(
        f"mean >0.1mm      = "
        f"{changed_01.mean().item() * 100.0:.2f}%"
    )


    print(
        f"mean >0.5mm      = "
        f"{changed_05.mean().item() * 100.0:.2f}%"
    )


    print(
        f"mean >1.0mm      = "
        f"{changed_10.mean().item() * 100.0:.2f}%"
    )


    print(
        "==============================================="
    )


    # ========================================================
    # NO-ACTION COMPARISON
    # ========================================================

    no_action_m = (
        NO_ACTION_MAE_MM
        /
        1000.0
    )


    below_drift = (
        mae
        <=
        no_action_m
    )


    near_drift = (
        mae
        <=
        no_action_m
        *
        1.5
    )


    strong = (
        mae
        >=
        no_action_m
        *
        2.0
    )


    very_strong = (
        mae
        >=
        no_action_m
        *
        3.0
    )


    print()
    print(
        "==============================================="
    )

    print(
        "DEFORMATION VS NO-ACTION DRIFT"
    )

    print(
        "==============================================="
    )


    print(
        f"no-action reference MAE = "
        f"{NO_ACTION_MAE_MM:.6f} mm"
    )


    print()


    print(
        f"<= 1.0x drift : "
        f"{below_drift.sum().item()} / {n} "
        f"({below_drift.float().mean().item() * 100.0:.2f}%)"
    )


    print(
        f"<= 1.5x drift : "
        f"{near_drift.sum().item()} / {n} "
        f"({near_drift.float().mean().item() * 100.0:.2f}%)"
    )


    print(
        f">= 2.0x drift : "
        f"{strong.sum().item()} / {n} "
        f"({strong.float().mean().item() * 100.0:.2f}%)"
    )


    print(
        f">= 3.0x drift : "
        f"{very_strong.sum().item()} / {n} "
        f"({very_strong.float().mean().item() * 100.0:.2f}%)"
    )


    print(
        "==============================================="
    )


    # ========================================================
    # ROLLOUT ANALYSIS
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "DEFORMATION BY ROLLOUT STEP"
    )

    print(
        "==============================================="
    )


    if rollout_steps is not None:

        step_tensor = (
            rollout_steps
        )


        rollout_length = int(
            step_tensor.max().item()
            +
            1
        )


        print(
            "source         = saved rollout metadata"
        )


    else:

        # Current collection default is four actions.
        rollout_length = 4


        step_tensor = (
            torch.arange(
                n
            )
            %
            rollout_length
        )


        print(
            "WARNING: rollout metadata missing."
        )


        print(
            "fallback       = sample_index % 4"
        )


    print(
        "rollout length =",
        rollout_length
    )


    print(
        "-----------------------------------------------"
    )


    rollout_step_mae = []

    rollout_step_median = []

    rollout_step_changed05 = []

    rollout_step_strong = []


    for step in range(
        rollout_length
    ):

        mask = (
            step_tensor
            ==
            step
        )


        if not mask.any():

            rollout_step_mae.append(
                float(
                    "nan"
                )
            )

            rollout_step_median.append(
                float(
                    "nan"
                )
            )

            rollout_step_changed05.append(
                float(
                    "nan"
                )
            )

            rollout_step_strong.append(
                float(
                    "nan"
                )
            )

            continue


        local_mae = mae[
            mask
        ]


        local_max = max_error[
            mask
        ]


        local_changed_01 = changed_01[
            mask
        ]


        local_changed_05 = changed_05[
            mask
        ]


        local_changed_10 = changed_10[
            mask
        ]


        local_below = below_drift[
            mask
        ]


        local_strong = strong[
            mask
        ]


        print(
            f"step {step + 1:02d} | "
            f"n={mask.sum().item():4d} | "
            f"MAE={local_mae.mean().item() * 1000.0:.4f}mm | "
            f"median={local_mae.median().item() * 1000.0:.4f}mm | "
            f"MAX={local_max.mean().item() * 1000.0:.3f}mm | "
            f">0.1={local_changed_01.mean().item() * 100.0:.2f}% | "
            f">0.5={local_changed_05.mean().item() * 100.0:.2f}% | "
            f">1.0={local_changed_10.mean().item() * 100.0:.2f}% | "
            f"<=drift={local_below.float().mean().item() * 100.0:.1f}% | "
            f">=2x={local_strong.float().mean().item() * 100.0:.1f}%"
        )


        rollout_step_mae.append(
            local_mae.mean().item()
            *
            1000.0
        )


        rollout_step_median.append(
            local_mae.median().item()
            *
            1000.0
        )


        rollout_step_changed05.append(
            local_changed_05.mean().item()
            *
            100.0
        )


        rollout_step_strong.append(
            local_strong.float().mean().item()
            *
            100.0
        )


    print(
        "==============================================="
    )


    # ========================================================
    # ACTION RANGE
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "ACTION PARAMETER RANGE"
    )

    print(
        "==============================================="
    )


    print(
        f"x       : "
        f"{x.min().item():.6f} "
        f"~ {x.max().item():.6f} m"
    )


    print(
        f"y       : "
        f"{y.min().item():.6f} "
        f"~ {y.max().item():.6f} m"
    )


    print(
        f"theta   : "
        f"{theta.min().item():+.4f} "
        f"~ {theta.max().item():+.4f} rad"
    )


    print(
        f"length  : "
        f"{length_mm.min().item():.3f} "
        f"~ {length_mm.max().item():.3f} mm"
    )


    print(
        f"depth   : "
        f"{depth_mm.min().item():.3f} "
        f"~ {depth_mm.max().item():.3f} mm"
    )


    print(
        "==============================================="
    )


    # ========================================================
    # DEPTH GROUPS
    # ========================================================

    if HAVE_COLLECTION_CONFIG:

        depth_groups = {

            "shallow":
            (
                (depth >= COLLECTION_DEPTH_SHALLOW_MIN)
                &
                (depth <= COLLECTION_DEPTH_SHALLOW_MAX)
            ),

            "medium":
            (
                (depth >= COLLECTION_DEPTH_MEDIUM_MIN)
                &
                (depth <= COLLECTION_DEPTH_MEDIUM_MAX)
            ),

            "deep":
            (
                (depth >= COLLECTION_DEPTH_DEEP_MIN)
                &
                (depth <= COLLECTION_DEPTH_DEEP_MAX)
            ),
        }


    else:

        depth_groups = {

            "shallow <3mm":
            (
                depth_mm
                <
                3.0
            ),

            "medium 3-6mm":
            (
                (depth_mm >= 3.0)
                &
                (depth_mm < 6.0)
            ),

            "deep >=6mm":
            (
                depth_mm
                >=
                6.0
            ),
        }


    print_group_summary(
        "DEFORMATION BY DEPTH",
        depth_groups,
        mae,
        max_error,
        changed_01,
        changed_05,
        changed_10,
        strong,
    )


    # ========================================================
    # LENGTH GROUPS
    # ========================================================

    if HAVE_COLLECTION_CONFIG:

        length_groups = {

            "short":
            (
                (length >= COLLECTION_LENGTH_SHORT_MIN)
                &
                (length <= COLLECTION_LENGTH_SHORT_MAX)
            ),

            "medium":
            (
                (length >= COLLECTION_LENGTH_MEDIUM_MIN)
                &
                (length <= COLLECTION_LENGTH_MEDIUM_MAX)
            ),

            "long":
            (
                (length >= COLLECTION_LENGTH_LONG_MIN)
                &
                (length <= COLLECTION_LENGTH_LONG_MAX)
            ),
        }


    else:

        # Only used if the current config cannot be imported.
        length_min = float(
            length_mm.min()
        )


        length_max = float(
            length_mm.max()
        )


        span = (
            length_max
            -
            length_min
        )


        boundary_1 = (
            length_min
            +
            span
            /
            3.0
        )


        boundary_2 = (
            length_min
            +
            2.0
            *
            span
            /
            3.0
        )


        length_groups = {

            "short":
            (
                length_mm
                <
                boundary_1
            ),

            "medium":
            (
                (length_mm >= boundary_1)
                &
                (length_mm < boundary_2)
            ),

            "long":
            (
                length_mm
                >=
                boundary_2
            ),
        }


    print_group_summary(
        "DEFORMATION BY LENGTH",
        length_groups,
        mae,
        max_error,
        changed_01,
        changed_05,
        changed_10,
        strong,
    )


    # ========================================================
    # THETA QUADRANTS
    # ========================================================

    theta_wrapped = torch.atan2(
        torch.sin(
            theta
        ),
        torch.cos(
            theta
        ),
    )


    theta_groups = {

        "+X direction":
        (
            torch.abs(
                theta_wrapped
            )
            <
            math.pi
            /
            4.0
        ),

        "+Y direction":
        (
            (theta_wrapped >= math.pi / 4.0)
            &
            (theta_wrapped < 3.0 * math.pi / 4.0)
        ),

        "-Y direction":
        (
            (theta_wrapped <= -math.pi / 4.0)
            &
            (theta_wrapped > -3.0 * math.pi / 4.0)
        ),

        "-X direction":
        (
            torch.abs(
                theta_wrapped
            )
            >=
            3.0
            *
            math.pi
            /
            4.0
        ),
    }


    print_group_summary(
        "DEFORMATION BY THETA DIRECTION",
        theta_groups,
        mae,
        max_error,
        changed_01,
        changed_05,
        changed_10,
        strong,
    )


    # ========================================================
    # CORRELATION
    # ========================================================

    depth_mae_corr = correlation(
        depth_mm,
        mae_mm,
    )


    depth_max_corr = correlation(
        depth_mm,
        max_mm,
    )


    length_mae_corr = correlation(
        length_mm,
        mae_mm,
    )


    length_max_corr = correlation(
        length_mm,
        max_mm,
    )


    print()
    print(
        "==============================================="
    )

    print(
        "ACTION / DEFORMATION CORRELATION"
    )

    print(
        "==============================================="
    )


    print(
        f"corr(depth, MAE)   = "
        f"{depth_mae_corr:+.4f}"
    )


    print(
        f"corr(depth, MAX)   = "
        f"{depth_max_corr:+.4f}"
    )


    print(
        f"corr(length, MAE)  = "
        f"{length_mae_corr:+.4f}"
    )


    print(
        f"corr(length, MAX)  = "
        f"{length_max_corr:+.4f}"
    )


    print(
        "==============================================="
    )


    # ========================================================
    # TOP / BOTTOM SAMPLES
    # ========================================================

    k = min(
        10,
        len(
            mae
        ),
    )


    smallest = torch.argsort(
        mae
    )[
        :k
    ]


    largest = torch.argsort(
        mae,
        descending=True,
    )[
        :k
    ]


    print()
    print(
        "==============================================="
    )

    print(
        f"{k} SMALLEST DEFORMATIONS"
    )

    print(
        "==============================================="
    )


    for idx in smallest.tolist():

        describe_sample(
            idx,
            actions,
            mae,
            max_error,
            changed_05,
            rollout_ids,
            rollout_steps,
        )


    print()
    print(
        "==============================================="
    )

    print(
        f"{k} LARGEST DEFORMATIONS"
    )

    print(
        "==============================================="
    )


    for idx in largest.tolist():

        describe_sample(
            idx,
            actions,
            mae,
            max_error,
            changed_05,
            rollout_ids,
            rollout_steps,
        )


    # ========================================================
    # PLOT:
    # MAE BY SAMPLE
    # ========================================================

    sample_x = list(
        range(
            1,
            n + 1
        )
    )


    plt.figure(
        figsize=(
            12,
            5,
        )
    )


    plt.plot(
        sample_x,
        mae_mm.numpy(),
        linewidth=1.0,
    )


    plt.axhline(
        NO_ACTION_MAE_MM,
        linestyle="--",
        label="No-action MAE",
    )


    plt.axhline(
        NO_ACTION_MAE_MM
        *
        2.0,
        linestyle=":",
        label="2x no-action MAE",
    )


    plt.xlabel(
        "Sample index"
    )


    plt.ylabel(
        "Mean |delta height| (mm)"
    )


    plt.title(
        "Deformation magnitude across dataset"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        /
        "mae_by_sample.png",
        dpi=160,
    )


    plt.close()


    # ========================================================
    # PLOT:
    # MAE HISTOGRAM
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )


    bins = max(
        10,
        min(
            50,
            n
        ),
    )


    plt.hist(
        mae_mm.numpy(),
        bins=bins,
    )


    plt.axvline(
        NO_ACTION_MAE_MM,
        linestyle="--",
        label="No-action MAE",
    )


    plt.axvline(
        NO_ACTION_MAE_MM
        *
        2.0,
        linestyle=":",
        label="2x no-action MAE",
    )


    plt.xlabel(
        "Mean |delta height| (mm)"
    )


    plt.ylabel(
        "Samples"
    )


    plt.title(
        "Dataset deformation distribution"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        /
        "mae_histogram.png",
        dpi=160,
    )


    plt.close()


    # ========================================================
    # PLOT:
    # ROLLOUT STEP MAE
    # ========================================================

    rollout_x = list(
        range(
            1,
            rollout_length
            +
            1
        )
    )


    plt.figure(
        figsize=(
            9,
            5,
        )
    )


    plt.plot(
        rollout_x,
        rollout_step_mae,
        marker="o",
        label="Mean MAE",
    )


    plt.plot(
        rollout_x,
        rollout_step_median,
        marker="o",
        label="Median MAE",
    )


    plt.axhline(
        NO_ACTION_MAE_MM,
        linestyle="--",
        label="No-action MAE",
    )


    plt.xlabel(
        "Rollout step"
    )


    plt.ylabel(
        "Mean |delta height| (mm)"
    )


    plt.title(
        "Deformation by rollout step"
    )


    plt.xticks(
        rollout_x
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        /
        "mae_by_rollout_step.png",
        dpi=160,
    )


    plt.close()


    # ========================================================
    # PLOT:
    # DEPTH VS MAE
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )


    plt.scatter(
        depth_mm.numpy(),
        mae_mm.numpy(),
        s=20,
    )


    plt.axhline(
        NO_ACTION_MAE_MM,
        linestyle="--",
        label="No-action MAE",
    )


    plt.xlabel(
        "Depth (mm)"
    )


    plt.ylabel(
        "Mean |delta height| (mm)"
    )


    plt.title(
        "Depth vs deformation"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        /
        "depth_vs_mae.png",
        dpi=160,
    )


    plt.close()


    # ========================================================
    # PLOT:
    # LENGTH VS MAE
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )


    plt.scatter(
        length_mm.numpy(),
        mae_mm.numpy(),
        s=20,
    )


    plt.axhline(
        NO_ACTION_MAE_MM,
        linestyle="--",
        label="No-action MAE",
    )


    plt.xlabel(
        "Length (mm)"
    )


    plt.ylabel(
        "Mean |delta height| (mm)"
    )


    plt.title(
        "Length vs deformation"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        /
        "length_vs_mae.png",
        dpi=160,
    )


    plt.close()


    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print()
    print(
        "==============================================="
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "==============================================="
    )


    print(
        "saved:"
    )


    print(
        " ",
        OUTPUT_DIR
        /
        "mae_by_sample.png"
    )


    print(
        " ",
        OUTPUT_DIR
        /
        "mae_histogram.png"
    )


    print(
        " ",
        OUTPUT_DIR
        /
        "mae_by_rollout_step.png"
    )


    print(
        " ",
        OUTPUT_DIR
        /
        "depth_vs_mae.png"
    )


    print(
        " ",
        OUTPUT_DIR
        /
        "length_vs_mae.png"
    )


    print(
        "==============================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        states,
        actions,
        next_states,
        rollout_ids,
        rollout_steps,

    ) = load_dataset()


    analyze(
        states,
        actions,
        next_states,
        rollout_ids,
        rollout_steps,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()