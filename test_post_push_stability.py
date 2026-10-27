import os

os.environ["VS_SHOW_VIEWER"] = "1"
os.environ["VS_MATERIAL"] = "medium"


import torch

import vs_simulator as sim


# ============================================================
# METRIC
# ============================================================

def print_diff(
    name,
    before,
    after,
):

    diff = (
        after
        -
        before
    )

    abs_diff = torch.abs(
        diff
    )


    mae = torch.mean(
        abs_diff
    )


    max_error = torch.max(
        abs_diff
    )


    changed_01 = torch.mean(
        (
            abs_diff
            >
            0.0001
        ).float()
    )


    changed_05 = torch.mean(
        (
            abs_diff
            >
            0.0005
        ).float()
    )


    changed_10 = torch.mean(
        (
            abs_diff
            >
            0.0010
        ).float()
    )


    print()
    print(
        "==============================================="
    )

    print(
        name
    )

    print(
        "==============================================="
    )

    print(
        f"MAE       = "
        f"{mae.item() * 1000.0:.6f} mm"
    )

    print(
        f"MAX ERROR = "
        f"{max_error.item() * 1000.0:.6f} mm"
    )

    print(
        f"|diff| > 0.1 mm = "
        f"{changed_01.item() * 100.0:.2f}%"
    )

    print(
        f"|diff| > 0.5 mm = "
        f"{changed_05.item() * 100.0:.2f}%"
    )

    print(
        f"|diff| > 1.0 mm = "
        f"{changed_10.item() * 100.0:.2f}%"
    )

    print(
        "==============================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    simulator = sim.initialize()


    print()
    print(
        "==============================================="
    )

    print(
        "POST-PUSH PASSIVE STABILITY TEST"
    )

    print(
        "==============================================="
    )


    # ========================================================
    # TEST 1
    # Initial no-action drift over action-like duration
    # ========================================================

    initial_0 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    # Approximately one action worth of simulation time.
    sim.hold(
        37
    )


    initial_37 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    print_diff(
        "INITIAL: NO ACTION 37 STEPS",
        initial_0,
        initial_37,
    )


    # ========================================================
    # PUSH
    # ========================================================

    before_push = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    action = torch.tensor(
        [
            0.44,
            0.00,
            0.00,
            0.040,
            0.005,
        ],
        device=sim.DEVICE,
        dtype=sim.FLOAT_DTYPE,
    )


    print()
    print(
        "Executing one reference push..."
    )

    print(
        "x      = 0.440"
    )

    print(
        "y      = 0.000"
    )

    print(
        "theta  = 0.000"
    )

    print(
        "length = 40 mm"
    )

    print(
        "depth  = 5 mm"
    )


    sim.linear_push(
        action,
        action_index=0,
    )


    immediately_after_push = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    print_diff(
        "ACTUAL PUSH",
        before_push,
        immediately_after_push,
    )


    # ========================================================
    # AFTER PUSH:
    # passive relaxation 1
    # ========================================================

    sim.hold(
        37
    )


    after_hold_37 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    print_diff(
        "POST-PUSH PASSIVE HOLD 37 STEPS",
        immediately_after_push,
        after_hold_37,
    )


    # ========================================================
    # AFTER PUSH:
    # passive relaxation 2
    # ========================================================

    sim.hold(
        37
    )


    after_hold_74 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    print_diff(
        "POST-PUSH PASSIVE HOLD 37 -> 74 STEPS",
        after_hold_37,
        after_hold_74,
    )


    # ========================================================
    # AFTER PUSH:
    # passive relaxation 3
    # ========================================================

    sim.hold(
        37
    )


    after_hold_111 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    print_diff(
        "POST-PUSH PASSIVE HOLD 74 -> 111 STEPS",
        after_hold_74,
        after_hold_111,
    )


    # ========================================================
    # TOTAL POST-PUSH RELAXATION
    # ========================================================

    print_diff(
        "TOTAL POST-PUSH PASSIVE CHANGE 0 -> 111 STEPS",
        immediately_after_push,
        after_hold_111,
    )


    print()
    print(
        "==============================================="
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()