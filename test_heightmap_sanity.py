import os

os.environ["VS_SHOW_VIEWER"] = "1"
os.environ["VS_MATERIAL"] = "medium"


import math
import torch
import matplotlib.pyplot as plt

import vs_simulator as sim


# ============================================================
# UTILS
# ============================================================

def save_triplet(
    before,
    after,
    diff,
    title,
    output_path,
    action=None,
):

    before_np = (
        before
        .detach()
        .cpu()
        .numpy()
    )


    after_np = (
        after
        .detach()
        .cpu()
        .numpy()
    )


    diff_np = (
        diff
        .detach()
        .cpu()
        .numpy()
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            5,
        ),
    )


    state_min = min(
        before_np.min(),
        after_np.min(),
    )


    state_max = max(
        before_np.max(),
        after_np.max(),
    )


    # ========================================================
    # BEFORE
    # ========================================================

    im0 = axes[0].imshow(
        before_np,
        vmin=state_min,
        vmax=state_max,
        origin="lower",
    )


    axes[0].set_title(
        "Before"
    )


    plt.colorbar(
        im0,
        ax=axes[0],
    )


    # ========================================================
    # AFTER
    # ========================================================

    im1 = axes[1].imshow(
        after_np,
        vmin=state_min,
        vmax=state_max,
        origin="lower",
    )


    axes[1].set_title(
        "After"
    )


    plt.colorbar(
        im1,
        ax=axes[1],
    )


    # ========================================================
    # DIFFERENCE
    # ========================================================

    diff_max = max(
        abs(
            diff_np
        ).max(),
        1e-8,
    )


    im2 = axes[2].imshow(
        diff_np,
        vmin=-diff_max,
        vmax=diff_max,
        origin="lower",
    )


    axes[2].set_title(
        "Difference"
    )


    plt.colorbar(
        im2,
        ax=axes[2],
    )


    # ========================================================
    # DRAW ACTION TRAJECTORY
    #
    # action =
    # [x, y, theta, length, depth]
    # ========================================================

    if action is not None:

        action_cpu = (
            action
            .detach()
            .cpu()
        )


        x = float(
            action_cpu[
                0
            ]
        )


        y = float(
            action_cpu[
                1
            ]
        )


        theta = float(
            action_cpu[
                2
            ]
        )


        length = float(
            action_cpu[
                3
            ]
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


        # ----------------------------------------------------
        # World -> pixel coordinate
        # ----------------------------------------------------

        def world_to_pixel(
            wx,
            wy,
        ):

            px = (
                (
                    wx
                    -
                    (
                        sim.CLAY_CENTER_X
                        -
                        sim.CLAY_SIZE_X
                        /
                        2.0
                    )
                )
                /
                sim.CLAY_SIZE_X
                *
                (
                    before.shape[
                        1
                    ]
                    -
                    1
                )
            )


            py = (
                (
                    wy
                    -
                    (
                        sim.CLAY_CENTER_Y
                        -
                        sim.CLAY_SIZE_Y
                        /
                        2.0
                    )
                )
                /
                sim.CLAY_SIZE_Y
                *
                (
                    before.shape[
                        0
                    ]
                    -
                    1
                )
            )


            return (
                px,
                py,
            )


        start_px, start_py = (
            world_to_pixel(
                x,
                y,
            )
        )


        end_px, end_py = (
            world_to_pixel(
                end_x,
                end_y,
            )
        )


        for ax in axes:

            # start point
            ax.scatter(
                [
                    start_px
                ],
                [
                    start_py
                ],
                s=45,
                marker="o",
            )


            # endpoint
            ax.scatter(
                [
                    end_px
                ],
                [
                    end_py
                ],
                s=45,
                marker="x",
            )


            # trajectory
            ax.plot(
                [
                    start_px,
                    end_px,
                ],
                [
                    start_py,
                    end_py,
                ],
                linewidth=2,
            )


    for ax in axes:

        ax.axis(
            "off"
        )


    fig.suptitle(
        title
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    name,
    before,
    after,
):

    diff = (
        after
        -
        before
    )


    mae = torch.mean(
        torch.abs(
            diff
        )
    )


    mse = torch.mean(
        diff
        *
        diff
    )


    max_err = torch.max(
        torch.abs(
            diff
        )
    )


    changed_01mm = torch.mean(
        (
            torch.abs(
                diff
            )
            >
            0.0001
        ).float()
    )


    changed_05mm = torch.mean(
        (
            torch.abs(
                diff
            )
            >
            0.0005
        ).float()
    )


    changed_10mm = torch.mean(
        (
            torch.abs(
                diff
            )
            >
            0.0010
        ).float()
    )


    values = torch.stack(
        [
            mae,
            mse,
            max_err,
            changed_01mm,
            changed_05mm,
            changed_10mm,
        ]
    ).detach().cpu().tolist()


    (
        mae_v,
        mse_v,
        max_v,
        c01,
        c05,
        c10,
    ) = values


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
        f"{mae_v * 1000.0:.6f} mm"
    )


    print(
        f"MSE       = "
        f"{mse_v:.9e}"
    )


    print(
        f"MAX ERROR = "
        f"{max_v * 1000.0:.6f} mm"
    )


    print(
        f"|diff| > 0.1 mm = "
        f"{c01 * 100.0:.2f}% pixels"
    )


    print(
        f"|diff| > 0.5 mm = "
        f"{c05 * 100.0:.2f}% pixels"
    )


    print(
        f"|diff| > 1.0 mm = "
        f"{c10 * 100.0:.2f}% pixels"
    )


    print(
        "==============================================="
    )


    return diff


# ============================================================
# MAX ERROR LOCATION
# ============================================================

def print_max_error_location(
    before,
    after,
    diff,
):

    abs_diff = torch.abs(
        diff
    )


    flat_index = torch.argmax(
        abs_diff
    )


    width = abs_diff.shape[
        1
    ]


    max_y = (
        flat_index
        //
        width
    )


    max_x = (
        flat_index
        %
        width
    )


    max_y_i = int(
        max_y.item()
    )


    max_x_i = int(
        max_x.item()
    )


    before_value = float(
        before[
            max_y_i,
            max_x_i
        ]
        *
        1000.0
    )


    after_value = float(
        after[
            max_y_i,
            max_x_i
        ]
        *
        1000.0
    )


    diff_value = float(
        diff[
            max_y_i,
            max_x_i
        ]
        *
        1000.0
    )


    # ========================================================
    # PIXEL -> WORLD XY
    # ========================================================

    x_world = (
        sim.CLAY_CENTER_X
        -
        sim.CLAY_SIZE_X
        /
        2.0
        +
        (
            max_x_i
            /
            (
                abs_diff.shape[
                    1
                ]
                -
                1
            )
        )
        *
        sim.CLAY_SIZE_X
    )


    y_world = (
        sim.CLAY_CENTER_Y
        -
        sim.CLAY_SIZE_Y
        /
        2.0
        +
        (
            max_y_i
            /
            (
                abs_diff.shape[
                    0
                ]
                -
                1
            )
        )
        *
        sim.CLAY_SIZE_Y
    )


    print()
    print(
        "==============================================="
    )


    print(
        "MAX PUSH DEFORMATION LOCATION"
    )


    print(
        "==============================================="
    )


    print(
        "pixel x =",
        max_x_i
    )


    print(
        "pixel y =",
        max_y_i
    )


    print(
        f"world x = "
        f"{x_world:.6f} m"
    )


    print(
        f"world y = "
        f"{y_world:.6f} m"
    )


    print(
        f"before  = "
        f"{before_value:.6f} mm"
    )


    print(
        f"after   = "
        f"{after_value:.6f} mm"
    )


    print(
        f"diff    = "
        f"{diff_value:.6f} mm"
    )


    print(
        "==============================================="
    )


# ============================================================
# ACTION INFO
# ============================================================

def print_action_info(
    action,
):

    values = (
        action
        .detach()
        .cpu()
        .tolist()
    )


    (
        x,
        y,
        theta,
        length,
        depth,
    ) = values


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


    print()
    print(
        "==============================================="
    )


    print(
        "TEST ACTION"
    )


    print(
        "==============================================="
    )


    print(
        f"x       = "
        f"{x:.6f} m"
    )


    print(
        f"y       = "
        f"{y:.6f} m"
    )


    print(
        f"theta   = "
        f"{theta:.6f} rad "
        f"({math.degrees(theta):.2f} deg)"
    )


    print(
        f"length  = "
        f"{length * 1000.0:.3f} mm"
    )


    print(
        f"depth   = "
        f"{depth * 1000.0:.3f} mm"
    )


    print()
    print(
        "expected XY trajectory:"
    )


    print(
        f"start   = "
        f"({x:.6f}, {y:.6f}) m"
    )


    print(
        f"end     = "
        f"({end_x:.6f}, {end_y:.6f}) m"
    )


    print(
        "==============================================="
    )


# ============================================================
# TRAJECTORY DEBUG
# ============================================================

def print_trajectory(
    action,
    height_map,
):

    trajectory = (
        sim.build_push_trajectory_gpu(
            action,
            height_map,
        )
    )


    trajectory_cpu = (
        trajectory
        .detach()
        .cpu()
    )


    print()
    print(
        "==============================================="
    )


    print(
        "PUSH TRAJECTORY"
    )


    print(
        "==============================================="
    )


    print(
        "shape =",
        tuple(
            trajectory_cpu.shape
        )
    )


    print()


    for i in range(
        trajectory_cpu.shape[
            0
        ]
    ):

        x = float(
            trajectory_cpu[
                i,
                0
            ]
        )


        y = float(
            trajectory_cpu[
                i,
                1
            ]
        )


        z = float(
            trajectory_cpu[
                i,
                2
            ]
        )


        print(
            f"{i:02d} | "
            f"x={x:.6f} "
            f"y={y:.6f} "
            f"z={z * 1000.0:.3f} mm"
        )


    start = trajectory_cpu[
        0
    ]


    end = trajectory_cpu[
        -1
    ]


    xy_distance = torch.sqrt(
        (
            end[
                0
            ]
            -
            start[
                0
            ]
        )
        **
        2
        +
        (
            end[
                1
            ]
            -
            start[
                1
            ]
        )
        **
        2
    )


    z_change = (
        end[
            2
        ]
        -
        start[
            2
        ]
    )


    print()


    print(
        "trajectory XY displacement =",
        f"{float(xy_distance) * 1000.0:.3f} mm"
    )


    print(
        "trajectory Z change        =",
        f"{float(z_change) * 1000.0:.3f} mm"
    )


    print(
        "==============================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        "results",
        exist_ok=True,
    )


    simulator = sim.initialize()

    print("추가 안정화: 100 step 대기")
    sim.hold(100)

    # ========================================================
    # TEST 1:
    # NO ACTION
    # ========================================================

    print()
    print(
        "TEST 1: NO-ACTION STABILITY"
    )


    h0 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    sim.hold(
        20
    )


    h1 = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    diff_hold = print_metrics(
        "NO-ACTION HOLD",
        h0,
        h1,
    )


    save_triplet(
        h0,
        h1,
        diff_hold,
        "No-action Height-map Stability",
        "results/heightmap_no_action.png",
    )


    # ========================================================
    # TEST 2:
    # SINGLE VISUAL SCULPTING PUSH
    # ========================================================

    print()
    print(
        "TEST 2: SINGLE VISUAL SCULPTING PUSH"
    )


    before_push = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    # ========================================================
    # VISUAL SCULPTING 5D ACTION
    #
    # a =
    #
    # [
    #     x,
    #     y,
    #     theta,
    #     length,
    #     depth,
    # ]
    #
    # Interpretation:
    #
    # x, y
    #   start position
    #
    # theta
    #   XY push direction
    #
    # length
    #   horizontal trajectory displacement
    #
    # depth
    #   final penetration depth
    #
    # This is NOT a constant-depth scraping action.
    #
    # As XY moves along theta, penetration gradually
    # increases toward depth.
    # ========================================================

    action = torch.tensor(
        [
            0.440,      # x
            0.000,      # y

            0.000,      # theta = +X direction

            0.030,      # length = 30 mm

            0.100,      # depth = 10 mm
        ],

        device=sim.DEVICE,

        dtype=sim.FLOAT_DTYPE,
    )


    # ========================================================
    # ACTION INFO
    # ========================================================

    print_action_info(
        action
    )


    # ========================================================
    # TRAJECTORY PREVIEW
    #
    # Important:
    # inspect trajectory numerically BEFORE execution.
    # ========================================================

    print_trajectory(
        action,
        before_push,
    )


    # ========================================================
    # EXECUTE PUSH
    # ========================================================

    sim.linear_push(
        action,
        action_index=0,
    )


    # ========================================================
    # STATE AFTER PUSH
    # ========================================================

    after_push = (
        sim.get_height_gpu()
        .detach()
        .clone()
    )


    # ========================================================
    # METRICS
    # ========================================================

    diff_push = print_metrics(
        "SINGLE VISUAL SCULPTING PUSH",
        before_push,
        after_push,
    )


    # ========================================================
    # MAX DEFORMATION LOCATION
    # ========================================================

    print_max_error_location(
        before_push,
        after_push,
        diff_push,
    )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    save_triplet(
        before_push,
        after_push,
        diff_push,
        "Single Visual Sculpting Push Height-map Difference",
        "results/heightmap_single_push.png",
        action=action,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

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


    print(
        "Visual Sculpting action:"
    )


    print(
        "  [x, y, theta, length, depth]"
    )


    print()


    print(
        "Expected test motion:"
    )


    print(
        "  start near surface at (0.440, 0.000)"
    )


    print(
        "  move approximately +30 mm in X"
    )


    print(
        "  progressively increase penetration"
    )


    print(
        "  finish approximately 5 mm below local surface"
    )


    print(
        "  lift vertically from endpoint"
    )


    print()


    print(
        "saved:"
    )


    print(
        "  results/heightmap_no_action.png"
    )


    print(
        "  results/heightmap_single_push.png"
    )


    print(
        "==============================================="
    )


if __name__ == "__main__":

    main()