"""중앙 고정 누르기 테스트."""

import math

import torch
import vs_simulator as sim

from vs_config import (
    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_BOTTOM_Z,
    DT,
)


HOVER_GAP = 0.015
HOLD_SECONDS = 0.2
DEPTHS = (0.004, 0.006, 0.008)


def surface_height(s, height_map, x, y):
    value = s.sample_surface_height_gpu(
        height_map,
        x,
        y,
    )

    return float(value.reshape(-1)[0])


def prepare_test():
    # sim.initialize()는 호출하지 않는다.
    # 기존의 긴 HOME XY / HOME DOWN 경로를 건너뛴다.
    s = sim.VisualSculptingSimulator()

    print(
        "\n[PREPARE] 초기 점토 안정화",
        flush=True,
    )

    for _ in range(8):
        s.robot.control_dofs_position(
            position=s.initial_q,
            dofs_idx_local=s.dof_idx,
        )
        s.step()

    # 현재 초기 자세의 수평 방향을 유지하면서
    # 도구의 로컬 +Z축을 수직 아래로 향하게 한다.
    reference = torch.as_tensor(
        s.tip.get_quat(relative=False),
        device=s.device,
        dtype=s.dtype,
    ).reshape(-1)[:4]

    down = torch.stack(
        [
            torch.zeros_like(reference[0]),
            reference[1],
            reference[2],
            torch.zeros_like(reference[3]),
        ]
    )

    norm = torch.linalg.vector_norm(down)

    if float(norm) < 1e-6:
        raise RuntimeError(
            "초기 도구 자세에서 아래 방향을 "
            "결정하지 못했습니다."
        )

    s.tool_quat_down = down / norm

    x = float(CLAY_CENTER_X)
    y = float(CLAY_CENTER_Y)

    height_map = (
        s.get_height_gpu()
        .detach()
        .clone()
    )

    surface = surface_height(
        s,
        height_map,
        x,
        y,
    )

    home = torch.tensor(
        [
            x,
            y,
            surface + HOVER_GAP,
        ],
        device=s.device,
        dtype=s.dtype,
    )

    s.last_q = (
        s.robot.get_dofs_position(
            dofs_idx_local=s.dof_idx
        )
        .reshape(-1)[:7]
        .detach()
        .clone()
    )

    q_home = s.solve_ik(home)

    # 테스트 초기 배치에서만 관절 위치를 직접 설정한다.
    # 실제 누르기·유지·상승에서는 직접 위치 설정을 사용하지 않는다.
    s.robot.set_dofs_position(
        q_home,
        dofs_idx_local=s.dof_idx,
        zero_velocity=True,
    )

    s.robot.control_dofs_position(
        position=q_home,
        dofs_idx_local=s.dof_idx,
    )

    s.last_q = q_home.detach().clone()

    actual = s._action_tip_position()

    error = float(
        torch.linalg.vector_norm(
            actual - home
        )
    )

    gap = float(actual[2]) - surface

    print(
        "[INITIAL PLACEMENT] "
        f"surface={surface * 1000:.2f}mm "
        f"target_z={float(home[2]) * 1000:.2f}mm "
        f"actual_z={float(actual[2]) * 1000:.2f}mm "
        f"gap={gap * 1000:.2f}mm "
        f"error={error * 1000:.2f}mm",
        flush=True,
    )

    if (
        not math.isfinite(error)
        or error > 0.003
        or gap < 0.008
    ):
        raise RuntimeError(
            "초기 IK 배치가 목표와 맞지 않습니다. "
            "누르기를 시작하지 않습니다."
        )

    s.initialized = True
    s.step(force_visual=True)

    return s, x, y


def main():
    s, x, y = prepare_test()

    hold_steps = max(
        1,
        int(math.ceil(HOLD_SECONDS / DT)),
    )

    last_ctx = None

    for index, depth in enumerate(DEPTHS, 1):
        before = (
            s.get_height_gpu()
            .detach()
            .clone()
        )

        surface = surface_height(
            s,
            before,
            x,
            y,
        )

        target_z = surface - depth

        if target_z < float(CLAY_BOTTOM_Z) + 0.003:
            raise RuntimeError(
                "누름 목표가 바닥에 너무 가깝습니다."
            )

        ctx = s._action_context()

        # 같은 중앙 위치에서 반복한다.
        hover = torch.tensor(
            [
                x,
                y,
                surface + HOVER_GAP,
            ],
            device=s.device,
            dtype=s.dtype,
        )

        print(
            f"\n[PRESS {index}/{len(DEPTHS)}] "
            f"surface={surface * 1000:.2f}mm "
            f"depth={depth * 1000:.1f}mm "
            f"target_z={target_z * 1000:.2f}mm",
            flush=True,
        )

        s._action_move(
            hover,
            6,
            ctx,
            "LOW HOVER",
        )

        # 수평 이동 없이 수직으로 요청 깊이까지 누른다.
        target = hover.clone()
        target[2] = target_z

        s._action_move(
            target,
            12,
            ctx,
            "PRESS DOWN",
            surface,
            contact=True,
        )

        s._action_track(
            target,
            ctx,
            "DEPTH VERIFIED",
            surface,
            report=True,
            position_tol=0.001,
            z_tol=0.0003,
        )

        # 실제로 누른 위치에서 유지한다.
        for _ in range(hold_steps):
            s._action_command(
                ctx["q"],
                ctx,
            )

        s._action_track(
            target,
            ctx,
            "HOLD VERIFIED",
            surface,
            report=True,
            position_tol=0.001,
            z_tol=0.0003,
        )

        actual = s._action_tip_position()

        achieved = (
            surface - float(actual[2])
        )

        if abs(achieved - depth) > 0.0003:
            raise RuntimeError(
                "실제 도구가 요청 깊이에 "
                "도달하지 못했습니다."
            )

        pressed = (
            s.get_height_gpu()
            .detach()
            .clone()
        )

        # 실제 XY를 유지한 채 수직으로 빠져나온다.
        lift = actual.clone()

        lift[2] = max(
            surface,
            float(pressed.max()),
        ) + HOVER_GAP

        s._action_move(
            lift,
            10,
            ctx,
            "RETRACT",
        )

        for _ in range(hold_steps):
            s._action_command(
                ctx["q"],
                ctx,
            )

        s._action_track(
            lift,
            ctx,
            "RETRACT VERIFIED",
            report=True,
            position_tol=0.004,
            z_tol=0.0015,
        )

        after = (
            s.get_height_gpu()
            .detach()
            .clone()
        )

        pressed_max = float(
            (pressed - before).abs().max()
        ) * 1000.0

        remaining_max = float(
            (after - before).abs().max()
        ) * 1000.0

        center_pressed = surface_height(
            s,
            pressed,
            x,
            y,
        )

        center_after = surface_height(
            s,
            after,
            x,
            y,
        )

        center_down_pressed = (
            surface - center_pressed
        ) * 1000.0

        center_down_after = (
            surface - center_after
        ) * 1000.0

        print(
            f"[RESULT {index}] "
            f"achieved_reference_depth="
            f"{achieved * 1000:.3f}mm "
            f"pressed_height_MAX={pressed_max:.3f}mm "
            f"after_lift_height_MAX={remaining_max:.3f}mm "
            f"center_down_during_press="
            f"{center_down_pressed:.3f}mm "
            f"center_down_after_lift="
            f"{center_down_after:.3f}mm",
            flush=True,
        )

        last_ctx = ctx
        s.step(force_visual=True)

    print(
        "\n누르기 3회 완료. "
        "점토 초기화 및 데이터셋 저장 없음.",
        flush=True,
    )

    if s.viewer_alive and last_ctx is not None:
        print(
            "결과 확인 중. 종료: Ctrl+C",
            flush=True,
        )

        while True:
            s._action_command(
                last_ctx["q"],
                last_ctx,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass