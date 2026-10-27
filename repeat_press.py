"""중앙에서 2mm → 3mm → 3mm 누르기와 상승을 반복한다."""

import math
import xml.etree.ElementTree as ET

import torch
import vs_simulator as sim

from vs_config import (
    CLAY_CENTER_X,
    CLAY_CENTER_Y,
    CLAY_BOTTOM_Z,
    DT,
)


DEPTHS = (0.002, 0.003, 0.003)

CLEARANCE = 0.015
WAYPOINT_DISTANCE = 0.001

POSITION_XY_TOL = 0.0015
POSITION_Z_TOL = 0.0005

MAX_EXTRA_DEPTH = 0.002
MAX_XY_DEVIATION = 0.006

HOLD_SECONDS = 0.2
AFTER_UP_SECONDS = 0.1

TRACK_MAX_STEPS = 300


class PressController:
    def __init__(self, simulator):
        self.s = simulator

        self.lower, self.upper = self.read_joint_limits()

        self.bias = torch.zeros_like(self.joints())
        self.last_command = self.joints()

        self.guard = None
        self.steps = 0

    def position(self):
        return (
            torch.as_tensor(
                self.s.tip.get_pos(relative=False),
                device=self.s.device,
                dtype=self.s.dtype,
            )
            .reshape(-1)[:3]
            .detach()
            .clone()
        )

    def joints(self):
        return (
            self.s.robot.get_dofs_position(
                dofs_idx_local=self.s.dof_idx
            )
            .reshape(-1)[:7]
            .detach()
            .clone()
        )

    def read_joint_limits(self):
        root = ET.parse(sim.ROBOT_URDF).getroot()

        xml_joints = {
            joint.attrib["name"]: joint
            for joint in root.findall("joint")
        }

        selected_indices = {
            int(index)
            for index in self.s.dof_idx
        }

        bounds = {}

        for joint in self.s.robot.joints:
            indices = [
                int(index)
                for index in joint.dofs_idx_local
            ]

            if not any(
                index in selected_indices
                for index in indices
            ):
                continue

            node = xml_joints.get(joint.name)

            limit = (
                node.find("limit")
                if node is not None
                else None
            )

            if len(indices) != 1 or limit is None:
                raise RuntimeError(
                    "관절 한계를 확인할 수 없음: "
                    + joint.name
                )

            bounds[indices[0]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )

        missing = selected_indices - set(bounds)

        if missing:
            raise RuntimeError(
                f"관절 한계 누락: {sorted(missing)}"
            )

        limits = torch.tensor(
            [
                bounds[int(index)]
                for index in self.s.dof_idx
            ],
            device=self.s.device,
            dtype=self.s.dtype,
        )

        return (
            limits[:, 0] + 0.001,
            limits[:, 1] - 0.001,
        )

    def solve_ik(self, target):
        target = torch.as_tensor(
            target,
            device=self.s.device,
            dtype=self.s.dtype,
        )

        q = self.s.robot.inverse_kinematics(
            link=self.s.tip,
            pos=target,
            quat=self.s.tool_quat_down,
            init_qpos=self.joints(),
            respect_joint_limit=True,
            max_samples=2,
            max_solver_iters=100,
            damping=0.08,
            pos_tol=0.0002,
            rot_tol=0.008,
            max_step_size=0.1,
            dofs_idx_local=self.s.dof_idx,
        )

        q = torch.as_tensor(
            q,
            device=self.s.device,
            dtype=self.s.dtype,
        ).reshape(-1)[:7]

        if q.numel() != 7:
            raise RuntimeError(
                "IK 결과가 7개 관절값이 아님."
            )

        if not bool(torch.isfinite(q).all()):
            raise RuntimeError(
                "IK가 유효하지 않은 관절값을 반환함."
            )

        self.s.last_q = q.detach().clone()

        return q

    def check_position(self):
        actual = self.position()

        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError(
                "실제 tip 위치가 유효하지 않음."
            )

        if self.guard is None:
            return

        reference_xy, minimum_z = self.guard

        if float(actual[2]) < minimum_z:
            raise RuntimeError(
                "요청 깊이보다 2mm 이상 과하게 내려감."
            )

        xy_error = float(
            torch.linalg.vector_norm(
                actual[:2] - reference_xy
            )
        )

        if xy_error > MAX_XY_DEVIATION:
            raise RuntimeError(
                f"수평 위치 이탈: {xy_error * 1000:.3f}mm"
            )

    def raw_step(self, command):
        self.s.robot.control_dofs_position(
            position=command,
            dofs_idx_local=self.s.dof_idx,
        )

        self.last_command = (
            command.detach().clone()
        )

        self.s.step()

        self.steps += 1
        self.check_position()

    def controlled_step(self, q):
        actual_q = self.joints()

        residual = torch.clamp(
            q - actual_q,
            -0.05,
            0.05,
        )

        bias = torch.clamp(
            self.bias + 6.0 * DT * residual,
            -0.12,
            0.12,
        )

        requested = q + bias

        command = torch.maximum(
            self.lower,
            torch.minimum(
                self.upper,
                requested,
            ),
        )

        # 관절 한계에서 적분 보정이 계속 쌓이지 않도록 제한.
        self.bias = (
            bias + command - requested
        ).detach()

        self.raw_step(command)

    def report(self, target, label, q=None):
        actual = self.position()
        difference = actual - target

        xy_error = float(
            torch.linalg.vector_norm(
                difference[:2]
            )
        )

        extra = ""

        if q is not None:
            joint_error = float(
                (q - self.joints()).abs().max()
            )

            extra = (
                f" joint_error_MAX={joint_error:.5f}rad"
            )

        print(
            f"[{label}] "
            f"target_z={float(target[2]) * 1000:.3f}mm "
            f"actual_z={float(actual[2]) * 1000:.3f}mm "
            f"xy_error={xy_error * 1000:.3f}mm "
            f"dz={float(difference[2]) * 1000:.3f}mm"
            f"{extra}",
            flush=True,
        )

    def track(
        self,
        target,
        label,
        z_tolerance=POSITION_Z_TOL,
    ):
        target = torch.as_tensor(
            target,
            device=self.s.device,
            dtype=self.s.dtype,
        )

        correction = torch.zeros_like(target)
        consecutive = 0

        for tick in range(TRACK_MAX_STEPS):
            if tick % 24 == 0:
                if tick:
                    correction = torch.clamp(
                        correction
                        + 0.35 * (
                            target - self.position()
                        ),
                        -0.008,
                        0.008,
                    )

                q = self.solve_ik(
                    target + correction
                )

            self.controlled_step(q)

            difference = (
                self.position() - target
            )

            xy_error = float(
                torch.linalg.vector_norm(
                    difference[:2]
                )
            )

            z_error = abs(
                float(difference[2])
            )

            reached = (
                xy_error <= POSITION_XY_TOL
                and z_error <= z_tolerance
            )

            consecutive = (
                consecutive + 1
                if reached
                else 0
            )

            if consecutive >= 3:
                return

            if tick in (119, 239):
                self.report(
                    target,
                    label + " WAIT",
                    q,
                )

        self.report(
            target,
            label + " FAILED",
            q,
        )

        raise RuntimeError(
            label + ": 位置 추종 제한시간 초과."
        )

    def vertical_move(self, target_z, label):
        start = self.position()

        end = start.clone()
        end[2] = float(target_z)

        distance = abs(
            float(end[2] - start[2])
        )

        count = max(
            1,
            int(
                math.ceil(
                    distance
                    / WAYPOINT_DISTANCE
                )
            ),
        )

        print(
            f"[{label} BEGIN] "
            f"start_z={float(start[2]) * 1000:.3f}mm "
            f"target_z={float(end[2]) * 1000:.3f}mm "
            f"waypoints={count}",
            flush=True,
        )

        # 실제 시작 위치에서 수직으로 이동한다.
        # 각 waypoint에 도달한 뒤 다음 위치를 명령한다.
        for index in range(count):
            target = torch.lerp(
                start,
                end,
                (index + 1) / count,
            )

            self.track(
                target,
                label,
            )

        self.report(
            end,
            label + " VERIFIED",
        )

        return end

    def hold(self, duration):
        # 도달에 사용한 보정 포함 명령을 그대로 유지한다.
        command = self.last_command.clone()

        for _ in range(
            max(1, round(duration / DT))
        ):
            self.raw_step(command)


@torch.no_grad()
def main():
    s = sim.VisualSculptingSimulator()
    controller = PressController(s)

    for _ in range(8):
        controller.raw_step(
            s.initial_q
        )

    reference = torch.as_tensor(
        s.tip.get_quat(relative=False),
        device=s.device,
        dtype=s.dtype,
    ).reshape(-1)[:4]

    # 현재 손목 방위를 유지하면서 도구 축을 아래로 정렬.
    down = reference.clone()
    down[0] = 0.0
    down[3] = 0.0

    norm = torch.linalg.vector_norm(down)

    if float(norm) < 1e-6:
        raise RuntimeError(
            "도구의 아래 방향 계산 실패."
        )

    s.tool_quat_down = down / norm

    initial_height = (
        s.get_height_gpu()
        .detach()
        .clone()
    )

    initial_surface = float(
        s.sample_surface_height_gpu(
            initial_height,
            CLAY_CENTER_X,
            CLAY_CENTER_Y,
        ).reshape(-1)[0]
    )

    hover_z = max(
        initial_surface,
        float(initial_height.max()),
    ) + CLEARANCE

    home = torch.tensor(
        [
            CLAY_CENTER_X,
            CLAY_CENTER_Y,
            hover_z,
        ],
        device=s.device,
        dtype=s.dtype,
    )

    q = controller.solve_ik(home)

    # 최초 한 번만 점토 위 공중에 배치한다.
    # 이후 누르기와 상승은 물리 시뮬레이션으로 수행한다.
    s.robot.set_dofs_position(
        q,
        dofs_idx_local=s.dof_idx,
        zero_velocity=True,
    )

    actual = controller.position()

    placement_error = float(
        torch.linalg.vector_norm(
            actual - home
        )
    )

    if placement_error > 0.003:
        raise RuntimeError(
            "초기 배치 오차가 3mm를 넘음."
        )

    if (
        float(actual[2])
        < float(initial_height.max()) + 0.008
    ):
        raise RuntimeError(
            "초기 배치가 점토에 너무 가까움."
        )

    controller.track(
        home,
        "START",
        z_tolerance=0.001,
    )

    controller.report(
        home,
        "START VERIFIED",
    )

    for index, depth in enumerate(
        DEPTHS,
        start=1,
    ):
        before = (
            s.get_height_gpu()
            .detach()
            .clone()
        )

        current = controller.position()

        surface = float(
            s.sample_surface_height_gpu(
                before,
                float(current[0]),
                float(current[1]),
            ).reshape(-1)[0]
        )

        target_z = surface - depth

        if (
            target_z
            < float(CLAY_BOTTOM_Z) + 0.003
        ):
            raise RuntimeError(
                "다음 누름 목표가 바닥에 너무 가까움."
            )

        retract_z = max(
            surface,
            float(before.max()),
        ) + CLEARANCE

        controller.guard = (
            current[:2].clone(),
            target_z - MAX_EXTRA_DEPTH,
        )

        press_error = None
        achieved_depth = None

        print(
            f"\n[PRESS {index}/{len(DEPTHS)}] "
            f"surface={surface * 1000:.3f}mm "
            f"requested_depth={depth * 1000:.1f}mm "
            f"target_z={target_z * 1000:.3f}mm",
            flush=True,
        )

        try:
            controller.vertical_move(
                target_z,
                "DOWN",
            )

            controller.hold(
                HOLD_SECONDS
            )

            actual = controller.position()

            if (
                abs(float(actual[2]) - target_z)
                > 0.001
            ):
                raise RuntimeError(
                    "유지 중 높이 오차가 1mm를 넘음."
                )

            achieved_depth = (
                surface - float(actual[2])
            )

            print(
                "[HOLD VERIFIED] "
                f"reference_depth="
                f"{achieved_depth * 1000:.3f}mm",
                flush=True,
            )

        except Exception as exc:
            press_error = exc

            actual = controller.position()

            print(
                f"[PRESS FAILED] {exc} "
                f"actual_z={float(actual[2]) * 1000:.3f}mm",
                flush=True,
            )

        finally:
            # 하강 실패나 Ctrl+C에도 상승을 시도한다.
            # 상승 성공 자체는 별도로 확인한다.
            controller.guard = None

            try:
                current_height = (
                    s.get_height_gpu()
                    .detach()
                    .clone()
                )

                safe_z = max(
                    retract_z,
                    float(current_height.max())
                    + CLEARANCE,
                    float(controller.position()[2]),
                )

                up_target = controller.vertical_move(
                    safe_z,
                    "UP",
                )

                controller.hold(
                    AFTER_UP_SECONDS
                )

                # 상승 후 유지 중에도 높이가 유지되는지 확인.
                controller.track(
                    up_target,
                    "UP FINAL",
                    z_tolerance=0.001,
                )

                controller.report(
                    up_target,
                    "UP FINAL VERIFIED",
                )

            except Exception as retract_error:
                print(
                    f"[UP FAILED] {retract_error}",
                    flush=True,
                )

                raise RuntimeError(
                    "상승 실패. 다음 누르기를 실행하지 않음."
                ) from retract_error

        if press_error is not None:
            raise RuntimeError(
                "누르기 실패 후 상승 완료. 반복 중단."
            ) from press_error

        after = (
            s.get_height_gpu()
            .detach()
            .clone()
        )

        difference = (
            after - before
        ).abs()

        print(
            f"[RESULT {index}] "
            f"achieved_reference_depth="
            f"{achieved_depth * 1000:.3f}mm "
            f"remaining_height_MAE="
            f"{float(difference.mean()) * 1000:.4f}mm "
            f"remaining_height_MAX="
            f"{float(difference.max()) * 1000:.3f}mm",
            flush=True,
        )

    print(
        "\n누르기 → 상승 3회 완료. "
        "점토 초기화·데이터 저장 없이 종료.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n사용자가 종료함.",
            flush=True,
        )