from dataclasses import dataclass
import numpy as np
import genesis as gs


# --------------------------------------------------
# Visual Sculpting Action
# --------------------------------------------------
@dataclass
class PushAction:
    x: float
    y: float
    theta: float
    length: float
    depth: float


def lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------
# Genesis
# --------------------------------------------------
gs.init(
    backend=gs.gpu,
    precision="32",
)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.45, -0.55, 0.40),
        camera_lookat=(0.0, 0.0, 0.04),
        camera_fov=35,
    ),
    show_viewer=True,
)

# 바닥
scene.add_entity(
    gs.morphs.Plane()
)

# --------------------------------------------------
# 현재는 rigid clay slab
# --------------------------------------------------
clay_height = 0.05

scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, clay_height / 2),
        size=(0.30, 0.30, clay_height),
        fixed=True,
    ),
    surface=gs.surfaces.Default(
        color=(0.72, 0.45, 0.28, 1.0),
    ),
)

# --------------------------------------------------
# Push tool
#
# 일단 단순한 작은 box tool 사용
# 나중에 실제 Visual Sculpting end-effector 형태로 맞춤
# --------------------------------------------------
tool_size = 0.025
tool_height = 0.08

tool = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.20),
        size=(tool_size, tool_size, tool_height),
        fixed=True,
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.2, 0.8, 1.0),
    ),
)

scene.build()


# --------------------------------------------------
# Visual Sculpting parameterized action
# --------------------------------------------------
action = PushAction(
    x=-0.08,
    y=-0.05,
    theta=np.deg2rad(35),
    length=0.12,
    depth=0.01,
)

# tool 중심 기준
surface_z = clay_height

# tool의 아래 끝이 clay surface에 닿으려면
# 중심은 surface_z + tool_height/2
contact_center_z = surface_z + tool_height / 2

# penetration depth만큼 더 아래로
push_center_z = contact_center_z - action.depth

# 안전 높이
hover_z = surface_z + 0.12


# 시작점
start_xy = np.array(
    [action.x, action.y],
    dtype=np.float32,
)

# 끝점
end_xy = np.array(
    [
        action.x + action.length * np.cos(action.theta),
        action.y + action.length * np.sin(action.theta),
    ],
    dtype=np.float32,
)

print("===== Visual Sculpting Action =====")
print("start:", start_xy)
print("end:", end_xy)
print("theta:", np.rad2deg(action.theta), "deg")
print("length:", action.length, "m")
print("depth:", action.depth, "m")


# --------------------------------------------------
# 1. 시작점 위로 이동
# --------------------------------------------------
tool.set_pos(
    np.array(
        [start_xy[0], start_xy[1], hover_z],
        dtype=np.float32,
    )
)

for _ in range(50):
    scene.step()


# --------------------------------------------------
# 2. 아래로 내려가기
# --------------------------------------------------
n_down = 80

for i in range(n_down):
    t = (i + 1) / n_down

    z = lerp(
        hover_z,
        push_center_z,
        t,
    )

    tool.set_pos(
        np.array(
            [
                start_xy[0],
                start_xy[1],
                z,
            ],
            dtype=np.float32,
        )
    )

    scene.step()


# --------------------------------------------------
# 3. theta 방향으로 linear push
# --------------------------------------------------
n_push = 150

for i in range(n_push):
    t = (i + 1) / n_push

    xy = start_xy + t * (end_xy - start_xy)

    tool.set_pos(
        np.array(
            [
                xy[0],
                xy[1],
                push_center_z,
            ],
            dtype=np.float32,
        )
    )

    scene.step()


# --------------------------------------------------
# 4. tool 올리기
# --------------------------------------------------
n_up = 80

for i in range(n_up):
    t = (i + 1) / n_up

    z = lerp(
        push_center_z,
        hover_z,
        t,
    )

    tool.set_pos(
        np.array(
            [
                end_xy[0],
                end_xy[1],
                z,
            ],
            dtype=np.float32,
        )
    )

    scene.step()


print("Push action execution complete.")
print("Ctrl+C로 종료")


# 결과 계속 보기
while True:
    scene.step()