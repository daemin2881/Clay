import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PushAction:
    x: float       # 시작 x 위치 [m]
    y: float       # 시작 y 위치 [m]
    theta: float   # push 방향 [rad]
    length: float  # push 길이 [m]
    depth: float   # 점토 안으로 누르는 깊이 [m]


def action_endpoint(action: PushAction):
    end_x = action.x + action.length * np.cos(action.theta)
    end_y = action.y + action.length * np.sin(action.theta)

    return end_x, end_y


# --------------------------------------------------
# 예시 Visual Sculpting action
# --------------------------------------------------
action = PushAction(
    x=-0.08,
    y=-0.05,
    theta=np.deg2rad(35),
    length=0.12,
    depth=0.01,
)

end_x, end_y = action_endpoint(action)

print("===== Push Action =====")
print(f"start  : ({action.x:.3f}, {action.y:.3f}) m")
print(f"theta  : {np.rad2deg(action.theta):.1f} deg")
print(f"length : {action.length:.3f} m")
print(f"depth  : {action.depth:.3f} m")
print(f"end    : ({end_x:.3f}, {end_y:.3f}) m")


# --------------------------------------------------
# 점토판 workspace
# 30 cm × 30 cm
# --------------------------------------------------
clay_half_size = 0.15

fig, ax = plt.subplots(figsize=(7, 7))

# 점토판 영역
ax.add_patch(
    plt.Rectangle(
        (-clay_half_size, -clay_half_size),
        0.30,
        0.30,
        fill=False,
        linewidth=2,
    )
)

# 시작점
ax.scatter(
    action.x,
    action.y,
    s=80,
    label="Start",
)

# push 방향
ax.arrow(
    action.x,
    action.y,
    end_x - action.x,
    end_y - action.y,
    width=0.002,
    length_includes_head=True,
)

# 끝점
ax.scatter(
    end_x,
    end_y,
    s=80,
    label="End",
)

ax.set_xlim(-0.18, 0.18)
ax.set_ylim(-0.18, 0.18)

ax.set_aspect("equal")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

ax.set_title(
    f"Visual Sculpting Push Action\n"
    f"theta={np.rad2deg(action.theta):.1f}°, "
    f"length={action.length:.3f} m, "
    f"depth={action.depth:.3f} m"
)

ax.grid(True)
ax.legend()

os.makedirs("outputs/action", exist_ok=True)

plt.savefig(
    "outputs/action/push_action.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()