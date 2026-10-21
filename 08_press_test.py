import os

import numpy as np
import trimesh
import genesis as gs


# ============================================================
# 0. Simulation parameters
# ============================================================

DT = 0.002
SUBSTEPS = 20

# 점토판 크기
CLAY_WIDTH = 0.30
CLAY_LENGTH = 0.30
CLAY_HEIGHT = 0.05

# 점토판 바닥 위치
CLAY_BOTTOM_Z = 0.01
CLAY_TOP_Z = CLAY_BOTTOM_Z + CLAY_HEIGHT

# 스틱 크기
TOOL_WIDTH = 0.04
TOOL_HEIGHT = 0.08

# 몇 m 깊이까지 누를 것인가
PRESS_DEPTH = 0.010       # 10 mm

# tool 시작 중심 높이
START_Z = 0.16


# ============================================================
# 1. Genesis initialization
# ============================================================

gs.init(
    backend=gs.gpu,
    precision="32",
)


# ============================================================
# 2. Scene
# ============================================================

scene = gs.Scene(

    sim_options=gs.options.SimOptions(
        dt=DT,
        substeps=SUBSTEPS,
        gravity=(0.0, 0.0, -9.81),
    ),

    # ToolEntity solver
    tool_options=gs.options.ToolOptions(),

    # 점토 MPM solver
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.25, -0.25, -0.10),
        upper_bound=(0.25, 0.25, 0.30),

        # 1 cm 정도의 deformation을 보기 위해
        # 처음보다 조금 촘촘하게 설정
        grid_density=128,
        particle_size=0.005,
    ),

    # rigid ground ↔ MPM coupling
    coupler_options=gs.options.LegacyCouplerOptions(
        rigid_mpm=True,
    ),

    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.40, -0.48, 0.32),
        camera_lookat=(0.0, 0.0, 0.05),
        camera_fov=35,
    ),

    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=False,
    ),

    show_viewer=True,
)


# ============================================================
# 3. Ground
# ============================================================

ground = scene.add_entity(

    material=gs.materials.Rigid(
        needs_coup=True,
        coup_friction=0.8,
    ),

    morph=gs.morphs.Plane(),
)


# ============================================================
# 4. Deformable clay slab
# ============================================================

clay = scene.add_entity(

    material=gs.materials.MPM.ElastoPlastic(

        # ★ 아직 실제 modelling clay 보정값 아님
        # press가 제대로 작동하는지 보기 위한 초기값
        E=2.0e5,
        nu=0.35,
        rho=1200.0,

        use_von_mises=True,
        von_mises_yield_stress=5.0e3,
    ),

    morph=gs.morphs.Box(

        pos=(
            0.0,
            0.0,
            CLAY_BOTTOM_Z + CLAY_HEIGHT / 2,
        ),

        size=(
            CLAY_WIDTH,
            CLAY_LENGTH,
            CLAY_HEIGHT,
        ),
    ),

    surface=gs.surfaces.Default(
        color=(0.72, 0.45, 0.28, 1.0),

        # 일단 particle 움직임이 보이게
        vis_mode="particle",
    ),
)


# ============================================================
# 5. Robot stick OBJ 생성
#
# Genesis ToolEntity는 primitive Box 대신
# Mesh를 사용하는 것이 필요함.
# ============================================================

os.makedirs(
    "assets/generated",
    exist_ok=True,
)

tool_mesh_path = os.path.abspath(
    "assets/generated/press_stick.obj"
)


# 매번 동일한 mesh를 만들어 저장
tool_mesh = trimesh.creation.box(
    extents=(
        TOOL_WIDTH,
        TOOL_WIDTH,
        TOOL_HEIGHT,
    )
)

tool_mesh.export(
    tool_mesh_path
)

print()
print("Tool mesh:")
print(tool_mesh_path)


# ============================================================
# 6. Robot stick = ToolEntity
#
# 핵심:
#
# Tool은 우리가 지정한 trajectory대로 움직임.
# 점토가 tool을 밀어내지는 않음.
#
# 대신 Tool collision 때문에 MPM 점토가 밀림.
# ============================================================

tool = scene.add_entity(

    material=gs.materials.Tool(
        collision=True,
        friction=0.5,

        # Tool ↔ MPM contact softness
        coup_softness=0.002,

        # Tool SDF resolution
        sdf_res=128,
    ),

    morph=gs.morphs.Mesh(
        file=tool_mesh_path,

        # Tool mesh는 내부 normalization이 있으므로
        # 최대 길이 = 0.08 m로 맞춤
        scale=TOOL_HEIGHT,

        pos=(
            0.0,
            0.0,
            START_Z,
        ),
    ),

    surface=gs.surfaces.Default(
        color=(0.10, 0.20, 0.90, 1.0),
    ),
)


# ============================================================
# 7. Build
# ============================================================

scene.build()


# ============================================================
# 8. Press geometry
# ============================================================

tool_half_height = TOOL_HEIGHT / 2


# tool 아래 끝이 clay top에 정확히 닿는 순간의
# tool center 높이
CONTACT_Z = (
    CLAY_TOP_Z
    + tool_half_height
)


# 원하는 penetration만큼 중심을 더 아래로 이동
TARGET_Z = (
    CONTACT_Z
    - PRESS_DEPTH
)


print()
print("========================================")
print("       TOOL -> CLAY PRESS TEST")
print("========================================")

print(
    f"Clay bottom       : "
    f"{CLAY_BOTTOM_Z:.4f} m"
)

print(
    f"Clay top          : "
    f"{CLAY_TOP_Z:.4f} m"
)

print(
    f"Tool width        : "
    f"{TOOL_WIDTH:.4f} m"
)

print(
    f"Tool height       : "
    f"{TOOL_HEIGHT:.4f} m"
)

print(
    f"Contact center z  : "
    f"{CONTACT_Z:.4f} m"
)

print(
    f"Press depth       : "
    f"{PRESS_DEPTH:.4f} m "
    f"({PRESS_DEPTH * 1000:.1f} mm)"
)

print(
    f"Target center z   : "
    f"{TARGET_Z:.4f} m"
)

print("========================================")
print()


# ============================================================
# 9. Tool command
#
# 중요:
#
# Genesis ToolEntity 내부에서는
#
# pos[i_b, k]
#
# 형태로 접근하기 때문에
# shape = (1, 3)을 사용해야 함.
# ============================================================

def command_tool(z, vz=0.0):

    position = np.array(
        [[
            0.0,
            0.0,
            float(z),
        ]],
        dtype=np.float32,
    )

    velocity = np.array(
        [[
            0.0,
            0.0,
            float(vz),
        ]],
        dtype=np.float32,
    )

    tool.set_position(
        position
    )

    tool.set_velocity(
        vel=velocity
    )


# ============================================================
# 10. 초기 점토 안정화
# ============================================================

print("0. Clay settling...")

command_tool(
    START_Z,
    0.0,
)

for _ in range(150):

    command_tool(
        START_Z,
        0.0,
    )

    scene.step()


# ============================================================
# 11. 잠시 시작 위치 유지
# ============================================================

print("1. Tool ready.")

for _ in range(100):

    command_tool(
        START_Z,
        0.0,
    )

    scene.step()


# ============================================================
# 12. Press
#
# START_Z
#     ↓
# CONTACT_Z
#     ↓
# TARGET_Z
#
# TARGET_Z까지 정확히 내려감.
# ============================================================

print()
print("2. Pressing clay...")

N_DOWN = 500


for i in range(N_DOWN):

    t0 = i / N_DOWN
    t1 = (i + 1) / N_DOWN


    z0 = (
        START_Z
        + t0 * (TARGET_Z - START_Z)
    )

    z1 = (
        START_Z
        + t1 * (TARGET_Z - START_Z)
    )


    # 현재 simulation step에서 필요한
    # tool z velocity
    vz = (
        z1 - z0
    ) / DT


    command_tool(
        z0,
        vz,
    )


    scene.step()


# 마지막에 정확한 목표 위치 고정
command_tool(
    TARGET_Z,
    0.0,
)


print()
print(
    f"Reached {PRESS_DEPTH * 1000:.1f} mm penetration."
)


# ============================================================
# 13. Hold
# ============================================================

print("3. Holding press...")


for _ in range(300):

    command_tool(
        TARGET_Z,
        0.0,
    )

    scene.step()


# ============================================================
# 14. Release
#
# TARGET_Z -> START_Z
# ============================================================

print("4. Releasing tool...")


N_UP = 500


for i in range(N_UP):

    t0 = i / N_UP
    t1 = (i + 1) / N_UP


    z0 = (
        TARGET_Z
        + t0 * (START_Z - TARGET_Z)
    )

    z1 = (
        TARGET_Z
        + t1 * (START_Z - TARGET_Z)
    )


    vz = (
        z1 - z0
    ) / DT


    command_tool(
        z0,
        vz,
    )


    scene.step()


# 정확한 시작 높이로 복귀
command_tool(
    START_Z,
    0.0,
)


# ============================================================
# 15. Tool을 뺀 뒤 점토 안정화
# ============================================================

print("5. Waiting for residual deformation...")


for _ in range(600):

    command_tool(
        START_Z,
        0.0,
    )

    scene.step()


# ============================================================
# 16. 완료
# ============================================================

print()
print("========================================")
print("PRESS COMPLETE")
print()
print(
    "확인할 것:"
)

print(
    "1. 스틱이 점토와 접촉했는가?"
)

print(
    "2. 스틱이 내려갈 때 점토 particle이 밀렸는가?"
)

print(
    f"3. {PRESS_DEPTH * 1000:.1f} mm press가 실제로 표현됐는가?"
)

print(
    "4. 스틱 제거 후 dent가 남았는가?"
)

print("========================================")
print()
print("Ctrl+C 로 종료")


# ============================================================
# 17. 결과 계속 보기
# ============================================================

while True:

    command_tool(
        START_Z,
        0.0,
    )

    scene.step()