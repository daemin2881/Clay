import os
import numpy as np
import genesis as gs

gs.init(
    backend=gs.gpu,
    precision="32",
)

scene = gs.Scene(
    show_viewer=True,
)

# 바닥
scene.add_entity(
    gs.morphs.Plane()
)

# 현재는 rigid clay slab
scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.025),
        size=(0.30, 0.30, 0.05),
    ),
    surface=gs.surfaces.Default(
        color=(0.72, 0.45, 0.28, 1.0),
    ),
)

# Visual Sculpting용 고정 top-down camera
camera = scene.add_camera(
    res=(512, 512),
    pos=(0.0, 0.0, 0.60),
    lookat=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fov=35,
    GUI=False,
)

scene.build()

# scene 안정화
for _ in range(10):
    scene.step()

# depth render
rgb, depth, segmentation, normal = camera.render(
    rgb=True,
    depth=True,
    segmentation=False,
    normal=False,
)

depth = np.asarray(depth, dtype=np.float32)

# spatial gradient
grad_y, grad_x = np.gradient(depth)

print("Depth shape:", depth.shape)
print("Grad X shape:", grad_x.shape)
print("Grad Y shape:", grad_y.shape)

print("Depth min:", np.nanmin(depth))
print("Depth max:", np.nanmax(depth))

# 저장
os.makedirs("outputs/state", exist_ok=True)

np.save("outputs/state/depth.npy", depth)
np.save("outputs/state/gradient_x.npy", grad_x)
np.save("outputs/state/gradient_y.npy", grad_y)

np.savez_compressed(
    "outputs/state/state.npz",
    depth=depth,
    grad_x=grad_x,
    grad_y=grad_y,
)

print("Visual Sculpting state 저장 완료")