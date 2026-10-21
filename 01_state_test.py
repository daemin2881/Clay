import os
import numpy as np
import genesis as gs


# --------------------------------------------------
# 1. Genesis 초기화
# --------------------------------------------------
gs.init(backend=gs.gpu)


# --------------------------------------------------
# 2. Scene 생성
# 아직 점토가 아님.
# depth pipeline 검증용 dummy object.
# --------------------------------------------------
scene = gs.Scene(
    show_viewer=True,
)


# 바닥
scene.add_entity(
    gs.morphs.Plane()
)


# 임시 물체
# 나중에 이 부분을 deformable clay로 교체할 예정
dummy_clay = scene.add_entity(
    gs.morphs.Box(
        pos=(0.0, 0.0, 0.05),
        size=(0.20, 0.20, 0.10),
    )
)


# --------------------------------------------------
# 3. Visual Sculpting 스타일 고정 top-down camera
# --------------------------------------------------
camera = scene.add_camera(
    res=(512, 512),
    pos=(0.0, 0.0, 0.8),
    lookat=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fov=30,
    GUI=False,
)


# --------------------------------------------------
# 4. Scene build
# --------------------------------------------------
scene.build()

# 몇 step 진행
for _ in range(10):
    scene.step()


# --------------------------------------------------
# 5. Depth render
# cam.render()는
# (rgb, depth, segmentation, normal)을 반환
# --------------------------------------------------
rgb, depth, segmentation, normal = camera.render(
    rgb=True,
    depth=True,
    segmentation=False,
    normal=False,
)

depth = np.asarray(depth, dtype=np.float32)

print("--------------------------------")
print("Depth shape :", depth.shape)
print("Depth dtype :", depth.dtype)
print("Depth min   :", np.nanmin(depth))
print("Depth max   :", np.nanmax(depth))
print("--------------------------------")


# --------------------------------------------------
# 6. Spatial gradient
# Visual Sculpting의 ∇D_t
# --------------------------------------------------
grad_y, grad_x = np.gradient(depth)

gradient_magnitude = np.sqrt(
    grad_x ** 2 + grad_y ** 2
)


print("Gradient X shape:", grad_x.shape)
print("Gradient Y shape:", grad_y.shape)
print("Gradient magnitude shape:", gradient_magnitude.shape)


# --------------------------------------------------
# 7. 저장
# --------------------------------------------------
os.makedirs("outputs/state_test", exist_ok=True)

np.save(
    "outputs/state_test/depth.npy",
    depth
)

np.save(
    "outputs/state_test/gradient_x.npy",
    grad_x
)

np.save(
    "outputs/state_test/gradient_y.npy",
    grad_y
)

np.save(
    "outputs/state_test/gradient_magnitude.npy",
    gradient_magnitude
)

# 한 파일에도 묶어서 저장
np.savez_compressed(
    "outputs/state_test/state.npz",
    depth=depth,
    grad_x=grad_x,
    grad_y=grad_y,
    grad_mag=gradient_magnitude,
)

print()
print("Visual Sculpting state 저장 완료")
print("outputs/state_test/state.npz")
