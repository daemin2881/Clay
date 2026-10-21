import genesis as gs

gs.init(
    backend=gs.gpu,
    precision="32",
)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.45, -0.55, 0.40),
        camera_lookat=(0.0, 0.0, 0.03),
        camera_fov=35,
    ),
    show_viewer=True,
)

# 바닥
scene.add_entity(
    morph=gs.morphs.Plane()
)

# 지금은 점토 "모양"만 만든다.
# 아직 deformable 아님.
clay_slab = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.025),
        size=(0.30, 0.30, 0.05),
    ),
    surface=gs.surfaces.Default(
        color=(0.72, 0.45, 0.28, 1.0),
    ),
)

scene.build()

print("Rigid clay slab created.")
print("Ctrl+C로 종료")

while True:
    scene.step()