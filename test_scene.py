import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
)

scene.add_entity(
    gs.morphs.Plane()
)

scene.add_entity(
    gs.morphs.Box(
        pos=(0.0, 0.0, 0.05),
        size=(0.2, 0.2, 0.1),
    )
)

scene.build()

while True:
    scene.step()
