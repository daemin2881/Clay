import genesis as gs


gs.init(
    backend=gs.gpu,
    precision="32",
)


scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),

    mpm_options=gs.options.MPMOptions(
        # safety padding 때문에 아래쪽 여유를 둠
        lower_bound=(-0.4, -0.4, -0.1),
        upper_bound=(0.4, 0.4, 0.4),

        # 첫 테스트는 비교적 거칠게
        particle_size=0.01,
    ),

    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.45, -0.55, 0.35),
        camera_lookat=(0.0, 0.0, 0.03),
        camera_fov=35,
    ),

    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=False,
    ),

    show_viewer=True,
)


# 바닥
scene.add_entity(
    morph=gs.morphs.Plane()
)


# --------------------------------------------------
# deformable clay slab
# --------------------------------------------------
clay = scene.add_entity(

    material=gs.materials.MPM.ElastoPlastic(
        # 지금은 Genesis 기본값에 가까운 시작점
        E=1e6,
        nu=0.2,
        rho=1000.0,

        use_von_mises=True,
        von_mises_yield_stress=1e4,
    ),

    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.035),
        size=(0.30, 0.30, 0.05),
    ),

    surface=gs.surfaces.Default(
        color=(0.72, 0.45, 0.28, 1.0),

        # 처음엔 particle로 보는 게 디버깅에 편함
        vis_mode="particle",
    ),
)


scene.build()

print("Deformable clay slab created.")
print("Ctrl+C로 종료")


while True:
    scene.step()