"""Greedy-initialized CEM with Adam refinement."""

import torch

from vs_planner import (
    ensure_bchw,
    clamp_actions_to_planning_range,
    encode_actions,
    decode_actions,
    rollout,
    joint_plan_loss,
    compute_loss_baseline,
    prepare_recent_actions,
    refine_plan_gpu,
)


def cem_gd_refine_plan(
    model,
    state,
    target,
    initial_plan,
    *,
    w3d=1.0,
    wviz=1.0,
    recent_actions=None,
    refine_actions=2,
    samples=16,
    elite=4,
    iterations=2,
    gd_steps=2,
):
    if not (samples >= 2 and 1 <= elite <= samples):
        raise ValueError("CEM samples/elite 설정 오류")
    if iterations < 1 or gd_steps < 0:
        raise ValueError("CEM 반복 또는 GD 설정 오류")

    state = ensure_bchw(state)
    target = ensure_bchw(target).to(
        device=state.device,
        dtype=state.dtype,
    )

    base = clamp_actions_to_planning_range(
        initial_plan.detach().clone()
    )

    n = min(max(0, int(refine_actions)), base.shape[0])

    if n == 0:
        return base

    history = prepare_recent_actions(
        recent_actions,
        state.device,
        state.dtype,
    )

    with torch.no_grad():
        b3, bv = compute_loss_baseline(state, target)

    def evaluate(plan):
        with torch.no_grad():
            prediction = rollout(model, state, plan)

            loss = joint_plan_loss(
                prediction,
                target,
                plan,
                b3,
                bv,
                w3d,
                wviz,
                history,
            )

            return float(loss.item())

    best = base.clone()
    best_score = evaluate(best)

    mean = encode_actions(base[:n]).detach()
    std = torch.full_like(mean, 0.30)

    for iteration in range(iterations):

        candidates = [(best_score, best.clone())]

        for _ in range(samples):

            with torch.no_grad():
                raw = mean + torch.randn_like(mean) * std

                prefix = clamp_actions_to_planning_range(
                    decode_actions(raw),
                    repair_endpoints=False,
                )

                plan = torch.cat(
                    [prefix, base[n:]],
                    dim=0,
                )

                score = evaluate(plan)

                candidates.append(
                    (score, plan.detach().clone())
                )

        candidates.sort(key=lambda item: item[0])

        elites = candidates[:elite]

        # CEM 우수 후보를 Adam으로 개선
        if gd_steps > 0:
            seed_plan = elites[0][1]

            improved = refine_plan_gpu(
                model,
                state,
                target,
                seed_plan,
                w3d=w3d,
                wviz=wviz,
                recent_actions=recent_actions,
                max_refine_actions=n,
                refine_steps=gd_steps,
            )

            improved_score = evaluate(improved)

            candidates.append(
                (improved_score, improved.detach().clone())
            )

            candidates.sort(key=lambda item: item[0])
            elites = candidates[:elite]

        if candidates[0][0] < best_score:
            best_score = candidates[0][0]
            best = candidates[0][1].detach().clone()

        # 개선된 우수 후보를 이용해 CEM 분포 갱신
        elite_raw = torch.stack([
            encode_actions(p[:n]).detach()
            for _, p in elites
        ])

        mean = elite_raw.mean(dim=0)

        std = elite_raw.std(
            dim=0,
            unbiased=False,
        ).clamp(min=0.06, max=0.8)

        print(
            f"[CEM-GD {iteration + 1}/{iterations}] "
            f"best={best_score:.6f}"
        )

    return best.detach()
