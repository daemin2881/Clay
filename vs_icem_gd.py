"""
Greedy-initialized iCEM with Adam refinement.

Based on:
Pinneri et al., Sample-efficient Cross-Entropy Method
for Real-time Planning (CoRL 2020).

Adapted for Visual Sculpting Neural Dynamics.
"""

import math
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


_PREVIOUS_ELITES = []


def reset_icem():
    global _PREVIOUS_ELITES
    _PREVIOUS_ELITES = []


def powerlaw_noise(shape, beta=1.0, device="cpu", dtype=torch.float32):
    """
    Power-law PSD Gaussian noise.

    shape = [horizon, action_dim]
    beta = 0: white noise
    beta = 1: pink noise
    beta = 2: brown noise
    """

    horizon, action_dim = shape

    if beta <= 0:
        return torch.randn(
            shape,
            device=device,
            dtype=dtype,
        )

    freqs = torch.fft.rfftfreq(
        horizon,
        device=device,
    )

    scale = torch.zeros_like(freqs)

    positive = freqs > 0

    scale[positive] = (
        freqs[positive] ** (-beta / 2.0)
    )

    real = torch.randn(
        len(freqs),
        action_dim,
        device=device,
        dtype=dtype,
    )

    imag = torch.randn_like(real)

    spectrum = torch.complex(real, imag)

    spectrum = spectrum * scale[:, None]

    spectrum[0] = 0

    if horizon % 2 == 0:
        spectrum[-1] = spectrum[-1].real + 0j

    noise = torch.fft.irfft(
        spectrum,
        n=horizon,
        dim=0,
    )

    noise = noise / noise.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)

    return noise


def icem_gd_refine_plan(
    model,
    state,
    target,
    initial_plan,
    *,
    w3d=1.0,
    wviz=1.0,
    recent_actions=None,
    refine_actions=5,
    samples=16,
    elite=4,
    iterations=3,
    gd_steps=3,
    rho=0.65,
    noise_beta=1.0,
    alpha=0.1,
    decay=1.5,
    reuse_fraction=0.5,
):
    global _PREVIOUS_ELITES

    if not (
        samples >= 2
        and 2 <= elite <= samples
        and iterations >= 1
        and gd_steps >= 0
        and 0 <= alpha <= 1
        and decay >= 1
        and 0 <= reuse_fraction <= 1
    ):
        raise ValueError("Invalid iCEM parameters")

    state = ensure_bchw(state)

    target = ensure_bchw(target).to(
        device=state.device,
        dtype=state.dtype,
    )

    base = clamp_actions_to_planning_range(
        initial_plan.detach().clone()
    )

    horizon = base.shape[0]

    n = min(
        max(0, int(refine_actions)),
        horizon,
    )

    if n == 0:
        return base

    history = prepare_recent_actions(
        recent_actions,
        state.device,
        state.dtype,
    )

    with torch.no_grad():
        b3, bv = compute_loss_baseline(
            state,
            target,
        )

    def evaluate(plan):
        with torch.no_grad():
            predicted = rollout(
                model,
                state,
                plan,
            )

            loss = joint_plan_loss(
                predicted,
                target,
                plan,
                b3,
                bv,
                w3d,
                wviz,
                history,
            )

            return float(loss.item())

    def feasible(plan):
        return clamp_actions_to_planning_range(
            plan.detach().clone()
        )

    def make_plan(raw):
        prefix = decode_actions(raw)

        plan = torch.cat(
            [prefix, base[n:]],
            dim=0,
        )

        return feasible(plan)

    best = base.clone()
    best_score = evaluate(best)

    mean = encode_actions(base[:n]).detach()

    std = torch.full_like(mean, 0.30)

    # Reuse elites from the previous MPC step.
    shifted_elites = []

    for previous in _PREVIOUS_ELITES:

        if previous.shape != base.shape:
            continue

        shifted = torch.cat(
            [
                previous[1:],
                base[-1:],
            ],
            dim=0,
        )

        shifted = feasible(
            shifted.to(
                device=base.device,
                dtype=base.dtype,
            )
        )

        shifted_elites.append(shifted)

    reused = []

    final_elites = []

    for iteration in range(iterations):

        # Official iCEM: decrease population
        # during optimization iterations.
        population = max(
            elite * 2,
            int(samples / (decay ** iteration)),
        )

        candidates = [
            (best_score, best.clone())
        ]

        # Shifted elites across MPC steps.
        if iteration == 0:

            reuse_count = int(
                len(shifted_elites) * reuse_fraction
            )

            for plan in shifted_elites[:reuse_count]:

                score = evaluate(plan)

                candidates.append(
                    (score, plan.clone())
                )

        # Reuse elites across CEM iterations.
        if iteration > 0:

            for plan in reused:

                score = evaluate(plan)

                candidates.append(
                    (score, plan.clone())
                )

        # Sample action sequences.
        for _ in range(population):

            noise = powerlaw_noise(
                mean.shape,
                beta=noise_beta,
                device=mean.device,
                dtype=mean.dtype,
            )

            raw = mean + std * noise

            plan = make_plan(raw)

            score = evaluate(plan)

            candidates.append(
                (score, plan.clone())
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        # Adam refinement of the best candidate.
        if gd_steps > 0:

            seed = candidates[0][1]

            improved = refine_plan_gpu(
                model,
                state,
                target,
                seed,
                w3d=w3d,
                wviz=wviz,
                recent_actions=recent_actions,
                max_refine_actions=n,
                refine_steps=gd_steps,
            )

            improved = feasible(improved)

            improved_score = evaluate(improved)

            candidates.append(
                (
                    improved_score,
                    improved.clone(),
                )
            )

            candidates.sort(
                key=lambda item: item[0]
            )

        if candidates[0][0] < best_score:

            best_score = candidates[0][0]

            best = candidates[0][1].clone()

        selected = candidates[:elite]

        final_elites = [
            plan.clone()
            for _, plan in selected
        ]

        reuse_count = max(
            1,
            int(elite * reuse_fraction),
        )

        reused = [
            plan.clone()
            for _, plan in selected[:reuse_count]
        ]

        elite_raw = torch.stack([
            encode_actions(plan[:n]).detach()
            for _, plan in selected
        ])

        new_mean = elite_raw.mean(dim=0)

        new_std = elite_raw.std(
            dim=0,
            unbiased=False,
        )

        # Official iCEM-style distribution smoothing.
        mean = (
            (1.0 - alpha) * new_mean
            + alpha * mean
        )

        std = (
            (1.0 - alpha) * new_std
            + alpha * std
        ).clamp(
            min=0.06,
            max=0.8,
        )

        print(
            f"[iCEM-GD {iteration + 1}/{iterations}] "
            f"population={population} "
            f"best={best_score:.6f}",
            flush=True,
        )

    # Save elite plans for the next MPC step.
    _PREVIOUS_ELITES = [
        plan.detach().clone()
        for plan in final_elites
    ]

    return best.detach()
