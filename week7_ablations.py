"""
Week 7 deliverable: ablation studies.

1. Sensor noise sweep -- how does interception rate degrade as sensor
   noise increases? (non_maneuvering, where the signal is cleanest)
2. Replan-interval sweep -- how much does replanning frequency matter?
3. Model-based planner vs. Week 3 baseline, head-to-head, same seeds --
   an honest comparison, since the full suite showed the added
   complexity of the model-based planner does NOT clearly beat the
   simple baseline yet. This is real failure-mode material for the
   report, not something to hide.

Uses smaller episode counts (30) for speed; the primary results.json
(Week 7 main script) uses the full 100 required by the PS.
"""

import numpy as np
from interception_sim import FastInterceptionSim, IMMEvaderModel, EpisodeStatus, QuadrotorLimits, plan_intercept, ConstantAccelerationKF
from interception_sim.baseline_planner import intercept_point_aiming, boundary_repulsion

DT = 1.0 / 50.0
ARENA_SIZE = np.array([50.0, 50.0, 30.0])
KP, KV = 3.0, 2.5
N_ABLATION_EPISODES = 30


def run_model_based(seed, evader_profile, sensor_noise_std=0.3, replan_interval=0.3, max_steps=3000):
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=sensor_noise_std, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)
    model, plan = None, None
    last_plan_t = -np.inf
    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        ps = sim.get_pursuer_state()
        t = ps.t
        if model is None:
            if obs.valid and obs.position is not None:
                model = IMMEvaderModel(obs.position, measurement_noise_std=sensor_noise_std)
            result = sim.step(np.zeros(3), DT)
        else:
            model.predict(DT)
            if obs.valid and obs.position is not None:
                model.update(obs.position)
            if plan is None or (t - last_plan_t) >= replan_interval:
                plan = plan_intercept(ps.position, ps.velocity, ps.acceleration, model,
                                       limits.max_speed, t)
                last_plan_t = t
            target_pos, target_vel, target_acc = plan.evaluate(t - last_plan_t)
            accel = target_acc + KP * (target_pos - ps.position) + KV * (target_vel - ps.velocity)
            rep = boundary_repulsion(ps.position, ARENA_SIZE, margin=2.0,
                                      max_repulsion_accel=limits.max_horizontal_accel)
            if np.linalg.norm(rep) > 1e-6:
                accel = rep
            mag = np.linalg.norm(accel)
            if mag > limits.max_horizontal_accel:
                accel = accel * (limits.max_horizontal_accel / mag)
            result = sim.step(accel, DT)
        done, status = sim.is_done()
        if done:
            return status
    return EpisodeStatus.TIMEOUT


def run_baseline(seed, evader_profile, sensor_noise_std=0.3, max_steps=3000):
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=sensor_noise_std, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)
    kf = None
    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        ps = sim.get_pursuer_state()
        if kf is None:
            if obs.valid and obs.position is not None:
                kf = ConstantAccelerationKF(obs.position, measurement_noise_std=sensor_noise_std)
            result = sim.step(np.zeros(3), DT)
        else:
            kf.predict(DT)
            if obs.valid and obs.position is not None:
                kf.update(obs.position)
            accel = intercept_point_aiming(ps.position, ps.velocity, kf.get_position(), kf.get_velocity(),
                                            pursuer_speed=limits.max_speed, max_accel=limits.max_horizontal_accel)
            rep = boundary_repulsion(ps.position, ARENA_SIZE, margin=2.0,
                                      max_repulsion_accel=limits.max_horizontal_accel)
            if np.linalg.norm(rep) > 1e-6:
                accel = rep
            mag = np.linalg.norm(accel)
            if mag > limits.max_horizontal_accel:
                accel = accel * (limits.max_horizontal_accel / mag)
            result = sim.step(accel, DT)
        done, status = sim.is_done()
        if done:
            return status
    return EpisodeStatus.TIMEOUT


def rate(run_fn, n=N_ABLATION_EPISODES, **kwargs):
    hits = sum(1 for seed in range(n) if run_fn(seed, **kwargs) == EpisodeStatus.INTERCEPTED)
    return hits / n


def ablation_sensor_noise():
    print("\n### Ablation 1: sensor noise sweep (non_maneuvering, model-based planner) ###")
    for sigma in [0.05, 0.1, 0.3, 0.6, 1.0, 1.5]:
        r = rate(run_model_based, evader_profile="non_maneuvering", sensor_noise_std=sigma)
        print(f"  noise_std={sigma:4.2f}m  ->  interception rate: {100*r:.1f}%")


def ablation_replan_interval():
    print("\n### Ablation 2: replan interval sweep (maneuvering, model-based planner) ###")
    for interval in [0.1, 0.2, 0.3, 0.5, 1.0]:
        r = rate(run_model_based, evader_profile="maneuvering", replan_interval=interval)
        print(f"  replan_interval={interval:4.2f}s  ->  interception rate: {100*r:.1f}%")


def ablation_baseline_vs_model_based():
    print("\n### Ablation 3: model-based planner vs. Week 3 baseline (same seeds) ###")
    for profile in ["non_maneuvering", "maneuvering"]:
        r_baseline = rate(run_baseline, evader_profile=profile)
        r_model = rate(run_model_based, evader_profile=profile)
        print(f"  {profile:20s}  baseline: {100*r_baseline:.1f}%   model-based: {100*r_model:.1f}%")


if __name__ == "__main__":
    ablation_sensor_noise()
    ablation_replan_interval()
    ablation_baseline_vs_model_based()
