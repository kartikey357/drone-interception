# PN Guidance Tuning Investigation — Maneuvering Evader Time-to-Intercept

**Context:** the PS's Performance Targets table requires "Time to interception versus
straight-line bound" to be < 3x (minimum) / < 1.5x (target). This metric was not
computed at all until this investigation — adding it revealed that while PN guidance
achieves 42% interception rate against the maneuvering evader (comfortably better than
every other approach tried: quintic planner 26%, DWA 23-27%, hybrid 27%), the
interceptions that DO succeed take far too long: **mean ratio 12.77x**, badly failing
even the minimum threshold.

## Root cause identified

Direct trace analysis of individual episodes (seed=1, maneuvering profile) showed a
clear overshoot-and-recover pattern:

```
t=1.02  miss=12.86m
t=2.02  miss=11.48m
t=3.02  miss=7.48m
t=4.02  miss=4.10m   <- nearly captured
t=5.02  miss=6.59m   <- diverging again
t=6.02  miss=12.94m
t=7.02  miss=17.80m  <- nearly lost entirely
t=8.02  miss=17.98m
t=9.02  miss=12.73m  <- re-converging
t=10.02 miss=2.83m
DONE INTERCEPTED at t=10.22s
```

The pursuer gets close (~4m) around t=4s, then swings back out to ~18m before a slower
second approach finally succeeds at t=10.22s — roughly doubling what should have been
the capture time. This is a known category of PN pathology: the lateral acceleration
term has a `1/distance` sensitivity (via `omega_los = (r × v_rel) / |r|²`) that can
become ill-conditioned very close to the target.

## Fixes attempted (all tested empirically, not just theorized)

| Attempt | Hypothesis | Result |
|---|---|---|
| Close-range lateral damping (blend lateral term to 0 below 2-8m) | Reduce ill-conditioned correction near intercept | **Made things worse at every setting** — rate dropped from 42% to as low as 15% as damping distance increased. The lateral term is apparently necessary right up to intercept, not harmful. |
| Higher constant N (1.0 → 4.0) | Faster, more decisive closure | N=4.0 genuinely cut mean capture time to 5-7s, but interception rate dropped to 28-36% — a real rate/time tradeoff, not a free improvement. |
| Adaptive N (high when far, blending to N=1.5 when close) | Get fast-closing benefit of high N without its instability near intercept | No configuration tested (blend distances 4/8/12m) clearly beat the plain N=1.5 baseline on both metrics simultaneously. |

## Current status (honest)

The original PN configuration (N=1.5, axial_speed_gain=3.0, no damping) remains the
best validated result: **42% interception rate, 12.77x time ratio**. Rate clears more
of the gap toward the 50% minimum than any other approach tried this project, but the
time-ratio requirement is not met, and none of the tactical parameter changes above
fixed it without a corresponding rate regression.

## Recommended path forward (future work / if more time available)

This is a genuine rate-vs-time tradeoff along a single tuning axis (N), which suggests
the fix isn't in this axis at all. More promising directions, not yet attempted:
- **Terminal guidance blending**: switch to a fundamentally different (not just
  differently-tuned) control law within some small radius of the target — e.g. direct
  velocity-matching or a short min-jerk terminal trajectory — rather than trying to make
  PN itself behave well at all ranges.
- **True Augmented Proportional Navigation (APN)**, which explicitly accounts for
  target acceleration in the guidance law (our IMM model already estimates this) rather
  than only LOS rate — this is the standard missile-guidance answer to intercepting
  accelerating/maneuvering targets and was not implemented here due to time constraints.
- **Model Predictive Control** with an explicit terminal cost on arrival velocity
  alignment, which could resolve the overshoot without the ad-hoc tuning tried above.

This entire investigation — the metric being missing, the root cause found via direct
trace analysis, and three principled-but-unsuccessful fix attempts — is exactly the
kind of evidence-based ablation/failure-analysis material the PS's report explicitly
asks for (20% of the grade), and is arguably stronger evidence of understanding than a
lucky fix would have been.
