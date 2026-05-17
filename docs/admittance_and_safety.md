# Admittance Control and Safety on Velocity-Mode Hardware

## What the AdmittanceLayer is

`AdmittanceLayer` implements a **joint-space first-order admittance**:

```
M_v v_dot + D_v v = tau          (+ K_v (q - q_anchor)  if stiffness is set)
```

Discretised with forward Euler at each control tick:

```
v_state[k+1] = v_state[k] + dt * M_v^{-1} * (tau - D_v * v_state[k] - K_v * (q - q_anchor))
v_des        = v_state[k+1] + v_ff
v_safe       = direction-preserving rescale of v_des to |v_i| <= q_vel_max_i
```

`v_ff` is a **feedforward velocity** that bypasses the dynamics entirely. This is the critical difference from a naïve static `v = D^{-1} tau` map: the feedforward lets trajectory tracking react at full bandwidth, while the admittance state low-passes the error-driven torque component.

The outer-loop break frequency per joint is `omega_c_i = D_v_i / M_v_i`. `make_inertia_weighted_damping(model, data, q_anchor, cutoff)` sets `D_v_i = cutoff * M_v_i` so all joints share the same `omega_c = cutoff` regardless of their inertia.

## What it is not

`AdmittanceLayer` translates a torque command into a **joint velocity reference**. It is not a torque execution path. The torque `tau` sent to `apply()` is never directly applied to the robot's motors — it is only used as the forcing function of the admittance ODE.

In particular:

- The wrench-to-torque projection (`J_b^T * F`) and the bias-compensation policy inside `DynamicTaskControllerBase` are visible to the caller but are not mechanically coupled to what the hardware servo does. The inner velocity servo produces whatever torque is needed to track `v_safe` — this torque has no guaranteed relationship to `tau`.
- The `Lambda`-scaled feedforward term in `GeometricPDController` does not improve trajectory tracking once an admittance follows it, because the Lambda term scales the wrench, which is then divided out again by `D_v` in the admittance. Only `v_ff` (the DLS-IDK of `Ad_{g_e} * xi_d`) contributes trajectory-tracking feedforward to the velocity command.

## Why the torque-side ASIF does not transfer

`DynamicTaskControllerBase` with `constraint_aware = true` runs an ASIF (active-set invariance filter) QP that certifies `tau_safe` under the model `M(q) v_dot + h(q,v) = tau`. This guarantee is **voided** when an admittance follows and a velocity-mode interface is the actuator:

1. The torque box `|tau| <= tau_max` constrains a quantity (`tau_safe`) that never reaches the motors.
2. The dynamic HOCBFs predict future joint accelerations as `M^{-1}(tau_safe - h)`, but the real acceleration follows from the inner-servo tracking dynamics, not from `tau_safe` directly.
3. The collision HOCBF and position HOCBF rely on the same incorrect dynamics model.

Only the velocity barrier `|v| <= v_max` survives approximately, but it is redundant with the direction-preserving rescale already inside `AdmittanceLayer`.

**Recommendation:** set `constraint_aware = false` on the upstream torque controller when it is followed by an admittance onto a velocity-mode interface. Use `safe_velocity_projection` (see below) for real kinematic safety guarantees.

## Recommended hardware cascade

```
GeometricPDController (constraint_aware = false, bias_compensation = None)
    |
    | JointTorque
    v
AdmittanceLayer::apply(model, q, tau, v_ff, dt, vel)
    |  v_ff = DLS-IDK of Ad_{g_e} * xi_d  (trajectory feedforward)
    |
    | JointVelocity (v_des = v_state + v_ff, v_safe = rescaled)
    v
safe_velocity_projection(model, data, col_model, col_data, vel.v, v_projected, opts)
    |  hard joint-position + velocity limits
    |  soft self-collision avoidance (PositionBarrier + CollisionBarrier)
    |
    | JointVelocity (projected)
    v
hw.write(vel)
```

See `tests/experiments/exp_3a_sim_hw.cpp` Variant D for a hardware implementation of
this cascade.  Note: Exp 3A omits `safe_velocity_projection` from Variants C/D/E so
that the four-way comparison isolates interface and admittance effects without
confounding with safety-layer overhead.  The cascade above (including the projection
step) represents the recommended **production** configuration.
Variant E runs the mechanically identical cascade against the MuJoCo simulator and is
useful for tuning `--cutoff` / `--mass-scale` offline before deploying to hardware
(`pixi run exp-3a-sim-admittance`).

`bias_compensation = None` on both variants. A velocity-mode actuator's internal velocity-tracking servo opposes gravity through its feedback dynamics (any gravity-induced motion creates `v_actual != 0`, which the servo immediately corrects); the user must not add `g(q)` or `h(q,v)` in the controller. Adding either would inject non-zero torque at rest, drive the admittance state away from zero, and cause drift. This applies equally to the xArm SDK in mode 4 and to MuJoCo `<velocity>` actuators.

## Tuning guide

### Step 1: choose M_v

Use `make_inertia_diag(model, data, q_anchor)` to set `M_v = diag(M(q_anchor))`. This ties the admittance bandwidth to the robot's own reflected inertia so `omega_c` has consistent physical meaning across joints. Scale uniformly with `--mass-scale` for gain-sweep experiments.

### Step 2: choose the cutoff frequency

`D_v_i = cutoff * M_v_i`. The admittance acts as a per-joint low-pass on the error-driven torque with bandwidth `omega_c = cutoff`.

Practical constraints:
- Keep `cutoff < inner-servo bandwidth / 5` to ensure at least one decade of phase margin between the outer admittance loop and the inner SDK velocity servo. The xArm SDK servo bandwidth is not published but is typically 50–200 rad/s; `cutoff = 30 rad/s` (~5 Hz) is conservative.
- Keep `cutoff < outer Nyquist / 2 = pi / dt`. At 125 Hz, the Nyquist is 392 rad/s; this constraint is easily satisfied.

`make_inertia_weighted_damping(model, data, q_anchor, cutoff)` encodes this in one call.

### Step 3: set controller gains

Because `v_ff` bypasses the admittance, trajectory tracking accuracy depends primarily on the feedforward quality. The P/D gains only need to correct residual steady-state errors and disturbances. Start from the simulation-validated gain set (`kKpPos = 4000`, `kKpRot = 60`, `kKdLin = 280`, `kKdAng = 2.4`); lower gains if the admittance state oscillates.

Symptom mapping:

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Oscillation at any gain | `cutoff` too high relative to inner-servo BW | Lower `--cutoff` |
| Sluggish response, large phase lag | `cutoff` too low | Raise `--cutoff` |
| Persistent steady-state position error | `kp_pos` too low, or `v_ff` not reaching hardware | Check FF; raise `kp_pos` |
| Persistent orientation error | `kp_rot` too low, or rescale clipping orientation channel | Check `max_ratio` in diagnostics; raise `kp_rot` |
| Arm drifts at rest | `bias_compensation` is not `None` | Set `bias_compensation = None` |
| `safe_velocity_projection` repeatedly `RELAXED` | Robot near joint limit or collision zone | Reduce trajectory aggressiveness or add clearance |

### Step 4: inspect diagnostics

`AdmittanceLayer::last_tick_diagnostics()` exposes per-tick `v_state`, `v_ff`, `v_des`, `v_safe`, and `max_ratio` / `rescaled`. `fill_admittance_diagnostics` maps these to the `v_ctrl / v_des / v_safe` log columns.

Key checks:
- `max_ratio` consistently > 1: the admittance output is saturating every tick. Either lower gains, lower `cutoff`, or reduce trajectory speed.
- `v_ff` dominates `v_state`: expected during fast trajectories. If `v_state` is zero at steady state but there is a non-zero pose error, `kp_pos` / `kp_rot` may be too low.
- `optik_modified = true` frequently: the kinematic safety filter is reshaping the velocity, meaning the admittance output regularly violated a CBF. Inspect which constraint is active (position vs collision).

## When to use the optional virtual spring (K_v)

Set `AdmittanceOptions::stiffness_diag` and `q_anchor` for a second-order admittance with a virtual rest configuration:

```
M_v v_dot + D_v v + K_v (q - q_anchor) = tau
```

This makes the admittance asymptotically stable in joint space independently of the upstream torque controller. Useful for:
- Teach/jog modes where `tau = 0` and the arm should return to a rest pose.
- Null-space stabilisation when the EE task leaves a joint degree of freedom unconstrained.

For trajectory tracking, the spring opposes deliberate joint motion and is **not recommended**; leave `stiffness_diag` empty (1st-order admittance) so the admittance does not fight the feedforward.
