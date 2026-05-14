# Authoring Controllers

## What This Library Is For

`xarm_geo` is a manipulator-specific geometric-controller scaffold for the
xArm family. The formalism is tangent-space Lie groups via `smooth`
(`SE(3)`, `SO(3)`, co-tangent spaces, `Ad`, `ad`), with `Eigen` for
joint-space linear algebra.

Two complementary things are published:

- A geometric vocabulary for writing controllers: Lie-group types, SE(3)
  error / gradient / transport helpers, passivity-friendly Coriolis
  bilinears, body-frame task primitives, CBF / HOCBF safety primitives.
- An opinionated execution scaffold: four controller base classes
  (kinematic / dynamic) x (task-space / joint-space), each handling size
  checks, kinematics refresh, bias-force compensation, and safety routing.

Comparison with other libraries:

  lie-group-controllers : P / PD on any `manif` group; no robot model.
  smooth_feedback       : PID / MPC / ASIF on any `smooth` group;
                          manipulator-agnostic.
  gafro                 : CGA primitives + Manipulator; no controller
                          scaffold -- users write everything themselves.
  dqrobotics            : DQ primitives + explicit controller hierarchy;
                          kinematic-only, no dynamic / torque-mode.
  xarm_geo (this)       : smooth-based primitives + four base classes +
                          CBF/HOCBF safety; kinematic AND dynamic; safety
                          as a first-class composable layer.

---

## Controller Categories

Controllers are organised by what enters the loop:

  Domain  : task-space  -- reference is (pose, twist, spatial_acc) in SE(3).
            joint-space -- reference is (q_ref, v_ref, a_ref) in R^dof.
  Output  : kinematic   -- JointVelocity (rad/s).
            dynamic     -- JointTorque   (N*m).

                  Kinematic (JointVelocity)      Dynamic (JointTorque)
  Task-space :    KinematicTaskControllerBase    DynamicTaskControllerBase
  Joint-space:    KinematicJointControllerBase   DynamicJointControllerBase

All four bases share the same template-method shape:

  update(model, data, ctx, out)
    1. Size checks.
    2. data.q <- ctx.fb.q.
    3. Kinematics: eager on task-space bases (compute_jacobians always
       runs); lazy on joint-space bases (runs on first kin.X() access).
    4. Construct KinematicsCache / DynamicsCache (dynamics always lazy).
    5. Hook: compute_command_*(...).        <-- this is what you write.
    6. Post-process:
         kinematic-task : (optimal_)inverse_diff_kinematics
         dynamic-task   : J_b^T * F, bias add, optional ASIF
         kinematic-joint: optional direction-preserving velocity rescale
         dynamic-joint  : bias add, optional ASIF
    7. Write `out` once.

### Hook-Side State Access

Inside a hook:

  data.q                           -- canonical config; always fresh.
  kin.body_jacobian(), kin.ee_pose(), ... -- kinematic outputs; lazy.
  dyn.M(), dyn.h(), dyn.g()              -- dynamic outputs; lazy.

Do not read data.body_jacobian / data.M / data.h / data.g directly inside
a hook -- use the cache accessors. They guarantee freshness and memoise
across the tick (hook + post-processing share the same cache instance).

### Bias-Force Compensation Policy (Dynamic Bases Only)

Set via the `bias_compensation` field:

  None        : out = tau_ctrl
                (IDA-PBC, port-Hamiltonian, explicit feedback linearisation)
  GravityOnly : out = tau_ctrl + g(q)
                (Bullo & Murray geometric PD, Slotine-Li adaptive, natural-PD)
  Full        : out = tau_ctrl + h(q, v)
                (Computed-Torque Control, Maithripala-style operational-space PD/PID)

This is a theory choice, not a numerical option. Each example controller
declares its recommended policy via `kRecommendedBiasCompensation`. The
bias term is computed lazily via dyn.g() / dyn.h() -- no separate flag
is needed.

---

## Geometric Vocabulary

The control law itself is composed from free functions and types in:

### xarm_geo/core/manifold.h -- Lie types and transport

  manifold::SE3, manifold::SO3 -- smooth::SE3d / SO3d wrappers with
    Twist, Wrench, Jacobian, SpatialInertia, Spline<D> aliases.
  CoTangent<G>  -- dual of the tangent space (wrenches, torques).
    Supports +, -, scalar multiplication, matrix multiplication on the
    left, dot() for the dual pairing.
  transport_tangent / transport_cotangent -- Ad / Ad^{-T} frame change.
  rpy_to_SO3, wrap_to_pi, wrap_to_range  -- small helpers.

### xarm_geo/control/feedback.h -- SE(3) feedback primitives

  se3_lie_group_gradient(g_e, kp_pos, kp_rot)
    Returns +nabla Phi(g_e) in the body frame (Bullo-Murray trace-function
    gradient). Almost-globally stable. See § Common Pitfalls for sign usage.
  se3_lie_algebra_gradient(g_e, kp_pos, kp_rot)
    Returns +nabla Phi_log (log-map gradient with right-Jacobian correction).
    Discontinuous at theta = pi; do not use for integral terms.
  se3_lie_group_gradient_wrench(g_e, kp_pos, kp_rot)
    Wrench-typed variant of se3_lie_group_gradient. Use in dynamic controllers
    where the gradient is composed with damping into a body-frame wrench.
  se3_lie_algebra_gradient_wrench(g_e, kp_pos, kp_rot)
    Wrench-typed variant of se3_lie_algebra_gradient. Same caveats apply
    (discontinuous at theta = pi; do not use for integral terms).
  se3_velocity_error(body_twist, g_e, target_twist_body)
    Body-frame xi_e = xi - Ad_{g_e} * xi_d.
  se3_transported_acc(g_e, xi_e, ad_xi_d, spatial_acc_body)
    Closed-form d/dt(Ad_{g_e} * xi_d) for inertial feedforward.
  SE3FeedbackGains  -- per-axis diagonal gain struct (kp_pos, kp_rot,
    kd_lin, kd_ang, ki_lin, ki_ang). Defaults are zero; set explicitly.
  GradientType      -- enum tag (LieGroup / LieAlgebra).

### xarm_geo/modelling/dynamics.h -- passivity-friendly bilinears

  compute_coriolis_times(model, data, v_a, v_b, out, g_fresh)
    out = C(q, v_a) * v_b via the symmetric Levi-Civita Christoffel form.
    Used by Bullo & Murray geometric PD and Slotine-Li adaptive laws.
  compute_mass_matrix, compute_bias_forces, compute_gravity_forces,
  inverse_dynamics, forward_dynamics -- free functions for concept-only
    controllers; in base subclasses, prefer dyn.M() / dyn.h() / dyn.g().

### xarm_geo/safety/{tasks,constraints,barriers}.h -- safety composition

  Task     : FrameTask, PostureTask, TwistTask
             Soft objectives in the Optimal-IDK QP cost.
  Constraint: VelocityLimit, PositionLimit
             Hard linear inequalities on dq.
  KinematicBarrier: PositionBarrier, CollisionBarrier
             CBF inequalities G dq <= b for the velocity-level QP.
  DynamicBarrier: DynPositionBarrier, DynVelocityBarrier, DynCollisionBarrier
             CBF / HOCBF inequalities A tau <= b for the ASIF QP.

The bases install opinionated default safety sets when constraint_aware =
true. To install a custom set, use the concept-only pattern (§ Authoring
patterns).

### xarm_geo/control/monitor.h -- convergence monitoring

  ConvergenceMonitor -- counts consecutive below-threshold error-norm ticks.
    Fields: threshold (default 1e-3), min_consecutive (default 50).
    Methods: update(error_norm), reset(), converged().

---

## Worked Example: GeometricPDController

File: include/xarm_geo/examples/controllers/geometric_pd_controller.h

Control law (Bullo-Murray 1999 / Maithripala 2006, body-frame PD + FF):

  F_task = nabla Phi(g_e) - K_d * xi_e              [F_PD, wrench-direct]
         + Lambda(q) * d/dt(Ad_{g_e} * xi_d)         [F_FF, if enabled]

se3_*_gradient_wrench returns +nabla Phi as a body-frame Wrench (see
§ Common Pitfalls for sign usage). Lambda(q) is computed via
compute_op_space_inertia with DLS regularisation.
d/dt(Ad_{g_e} * xi_d) is provided by se3_transported_acc (includes the
ad-coupling term -ad_{xi_e}(Ad xi_d) so the FF residual vanishes at
perfect tracking).

The hook:

```cpp
auto compute_command_wrench(const Model & /*model*/, Data & /*data*/,
                            KinematicsCache &kin, DynamicsCache &dyn,
                            const TaskControllerContext &ctx,
                            manifold::SE3::Wrench &cmd_wrench) noexcept
    -> bool override {

    const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
    const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
    const manifold::SE3::Wrench grad =
        (gradient == GradientType::LieAlgebra)
            ? se3_lie_algebra_gradient_wrench(g_e, gains.kp_pos, gains.kp_rot)
            : se3_lie_group_gradient_wrench(g_e, gains.kp_pos, gains.kp_rot);
    ad_xi_d_ = g_e.Ad() * ctx.ref.twist;
    xi_e_    = body_twist - ad_xi_d_;

    // F_PD = nabla Phi(g_e) - K_d * xi_e
    cmd_wrench.head<3>().noalias() =
        grad.head<3>() - gains.kd_lin.cwiseProduct(xi_e_.head<3>());
    cmd_wrench.tail<3>().noalias() =
        grad.tail<3>() - gains.kd_ang.cwiseProduct(xi_e_.tail<3>());

    // F_FF = Lambda(q) * d/dt(Ad_{g_e} * xi_d)
    if (use_feedforward) {
        M_llt_.compute(dyn.M());
        if (compute_op_space_inertia(M_llt_, kin.body_jacobian(), lambda_,
                                     M_inv_Jt_, lambda_damping)) {
            d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_,
                                             ctx.ref.spatial_acc);
            cmd_wrench += manifold::SE3::Wrench(lambda_ * d_ad_xi_d_);
        }
    }
    return true;
}
```

Key points:
- All kinematic / dynamic reads go through kin / dyn. No direct data.M etc.
- kin and dyn are shared with the base post-processing pipeline -- if
  bias_compensation == Full, the base's dyn.h() call reuses the cached M.
- All scratch (M_llt_, M_inv_Jt_, lambda_, ...) is pre-allocated as
  members. Zero allocation per tick.
- Body frame throughout: g_e = g^{-1} * g_d, xi_e in body frame,
  wrench in body frame, base projects via J_b^T * cmd_wrench.
- compute_op_space_inertia uses Cholesky + DLS fallback; silently skips
  FF if M(q) is not positive definite.

---

## Authoring Patterns

In escalating order of how much of the base pipeline you customise.

### 1. Subclass a Base

The common case. Fits the 2x2 grid; uses the default safety routing
(or no safety). Override one virtual hook; the base does the rest.

Example controllers and their bases:

  GeometricPController  : KinematicTaskControllerBase  (n/a)
  GeometricPIController : KinematicTaskControllerBase  (n/a)
  GeometricPDController : DynamicTaskControllerBase    (Full)
  GeometricPIDController: DynamicTaskControllerBase    (Full)
  JointPController      : KinematicJointControllerBase (n/a)
  JointPDController     : DynamicJointControllerBase   (GravityOnly)
  EuclideanPController  : KinematicTaskControllerBase  (n/a)   [baseline]
  EuclideanPDController : DynamicTaskControllerBase    (Full)  [baseline]

The Euclidean* controllers are deliberately non-geometric textbook
baselines (world-frame position error + ZYX-Euler orientation error,
naive feedforward, no Ad / ad^* transport). They share the same public
surface (gains, use_feedforward, constraint_aware) as their geometric
counterparts so that A/B comparisons isolate the control law. NOT
recommended for production use.

To start: copy the closest example, rename, modify compute_command_*,
set kRecommendedBiasCompensation if applicable, add the appropriate
static_assert(...Controller<YourClass>).

### 2. Subclass a Base + Call Modelling Free Functions

When the hook needs something the cache does not expose: Coriolis
bilinear, forward dynamics, custom collision queries, IK random restarts.
Call the free function directly; the cache and the function share data.

JointPDController (Coriolis feedforward):

```cpp
if (use_coriolis_ff) {
    (void)dyn.g();   // ensure data.g is fresh before passing g_fresh=true
    compute_coriolis_times(model, data, ctx.fb.v, ctx.ref.v,
                           C_qdot_rdot_, /*g_fresh=*/true);
    tau_ctrl.tau.noalias() += C_qdot_rdot_;
}
```

### 3. Concept-Only -- Custom Safety Routing or Fully Custom Pipeline

When the default safety set (Optimal-IDK or ASIF with default barriers)
is not appropriate. Do NOT subclass. Write a class that satisfies the
relevant *Controller concept directly:

```cpp
auto update(const Model &m, Data &d, const TaskControllerContext &ctx,
            JointVelocity &out) noexcept -> ControllerStatus;
```

The class is accepted by any code taking KinematicTaskController<T>.
You own the full pipeline: size checks, compute_jacobians, control law,
safety routing, status mapping via to_controller_status().

Canonical example: PostureBiasedPController
  (include/xarm_geo/examples/controllers/custom_controller.h)
  -- geometric P law with an augmented Optimal-IDK task set (TwistTask
     + PostureTask + VelocityLimit + PositionLimit + CollisionBarrier).

### 4. Concept-Only with Mode State -- Hybrid / Synergistic Controllers

For controllers that are not of "one control law per tick" form:
synergistic potentials (Mayhew-Sanfelice-Teel), hysteresis-based global
SO(3) stabilisation, mode-switched MPC, Slotine-Li adaptive, etc.

Use the concept-only pattern, with mode / parameter state as private
members:

```cpp
auto update(const Model &m, Data &d, const TaskControllerContext &ctx,
            JointVelocity &out) noexcept -> ControllerStatus {
    d.q = ctx.fb.q;
    compute_jacobians(m, d);

    update_mode_state(d, ctx);           // switch / update internal mode
    cmd_twist = compute_in_mode(d, ctx); // mode-specific control law

    // ... safety routing, write out, return status.
}
```

No dedicated base is currently provided for hybrid controllers; the
right structure is too problem-specific. If the same shape recurs
across several controllers, that is a signal to factor a fifth base.

---

## Setpoint Regulation

Any tracking controller becomes a setpoint regulator when fed a constant
target. Use `TaskSetpointTrajectory` (or `JointSetpointTrajectory`) from
`xarm_geo/trajectory/trajectory.h`:

```cpp
// Build the constant target.
xarm_geo::TaskSetpointTrajectory setpoint;
setpoint.pose = /* desired SE(3) pose */;

// Instantiate a tracking controller as usual.
GeometricPDController ctrl(model);
ctrl.gains.kp_pos.setConstant(10.0);
ctrl.gains.kp_rot.setConstant(5.0);
ctrl.gains.kd_lin.setConstant(4.0);
ctrl.gains.kd_ang.setConstant(2.0);
ctrl.bias_compensation = BiasCompensation::Full;

// Control loop (1 kHz example).
JointTorque out(model.dof);
TaskTarget ts_target;
while (running) {
    setpoint.evaluate(t, ts_target);   // twist and spatial_acc are zero

    TaskControllerContext ctx{fb, ts_target, dt};
    ctrl.update(model, data, ctx, out);
    interface.send(out);
}
```

For convergence detection, add a ConvergenceMonitor (see § Convergence
monitoring below).

Note: for PI / PID controllers, call reset() when changing the
regulation target to clear integrator state from the previous setpoint.

---

## Convergence Monitoring

Controllers that want to expose a "have we converged?" predicate hold a
ConvergenceMonitor member and satisfy ConvergenceObservable:

```cpp
class MySetpointController : public DynamicTaskControllerBase {
    ConvergenceMonitor monitor_;

protected:
    auto compute_command_wrench(...) noexcept -> bool override {
        // ... compute control law ...
        const double err = grad.norm() + xi_e_.norm();
        monitor_.update(err);
        // ...
        return true;
    }

public:
    [[nodiscard]] auto converged() const noexcept -> bool {
        return monitor_.converged();
    }
    void reset() noexcept {
        e_I_.setZero();
        monitor_.reset();
    }
};

static_assert(xarm_geo::ConvergenceObservable<MySetpointController>);
static_assert(xarm_geo::ResettableController<MySetpointController>);
```

ConvergenceMonitor fields (both public):
  threshold       : error norm below which a tick counts (default 1e-3).
  min_consecutive : consecutive below-threshold ticks to declare
                    converged (default 50).

Use cases:
  Setpoint regulation : monitor grad.norm() + xi_e_.norm() (or
                        log(g_e).norm() for a purely geometric metric).
  Tracking settling   : monitor xi_e_.norm() once a trajectory ends.

Note: converged() is a settling-time heuristic, not a stability proof.
A controller can satisfy ConvergenceObservable without being globally
stable. Do not use it as evidence of Lyapunov stability.

---

## Common Pitfalls

### Reading data.M / data.h / data.g Directly in a Hook

Use dyn.M() / dyn.h() / dyn.g(). Direct data.X reads inside a hook
bypass the cache and may return stale values.

### Forgetting TwistTask::dt

If dt = 0 (the default), the QP residual collapses to zero and the
controller silently does nothing. Always set twist_task.dt before
calling optimal_inverse_diff_kinematics.

### Mixing Frame Conventions

The library is body-frame throughout: J_b, xi in body frame, g_e =
g^{-1} * g_d, wrenches in body frame, J_b^T for joint-torque projection.
Mixing in space-frame quantities compiles but produces wrong commands.

### Mutating data.q Inside a Hook

data.q is canonical; the base sets it from ctx.fb.q before the hook.
Writing to it mid-hook corrupts the cache and any subsequent free-
function calls. Use a local VectorXd for hypothetical configurations.

### Subtracting Instead of Adding the Gradient

Both `se3_lie_group_gradient` and `se3_lie_algebra_gradient` return `+nabla Phi`
in the body frame. Under the left-error convention `g_e = g^{-1} g_d`,
body-frame descent is achieved by **adding** the returned value:

```
cmd_twist  = ad_xi_d + grad        (kinematic P law)
cmd_wrench = grad - K_d * xi_e     (dynamic PD law)
```

Writing `ad_xi_d - grad` or `-grad - K_d * xi_e` flips the proportional
term into a repulsive force: the controller pushes the EE away from the
setpoint and any small perturbation grows without bound. The naive
"subtract the gradient" reflex from Euclidean gradient descent is wrong
here; see `feedback.h` for the chain-rule derivation through `g_e`.

### Carrying Integrator State Across Trajectories

Provide reset() and satisfy ResettableController so that harnesses can
zero the integrator between trajectories. For setpoint controllers, also
call reset() when the target changes.

### Integrating the Log-Map Gradient Near theta = pi

se3_lie_algebra_gradient is discontinuous at theta = pi. Integrating it
accumulates branch-cut jumps. Use se3_lie_group_gradient for PI / PID
integral terms.

---

## Extending Beyond SE(3)

The task-space bases bake SE(3) into the hook type signatures. To track
on a product space (SE(3) x R^n for joint-augmented control, SE(3) x
SE(3) for bimanual, T*SE(3) for IDA-PBC), use the concept-only pattern
(§ Authoring patterns, pattern 3 or 4). smooth natively supports
Bundle<> types; manifold::CoTangent<G> generalises to any LieGroup.

No BundleTaskControllerBase is currently provided. If the same bundle-
typed structure recurs, factor it out.
