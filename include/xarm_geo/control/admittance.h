#pragma once

#include <chrono>

#include <Eigen/Dense>

#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>

namespace xarm_geo {

    // --- Admittance Layer ---
    //
    // Stateful joint-space first-order admittance:
    //
    //     M_v * v_dot + D_v * v  = tau          (+ K_v * (q - q_anchor)  if stiffness set)
    //
    // Discretised with forward Euler:
    //
    //     v_state[k+1] = v_state[k] + dt * M_v^{-1} *
    //                     ( tau - D_v * v_state[k] - K_v * (q - q_anchor) )
    //
    // The outer-loop break frequency is  omega_c_i = D_v_i / M_v_i.
    // Use make_inertia_weighted_damping() to set D_v relative to M_v(q) for a
    // target omega_c; keep omega_c < inner-servo bandwidth / 5.
    //
    // A velocity feedforward v_ff bypasses the dynamics entirely:
    //
    //     v_des  = v_state[k+1] + v_ff
    //     v_safe = direction-preserving rescale of v_des to |v_i| <= q_vel_max_i
    //
    // v_ff should be the reference joint velocity derived from the desired
    // EE twist (e.g. DLS-IDK of Ad_{g_e} * xi_d) so that tracking of moving
    // targets does not lag through the low-pass.
    //
    // Call seed_from(v_meas) once before entering the control loop to avoid a
    // first-tick step when the arm is already moving.

    struct AdmittanceOptions {
        // Required: virtual joint-space mass (strictly positive, dof entries).
        Eigen::VectorXd mass_diag;

        // Required: virtual joint-space damping (strictly positive, dof entries).
        Eigen::VectorXd damping_diag;

        // Optional: virtual joint-space stiffness (empty -> 1st-order / no spring).
        // When set, q_anchor must be the same size.
        Eigen::VectorXd stiffness_diag;
        Eigen::VectorXd q_anchor;
    };

    class AdmittanceLayer {
    public:
        explicit AdmittanceLayer(int dof, const AdmittanceOptions &opts);

        // --- Lifecycle ---

        // Zero the admittance velocity state.
        void reset() noexcept;

        // Seed the admittance state from a measured joint velocity, avoiding a
        // first-tick discontinuity when the arm is already moving.
        void seed_from(Eigen::Ref<const Eigen::VectorXd> v) noexcept;

        // --- Application ---

        // One Euler step of the admittance ODE followed by direction-preserving
        // velocity-limit rescale. v_ff is added after the dynamics conversion
        // (bypasses the low-pass) and written into last_tick_diagnostics().
        void apply(const Model &model, Eigen::Ref<const Eigen::VectorXd> q,
                   const JointTorque &tau_in, Eigen::Ref<const Eigen::VectorXd> v_ff,
                   std::chrono::nanoseconds dt, JointVelocity &v_out) noexcept;

        // Convenience: zero feedforward.
        void apply(const Model &model, Eigen::Ref<const Eigen::VectorXd> q,
                   const JointTorque &tau_in, std::chrono::nanoseconds dt,
                   JointVelocity &v_out) noexcept;

        // --- Diagnostics ---

        struct TickDiagnostics {
            Eigen::VectorXd v_state;  // admittance state after this step
            Eigen::VectorXd v_ff;     // feedforward input
            Eigen::VectorXd v_des;    // v_state + v_ff, before rescale
            Eigen::VectorXd v_safe;   // post velocity-limit rescale (== v_out)
            double max_ratio = 0.0;   // max |v_des_i| / v_max_i; > 1 iff rescaled
            bool rescaled = false;
        };

        [[nodiscard]] auto last_tick_diagnostics() const noexcept -> const TickDiagnostics & {
            return diag_;
        }

    private:
        int dof_;
        Eigen::VectorXd m_inv_diag_;  // 1 / M_v
        Eigen::VectorXd d_diag_;      // D_v
        Eigen::VectorXd k_diag_;      // K_v (empty -> no spring)
        Eigen::VectorXd q_anchor_;    // rest configuration for spring
        Eigen::VectorXd v_state_;     // integrator state
        TickDiagnostics diag_;
    };

    // --- Inertia Helpers ---
    //
    // Build AdmittanceOptions.mass_diag / damping_diag from the robot's own
    // joint-space inertia, evaluated at a representative configuration.
    // These use compute_mass_matrix internally and write into data.M.

    // Returns the diagonal of M(q_anchor) + joint_armature.
    // Good default for mass_diag: the admittance then reflects the robot's own
    // reflected inertia and the cutoff is truly in rad/s relative to that.
    [[nodiscard]] auto make_inertia_diag(const Model &model, Data &data,
                                         Eigen::Ref<const Eigen::VectorXd> q_anchor)
        -> Eigen::VectorXd;

    // Returns D_i = cutoff_rad_s * M_ii(q_anchor), so that the per-joint
    // admittance break frequency omega_c_i = D_i / M_i == cutoff_rad_s.
    // Default 30 rad/s (~5 Hz) is conservative relative to both the 125 Hz
    // outer-loop Nyquist and typical inner-servo bandwidths.
    [[nodiscard]] auto make_inertia_weighted_damping(const Model &model, Data &data,
                                                     Eigen::Ref<const Eigen::VectorXd> q_anchor,
                                                     double cutoff_rad_s = 30.0) -> Eigen::VectorXd;

    // --- Safe Velocity Projection ---
    //
    // Projects a candidate joint velocity v_in to the closest v_safe satisfying
    // hard joint-position and joint-velocity limits, plus soft self/environment
    // collision avoidance and soft position CBFs.
    //
    // Internally wraps optimal_inverse_diff_kinematics (composable overload):
    //   - PostureTask anchored at q_target = data.q + v_in * dt (minimises
    //     ||dq - v_in * dt||^2 in joint space).
    //   - Hard PositionLimit + VelocityLimit constraints from model.limits.
    //   - Soft PositionBarrier + CollisionBarrier (alpha = asif_defaults::kBarrierAlpha0,
    //     activation = asif_defaults::kCollisionActivationDistance).
    //
    // Pre-conditions: compute_jacobians(model, data) for the current data.q.
    [[nodiscard]] auto
    safe_velocity_projection(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, Eigen::Ref<const Eigen::VectorXd> v_in,
                             Eigen::Ref<Eigen::VectorXd> v_safe,
                             const OptimalIKOptions &opts = OptimalIKOptions()) -> OptimalIKStatus;

}  // namespace xarm_geo
