#pragma once

#include <limits>

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {

    // --- Control Barrier Functions (CBFs) ---
    //
    // Forward invariance of a safe set S = { x : h(x) >= 0 } is enforced via
    // the continuous-time CBF condition  ḣ + alpha * h >= 0  (or HOCBF
    //  ḧ + alpha_1 * ḣ + alpha_0 * h >= 0  for relative-degree-2 cases).
    //
    // Two parallel hierarchies are provided, one per QP decision variable:
    //   - KinematicBarrier (variable: dq) -- used by the velocity-level
    //     Optimal Inverse Differential Kinematics solver.
    //   - DynamicBarrier (variable: tau) -- used by the joint-torque ASIF
    //     filter.

    // --- KinematicBarrier (decision variable: dq) ---
    //
    // Emits a one-sided inequality block in the velocity-level QP:
    //
    //     G_b * dq <= b_b   with   G_b = -J_h / dt,   b_b = alpha * (h - margin)
    //
    // Subclasses fill G (rows() x dof) and b (rows()) in compute().

    struct KinematicBarrier {
        virtual ~KinematicBarrier() = default;

        [[nodiscard]] virtual auto rows() const noexcept -> int = 0;

        // `col_model` / `col_data` may be nullptr for barriers that do not
        // need collision data. `data` is non-const so subclasses may write to
        // scratch buffers (e.g. data.collision.point_jacobian_*); canonical
        // state fields must not be mutated.
        virtual void compute(const Model &model, Data &data, const CollisionModel *col_model,
                             const CollisionData *col_data, Eigen::Ref<Eigen::MatrixXd> G,
                             Eigen::Ref<Eigen::VectorXd> b) const = 0;

        double alpha = 1.0;   // Class-K gain
        double margin = 0.0;  // Tightens the inequality
        double dt = 1.0;      // Discretisation step size
    };

    // --- PositionBarrier (Soft Joint Limits) ---
    //
    // For each joint i:
    //     h_upper_i(q) = q_max_i - q_i
    //     h_lower_i(q) = q_i - q_min_i
    //
    // Soft alternative to the hard PositionLimit constraint -- yields a
    // graceful slow-down near limits rather than an abrupt clamp.

    struct PositionBarrier final : public KinematicBarrier {
        explicit PositionBarrier(const Model &model) : dof(model.dof) {}

        const int dof;

        [[nodiscard]] auto rows() const noexcept -> int override { return 2 * dof; }

        void compute(const Model &model, Data &data, const CollisionModel *col_model,
                     const CollisionData *col_data, Eigen::Ref<Eigen::MatrixXd> G,
                     Eigen::Ref<Eigen::VectorXd> b) const override;
    };

    // --- CollisionBarrier (Self / Environment Distance) ---
    //
    // For each collision pair k:
    //     h_k(q) = d_k(q) - activation_distance
    //
    // Requires update_geometry_poses + compute_min_distance to have been
    // called for the current data.q (the composable Optimal IDK solver runs
    // these prerequisites automatically when collision data is present).

    struct CollisionBarrier final : public KinematicBarrier {
        CollisionBarrier(const Model & /*model*/, const CollisionModel &col_model,
                         double activation_distance_)
            : num_pairs(static_cast<int>(col_model.collision_pairs.size())),
              activation_distance(activation_distance_) {}

        const int num_pairs;
        double activation_distance;  // d_safe (m); barrier active when d_k < this

        [[nodiscard]] auto rows() const noexcept -> int override { return num_pairs; }

        void compute(const Model &model, Data &data, const CollisionModel *col_model,
                     const CollisionData *col_data, Eigen::Ref<Eigen::MatrixXd> G,
                     Eigen::Ref<Eigen::VectorXd> b) const override;
    };

    // --- DynamicBarrier (decision variable: tau) ---
    //
    // Expresses the CBF condition in the joint-torque variable. The chain
    // rule  v_dot = M^-1 (tau - h_bias)  expands the CBF / HOCBF condition
    // into a linear inequality A * tau <= b.
    //
    // Pre-conditions (caller responsibility, normally satisfied by asif_filter):
    //   - data.q is the current joint configuration.
    //   - compute_jacobians(model, data) has been called.
    //   - data.M and data.h are populated (compute_mass_matrix +
    //     compute_bias_forces).
    //   - data.asif.M_inv and data.asif.M_inv_h are populated (asif_filter
    //     does this once at the top of each call).
    //   - For barriers reading collision data, update_geometry_poses and
    //     compute_min_distance must have been called first.

    struct DynamicBarrier {
        virtual ~DynamicBarrier() = default;

        [[nodiscard]] virtual auto rows() const noexcept -> int = 0;

        // Fill A (rows x dof) and b (rows) with the CBF inequality
        // A * tau <= b. `v` is the current joint velocity.
        virtual void compute_torque_constraint(const Model &model, Data &data,
                                               const CollisionModel *col_model,
                                               const CollisionData *col_data,
                                               const Eigen::Ref<const Eigen::VectorXd> &v,
                                               Eigen::Ref<Eigen::MatrixXd> A,
                                               Eigen::Ref<Eigen::VectorXd> b) const = 0;

        // Optional post-solve evaluation at a candidate (q, v). Returns the
        // minimum h(q, v) over this barrier's rows; +infinity means
        // "trivially safe / not implemented". Used by asif_validate(). Caller
        // refreshes collision queries at the candidate q where applicable.
        [[nodiscard]] virtual auto
        evaluate_at(const Model & /*model*/, const Data & /*data*/,
                    const CollisionModel * /*col_model*/, const CollisionData * /*col_data*/,
                    const Eigen::Ref<const Eigen::VectorXd> & /*q*/,
                    const Eigen::Ref<const Eigen::VectorXd> & /*v*/) const -> double {
            return std::numeric_limits<double>::infinity();
        }

        double alpha_0 = 1.0;  // CBF gain (also used for relative-degree-1)
        double alpha_1 = 1.0;  // HOCBF secondary gain (relative-degree-2 only)
        double margin = 0.0;   // Tightens the inequality
    };

    // --- DynPositionBarrier (HOCBF; relative degree 2 in tau) ---
    //
    // For each joint i:
    //     h_upper_i(q) = q_max_i - q_i
    //     h_lower_i(q) = q_i - q_min_i

    struct DynPositionBarrier final : public DynamicBarrier {
        explicit DynPositionBarrier(const Model &model) : dof(model.dof) {}

        const int dof;

        [[nodiscard]] auto rows() const noexcept -> int override { return 2 * dof; }

        void compute_torque_constraint(const Model &model, Data &data,
                                       const CollisionModel *col_model,
                                       const CollisionData *col_data,
                                       const Eigen::Ref<const Eigen::VectorXd> &v,
                                       Eigen::Ref<Eigen::MatrixXd> A,
                                       Eigen::Ref<Eigen::VectorXd> b) const override;

        [[nodiscard]] auto
        evaluate_at(const Model &model, const Data &data, const CollisionModel *col_model,
                    const CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &q,
                    const Eigen::Ref<const Eigen::VectorXd> &v) const -> double override;
    };

    // --- DynVelocityBarrier (CBF; relative degree 1 in tau) ---
    //
    // For each joint i:
    //     h_upper_i(v) = v_max_i - v_i
    //     h_lower_i(v) = v_i + v_max_i        (i.e. v_i >= -v_max_i)

    struct DynVelocityBarrier final : public DynamicBarrier {
        explicit DynVelocityBarrier(const Model &model) : dof(model.dof) {}

        const int dof;
        Eigen::VectorXd v_max;  // Optional override (empty -> use model.limits[i].q_vel_max)

        [[nodiscard]] auto rows() const noexcept -> int override { return 2 * dof; }

        void compute_torque_constraint(const Model &model, Data &data,
                                       const CollisionModel *col_model,
                                       const CollisionData *col_data,
                                       const Eigen::Ref<const Eigen::VectorXd> &v,
                                       Eigen::Ref<Eigen::MatrixXd> A,
                                       Eigen::Ref<Eigen::VectorXd> b) const override;

        [[nodiscard]] auto
        evaluate_at(const Model &model, const Data &data, const CollisionModel *col_model,
                    const CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &q,
                    const Eigen::Ref<const Eigen::VectorXd> &v) const -> double override;
    };

    // --- DynCollisionBarrier (HOCBF; relative degree 2 in tau) ---
    //
    // For each collision pair k:
    //     h_k(q) = d_k(q) - activation_distance
    //
    // Drops the J_h_dot * v term in the second derivative (standard
    // manipulator-CBF approximation); alpha_0 / alpha_1 must be tuned
    // conservatively to absorb its effect.

    struct DynCollisionBarrier final : public DynamicBarrier {
        DynCollisionBarrier(const Model & /*model*/, const CollisionModel &col_model,
                            double activation_distance_)
            : num_pairs(static_cast<int>(col_model.collision_pairs.size())),
              activation_distance(activation_distance_) {}

        const int num_pairs;
        double activation_distance;

        [[nodiscard]] auto rows() const noexcept -> int override { return num_pairs; }

        void compute_torque_constraint(const Model &model, Data &data,
                                       const CollisionModel *col_model,
                                       const CollisionData *col_data,
                                       const Eigen::Ref<const Eigen::VectorXd> &v,
                                       Eigen::Ref<Eigen::MatrixXd> A,
                                       Eigen::Ref<Eigen::VectorXd> b) const override;

        // Reads col_data->distance_results directly; caller (asif_validate)
        // refreshes collision queries at the candidate q first.
        [[nodiscard]] auto
        evaluate_at(const Model &model, const Data &data, const CollisionModel *col_model,
                    const CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &q,
                    const Eigen::Ref<const Eigen::VectorXd> &v) const -> double override;
    };

}  // namespace xarm_geo
