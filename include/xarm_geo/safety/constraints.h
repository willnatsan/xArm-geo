#pragma once

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Constraint Base Class ---
    //
    // A Constraint contributes one or more *hard* linear inequality rows to the
    // Optimal IK QP:
    //
    //     l <= G * dq <= u
    //
    // Subclasses fill G (rows() x dof), l (rows()), and u (rows()) in `compute()`.
    // Use +/- infinity for one-sided bounds.

    struct Constraint {
        virtual ~Constraint() = default;

        [[nodiscard]] virtual auto rows() const noexcept -> int = 0;

        // `data` is non-const so subclasses may write to scratch buffers in
        // `data` while reading the canonical state. The Constraint is not
        // expected to mutate any of the canonical state fields.
        virtual void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> G,
                             Eigen::Ref<Eigen::VectorXd> l,
                             Eigen::Ref<Eigen::VectorXd> u) const = 0;
    };

    // --- Joint Velocity Limit Constraint ---
    //
    // Enforces |v_i| <= q_vel_max_i for each joint i, where v_i = dq_i / dt.
    // Expressed as box bounds on dq:
    //
    //     -dt * q_vel_max <= dq <= dt * q_vel_max
    //
    // (G = I).
    //
    // q_vel_max is taken from model.limits[i].q_vel_max by default; override
    // by setting the optional `v_max` member to a vector of size dof.

    struct VelocityLimit final : public Constraint {
        VelocityLimit(const Model &model, double dt_) : dof(model.dof), dt(dt_) {}

        const int dof;          // Cached at construction
        double dt = 0.0;        // Step length used to convert v_max -> dq bound
        Eigen::VectorXd v_max;  // Optional Override (Empty -> Use model.limits)

        [[nodiscard]] auto rows() const noexcept -> int override { return dof; }

        void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> G,
                     Eigen::Ref<Eigen::VectorXd> l, Eigen::Ref<Eigen::VectorXd> u) const override;
    };

    // --- Joint Position Limit Constraint ---
    //
    // Enforces q_min <= q + dq <= q_max for each joint, linearised as:
    //
    //     q_min - q <= dq <= q_max - q
    //
    // (G = I).
    //
    // Reads the current configuration `q` from `data.q` (the canonical state).
    // The user is responsible for ensuring `data.q` reflects the configuration
    // for which the constraint should be evaluated.

    struct PositionLimit final : public Constraint {
        explicit PositionLimit(const Model &model) : dof(model.dof) {}

        const int dof;  // Cached at construction

        [[nodiscard]] auto rows() const noexcept -> int override { return dof; }

        void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> G,
                     Eigen::Ref<Eigen::VectorXd> l, Eigen::Ref<Eigen::VectorXd> u) const override;
    };

}  // namespace xarm_geo
