#include <algorithm>
#include <cmath>
#include <limits>

#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/safety/barriers.h>

namespace xarm_geo {

    // --- PositionBarrier ---
    //
    // For each joint i (rows 2*i, 2*i+1):
    //   Row 2*i:   +dq_i / dt <= alpha * (q_max_i - q_i - margin)
    //   Row 2*i+1: -dq_i / dt <= alpha * (q_i - q_min_i - margin)

    void PositionBarrier::compute(const Model &model, Data &data,
                                  const CollisionModel * /*col_model*/,
                                  const CollisionData * /*col_data*/, Eigen::Ref<Eigen::MatrixXd> G,
                                  Eigen::Ref<Eigen::VectorXd> b) const {

        G.setZero();

        const double inv_dt = 1.0 / dt;

        for (int i = 0; i < model.dof; ++i) {
            const double q_i = data.q[i];
            const double q_min = model.limits[i].q_min;
            const double q_max = model.limits[i].q_max;

            // Upper limit row
            G(2 * i, i) = +inv_dt;
            b(2 * i) = alpha * ((q_max - q_i) - margin);

            // Lower limit row
            G(2 * i + 1, i) = -inv_dt;
            b(2 * i + 1) = alpha * ((q_i - q_min) - margin);
        }
    }

    // --- CollisionBarrier ---
    //
    // For each collision pair k (one row per pair):
    //   h_k     = d_k - activation_distance
    //   J_h_k   = n^T * (J_point_2 - J_point_1)        (1 x dof)
    //   G(k, :) = -J_h_k / dt
    //   b(k)    = alpha * (h_k - margin)

    void CollisionBarrier::compute(const Model &model, Data &data, const CollisionModel *col_model,
                                   const CollisionData *col_data, Eigen::Ref<Eigen::MatrixXd> G,
                                   Eigen::Ref<Eigen::VectorXd> b) const {

        if (col_model == nullptr || col_data == nullptr) {
            G.setZero();
            b.setConstant(std::numeric_limits<double>::infinity());
            return;
        }

        G.setZero();
        const double inv_dt = 1.0 / dt;

        auto &J_p1 = data.collision.point_jacobian_1;
        auto &J_p2 = data.collision.point_jacobian_2;

        for (int k = 0; k < num_pairs; ++k) {
            const auto &pair = col_model->collision_pairs[k];
            const auto &dist_result = col_data->distance_results[k];

            const double d_k = dist_result.min_distance;
            const Eigen::Vector3d &p1 = dist_result.nearest_points[0];
            const Eigen::Vector3d &p2 = dist_result.nearest_points[1];

            const double d_safe = std::max(std::abs(d_k), 1e-9);
            Eigen::Vector3d n = (p2 - p1) / d_safe;
            if (d_k < 0.0) { n = -n; }

            const auto &g1 = col_model->geometries[pair.obj1_idx];
            const auto &g2 = col_model->geometries[pair.obj2_idx];

            compute_point_jacobian(model, data, g1.parent_joint, p1, J_p1);
            compute_point_jacobian(model, data, g2.parent_joint, p2, J_p2);

            G.row(k) = (-(n.transpose() * (J_p2 - J_p1))) * inv_dt;

            const double h_k = d_k - activation_distance;
            b(k) = alpha * (h_k - margin);
        }
    }

    // --- DynPositionBarrier ---

    void DynPositionBarrier::compute_torque_constraint(const Model &model, Data &data,
                                                       const CollisionModel * /*col_model*/,
                                                       const CollisionData * /*col_data*/,
                                                       const Eigen::Ref<const Eigen::VectorXd> &v,
                                                       Eigen::Ref<Eigen::MatrixXd> A,
                                                       Eigen::Ref<Eigen::VectorXd> b) const {

        const auto &M_inv = data.asif.M_inv;
        const auto &M_inv_h = data.asif.M_inv_h;

        for (int i = 0; i < model.dof; ++i) {
            const double q_i = data.q[i];
            const double v_i = v[i];
            const double q_min = model.limits[i].q_min;
            const double q_max = model.limits[i].q_max;

            // Upper limit row (h_upper = q_max - q_i)
            A.row(2 * i) = M_inv.row(i);
            b(2 * i) = alpha_0 * ((q_max - q_i) - margin) + alpha_1 * (-v_i) + M_inv_h[i];

            // Lower limit row (h_lower = q_i - q_min)
            A.row(2 * i + 1) = -M_inv.row(i);
            b(2 * i + 1) = alpha_0 * ((q_i - q_min) - margin) + alpha_1 * (+v_i) - M_inv_h[i];
        }
    }

    auto DynPositionBarrier::evaluate_at(const Model &model, const Data & /*data*/,
                                         const CollisionModel * /*col_model*/,
                                         const CollisionData * /*col_data*/,
                                         const Eigen::Ref<const Eigen::VectorXd> &q,
                                         const Eigen::Ref<const Eigen::VectorXd> & /*v*/) const
        -> double {

        double h_min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < model.dof; ++i) {
            h_min = std::min(h_min, model.limits[i].q_max - q[i] - margin);
            h_min = std::min(h_min, q[i] - model.limits[i].q_min - margin);
        }
        return h_min;
    }

    // --- DynVelocityBarrier ---

    void DynVelocityBarrier::compute_torque_constraint(const Model &model, Data &data,
                                                       const CollisionModel * /*col_model*/,
                                                       const CollisionData * /*col_data*/,
                                                       const Eigen::Ref<const Eigen::VectorXd> &v,
                                                       Eigen::Ref<Eigen::MatrixXd> A,
                                                       Eigen::Ref<Eigen::VectorXd> b) const {

        const auto &M_inv = data.asif.M_inv;
        const auto &M_inv_h = data.asif.M_inv_h;

        const bool use_override = (v_max.size() == model.dof);

        for (int i = 0; i < model.dof; ++i) {
            const double v_i = v[i];
            const double v_max_i = use_override ? v_max[i] : model.limits[i].q_vel_max;

            // Upper bound row (h_upper = v_max - v_i)
            A.row(2 * i) = M_inv.row(i);
            b(2 * i) = alpha_0 * ((v_max_i - v_i) - margin) + M_inv_h[i];

            // Lower bound row (h_lower = v_i + v_max -> v_i >= -v_max)
            A.row(2 * i + 1) = -M_inv.row(i);
            b(2 * i + 1) = alpha_0 * ((v_i + v_max_i) - margin) - M_inv_h[i];
        }
    }

    auto DynVelocityBarrier::evaluate_at(const Model &model, const Data & /*data*/,
                                         const CollisionModel * /*col_model*/,
                                         const CollisionData * /*col_data*/,
                                         const Eigen::Ref<const Eigen::VectorXd> & /*q*/,
                                         const Eigen::Ref<const Eigen::VectorXd> &v) const
        -> double {

        double h_min = std::numeric_limits<double>::infinity();
        const bool use_override = (v_max.size() == model.dof);
        for (int i = 0; i < model.dof; ++i) {
            const double v_max_i = use_override ? v_max[i] : model.limits[i].q_vel_max;
            h_min = std::min(h_min, v_max_i - v[i] - margin);
            h_min = std::min(h_min, v[i] + v_max_i - margin);
        }
        return h_min;
    }

    // --- DynCollisionBarrier ---

    void DynCollisionBarrier::compute_torque_constraint(const Model &model, Data &data,
                                                        const CollisionModel *col_model,
                                                        const CollisionData *col_data,
                                                        const Eigen::Ref<const Eigen::VectorXd> &v,
                                                        Eigen::Ref<Eigen::MatrixXd> A,
                                                        Eigen::Ref<Eigen::VectorXd> b) const {

        if (col_model == nullptr || col_data == nullptr) {
            A.setZero();
            b.setConstant(std::numeric_limits<double>::infinity());
            return;
        }

        const auto &M_inv = data.asif.M_inv;
        const auto &M_inv_h = data.asif.M_inv_h;

        auto &J_p1 = data.collision.point_jacobian_1;
        auto &J_p2 = data.collision.point_jacobian_2;

        for (int k = 0; k < num_pairs; ++k) {
            const auto &pair = col_model->collision_pairs[k];
            const auto &dist_result = col_data->distance_results[k];

            const double d_k = dist_result.min_distance;
            const Eigen::Vector3d &p1 = dist_result.nearest_points[0];
            const Eigen::Vector3d &p2 = dist_result.nearest_points[1];

            const double d_safe = std::max(std::abs(d_k), 1e-9);
            Eigen::Vector3d n = (p2 - p1) / d_safe;
            if (d_k < 0.0) { n = -n; }

            const auto &g1 = col_model->geometries[pair.obj1_idx];
            const auto &g2 = col_model->geometries[pair.obj2_idx];

            compute_point_jacobian(model, data, g1.parent_joint, p1, J_p1);
            compute_point_jacobian(model, data, g2.parent_joint, p2, J_p2);

            const Eigen::RowVectorXd J_h = n.transpose() * (J_p2 - J_p1);

            A.row(k) = -(J_h * M_inv);

            const double h_k = d_k - activation_distance;
            b(k) = alpha_0 * (h_k - margin) + alpha_1 * (J_h * v).value() - (J_h * M_inv_h).value();
        }
    }

    auto DynCollisionBarrier::evaluate_at(const Model & /*model*/, const Data & /*data*/,
                                          const CollisionModel * /*col_model*/,
                                          const CollisionData *col_data,
                                          const Eigen::Ref<const Eigen::VectorXd> & /*q*/,
                                          const Eigen::Ref<const Eigen::VectorXd> & /*v*/) const
        -> double {

        if (col_data == nullptr) { return std::numeric_limits<double>::infinity(); }

        // Caller has already refreshed col_data at the candidate q.
        double h_min = std::numeric_limits<double>::infinity();
        for (int k = 0; k < num_pairs; ++k) {
            const double d_k = col_data->distance_results[k].min_distance;
            h_min = std::min(h_min, d_k - activation_distance - margin);
        }
        return h_min;
    }

}  // namespace xarm_geo
