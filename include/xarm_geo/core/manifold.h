#pragma once

#include <Eigen/Dense>
#include <smooth/lie_groups.hpp>
#include <smooth/se3.hpp>
#include <smooth/so3.hpp>

namespace xarm_geo::manifold {
    using smooth::LieGroup;

    // --- Co-Tangent Space (Dual of Tangent Space) ---
    // Represents generalized forces (e.g., Wrenches, Torques) -> Not Implemented in `smooth`

    template <LieGroup G> struct CoTangent {
        using Scalar = typename G::Scalar;
        static constexpr int dof = G::Tangent::SizeAtCompileTime;
        using VectorType = Eigen::Matrix<Scalar, dof, 1>;

        VectorType coeffs = VectorType::Zero();

        CoTangent() = default;
        explicit CoTangent(const VectorType &v) : coeffs(v) {}

        // Math Operations

        [[nodiscard]] auto operator+(const CoTangent &rhs) const -> CoTangent {
            return CoTangent(coeffs + rhs.coeffs);
        }

        [[nodiscard]] auto operator-(const CoTangent &rhs) const -> CoTangent {
            return CoTangent(coeffs - rhs.coeffs);
        }

        [[nodiscard]] auto operator-() const -> CoTangent { return CoTangent(-coeffs); }

        [[nodiscard]] friend auto operator*(Scalar lhs, const CoTangent &rhs) -> CoTangent {
            return CoTangent(lhs * rhs.coeffs);
        }

        [[nodiscard]] friend auto operator*(const CoTangent &lhs, Scalar rhs) -> CoTangent {
            return CoTangent(lhs.coeffs * rhs);
        }

        // Dual Pairing (e.g., Mechanical Power for SE(3): P = W^T * V)
        [[nodiscard]] auto dot(const typename G::Tangent &tangent) const -> Scalar {
            return coeffs.dot(tangent);
        }
    };

    // --- SE(3) & SO(3) Wrappers ---

    // SE(3) Wrapper (6D Spatial Kinematics/Dynamics)
    struct SE3 : public smooth::SE3d {
        using smooth::SE3d::SE3d;
        SE3(const smooth::SE3d &base) : smooth::SE3d(base) {}

        using Jacobian = Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Eigen::Dynamic>;
        using SpatialInertia =
            Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Tangent::SizeAtCompileTime>;
        using Twist = smooth::SE3d::Tangent;
        using Wrench = CoTangent<smooth::SE3d>;
    };

    // SO(3) Wrapper (3D Rotational Kinematics/Dynamics)
    struct SO3 : public smooth::SO3d {
        using smooth::SO3d::SO3d;
        SO3(const smooth::SO3d &base) : smooth::SO3d(base) {}

        using Jacobian = Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Eigen::Dynamic>;
        using RotInertia =
            Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Tangent::SizeAtCompileTime>;
        using AngularVel = smooth::SO3d::Tangent;
        using Torque = CoTangent<smooth::SO3d>;
    };

    // --- Geometric Transport Functions ---

    // Transport a tangent vector (e.g., velocity) from frame B to frame A
    // Math: V_A = Ad_{T_AB} * V_B
    template <LieGroup G>
    [[nodiscard]] inline auto transport_tangent(const typename G::Tangent &V_B, const G &T_AB) ->
        typename G::Tangent {
        return T_AB.Ad() * V_B;
    }

    // Transport a co-tangent vector (e.g., wrench) from frame B to frame A
    // Math: W_A = Ad_{T_AB^{-1}}^T * W_B
    template <LieGroup G>
    [[nodiscard]] inline auto transport_cotangent(const CoTangent<G> &W_B, const G &T_AB)
        -> CoTangent<G> {
        return CoTangent<G>(T_AB.inverse().Ad().transpose() * W_B.coeffs);
    }

    // --- Matrix Type Aliases ---

    // Maps Joint Space Velocity -> Tangent Space
    template <LieGroup G>
    using Jacobian =
        Eigen::Matrix<typename G::Scalar, G::Tangent::SizeAtCompileTime, Eigen::Dynamic>;

    // Maps Tangent Space -> Co-Tangent Space
    template <LieGroup G>
    using Inertia = Eigen::Matrix<typename G::Scalar, G::Tangent::SizeAtCompileTime,
                                  G::Tangent::SizeAtCompileTime>;

    // Mass is mathematically identical in shape to Inertia in this context
    template <LieGroup G> using Mass = Inertia<G>;

}  // namespace xarm_geo::manifold
