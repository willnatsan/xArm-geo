#pragma once

#include <Eigen/Dense>
#include <smooth/lie_groups.hpp>
#include <smooth/se3.hpp>
#include <smooth/so3.hpp>
#include <smooth/spline/bspline.hpp>

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
        CoTangent(const VectorType &v) : coeffs(v) {}

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

        // Matrix * CoTangent -> CoTangent (e.g., Ad^T * W, J^T * W)
        template <typename Derived>
        [[nodiscard]] friend auto operator*(const Eigen::MatrixBase<Derived> &M, const CoTangent &W)
            -> CoTangent {
            return CoTangent(M * W.coeffs);
        }

        // Compound Assignment

        auto operator+=(const CoTangent &rhs) -> CoTangent & {
            coeffs += rhs.coeffs;
            return *this;
        }

        auto operator-=(const CoTangent &rhs) -> CoTangent & {
            coeffs -= rhs.coeffs;
            return *this;
        }

        template <typename Derived>
        auto operator+=(const Eigen::MatrixBase<Derived> &rhs) -> CoTangent & {
            coeffs += rhs;
            return *this;
        }

        template <typename Derived>
        auto operator-=(const Eigen::MatrixBase<Derived> &rhs) -> CoTangent & {
            coeffs -= rhs;
            return *this;
        }

        // Block Accessors (return Eigen block expressions; usable as lvalues)

        template <int N> auto head() { return coeffs.template head<N>(); }
        template <int N> auto head() const { return coeffs.template head<N>(); }
        template <int N> auto tail() { return coeffs.template tail<N>(); }
        template <int N> auto tail() const { return coeffs.template tail<N>(); }

        auto head(int n) { return coeffs.head(n); }
        auto head(int n) const { return coeffs.head(n); }
        auto tail(int n) { return coeffs.tail(n); }
        auto tail(int n) const { return coeffs.tail(n); }

        // NoAlias Proxy (enables `wrench.noalias() += M * v;` style assignments)
        auto noalias() { return coeffs.noalias(); }

        // Dual Pairing (e.g., Mechanical Power for SE(3): P = W^T * V)
        template <typename TangentType>
        [[nodiscard]] auto dot(const TangentType &tangent) const -> Scalar {
            return coeffs.dot(tangent);
        }
    };

    // --- SE(3) & SO(3) Wrappers ---

    // SE(3) Wrapper (6D Spatial Kinematics/Dynamics)
    struct SE3 final : public smooth::SE3d {
        using smooth::SE3d::SE3d;
        SE3(const smooth::SE3d &base) : smooth::SE3d(base) {}

        using Jacobian = Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Eigen::Dynamic>;
        using SpatialInertia =
            Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Tangent::SizeAtCompileTime>;
        using SpatialAcceleration = smooth::SE3d::Tangent;
        using Twist = smooth::SE3d::Tangent;
        using Wrench = CoTangent<smooth::SE3d>;

        template <int Degree> using Spline = smooth::BSpline<Degree, SE3>;
    };

    // SO(3) Wrapper (3D Rotational Kinematics/Dynamics)
    struct SO3 final : public smooth::SO3d {
        using smooth::SO3d::SO3d;
        SO3(const smooth::SO3d &base) : smooth::SO3d(base) {}

        using Jacobian = Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Eigen::Dynamic>;
        using RotInertia =
            Eigen::Matrix<Scalar, Tangent::SizeAtCompileTime, Tangent::SizeAtCompileTime>;
        using RotAcceleration = smooth::SO3d::Tangent;
        using AngularVel = smooth::SO3d::Tangent;
        using Torque = CoTangent<smooth::SO3d>;

        template <int Degree> using Spline = smooth::BSpline<Degree, SO3>;
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
    template <LieGroup G, int DOF>
    using Jacobian = Eigen::Matrix<typename G::Scalar, G::Tangent::SizeAtCompileTime, DOF>;

    // Maps Tangent Space -> Co-Tangent Space
    template <LieGroup G>
    using Inertia = Eigen::Matrix<typename G::Scalar, G::Tangent::SizeAtCompileTime,
                                  G::Tangent::SizeAtCompileTime>;

    // Mass is mathematically identical in shape to Inertia in this context
    template <LieGroup G> using Mass = Inertia<G>;

    // --- Helper Functions ---

    // Convert Roll-Pitch-Yaw (ZYX Euler angles) to SO(3) rotation using smooth's exp map
    [[nodiscard]] inline auto rpy_to_SO3(double roll, double pitch, double yaw) -> SO3 {
        return SO3::exp(yaw * Eigen::Vector3d::UnitZ()) *
               SO3::exp(pitch * Eigen::Vector3d::UnitY()) *
               SO3::exp(roll * Eigen::Vector3d::UnitX());
    }

    // Wrap Angle to [-pi, pi]
    [[nodiscard]] inline auto wrap_to_pi(double angle) -> double {
        return std::remainder(angle, 2.0 * std::numbers::pi);
    }

    // Wrap Angle to [center - pi, center + pi], respecting Joint Limit Midpoints
    [[nodiscard]] inline auto wrap_to_range(double angle, double center) -> double {
        return center + std::remainder(angle - center, 2.0 * std::numbers::pi);
    }

}  // namespace xarm_geo::manifold
