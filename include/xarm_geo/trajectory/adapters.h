#pragma once

#include <array>
#include <stdexcept>
#include <utility>

#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- ConcatenatedTask<Ts...> ---
    //
    // Chains two or more TaskTrajectory segments end-to-end. The combined
    // duration is the sum of the individual durations. evaluate(t) locates the
    // active segment and forwards the time offset into it.
    //
    // Construction:
    //   ConcatenatedTask traj{seg0, seg1, seg2};   // CTAD; takes by move
    //
    // Caveats:
    //   - All segment durations must be finite; an infinite-duration segment
    //     (e.g. a setpoint) will throw std::invalid_argument at construction.
    //   - Derivatives (twist, spatial_acc) are NOT smoothed at seam boundaries.
    //     If seg_i ends with non-zero velocity and seg_{i+1} begins at rest,
    //     the controller will observe a step change in target.twist.

    template <TaskTrajectory... Ts> class ConcatenatedTask {
        static constexpr std::size_t N = sizeof...(Ts);
        static_assert(N >= 2, "ConcatenatedTask requires at least 2 segments");

    public:
        explicit ConcatenatedTask(Ts... segs) : segments_(std::move(segs)...) {
            // Build prefix-sum array and validate finite durations.
            prefix_[0] = 0.0;
            build_prefix(std::index_sequence_for<Ts...>{});
            total_duration_ = prefix_[N];
        }

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            if (t < 0.0 || t > total_duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

            // Locate active segment via linear scan of prefix sums.
            std::size_t idx = N - 1;
            for (std::size_t i = 0; i < N - 1; ++i) {
                if (t < prefix_[i + 1]) {
                    idx = i;
                    break;
                }
            }

            const double local_t = t - prefix_[idx];
            return dispatch_evaluate(idx, local_t, target, std::index_sequence_for<Ts...>{});
        }

        [[nodiscard]] auto duration() const noexcept -> double { return total_duration_; }

    private:
        std::tuple<Ts...> segments_;
        std::array<double, N + 1> prefix_{};
        double total_duration_ = 0.0;

        template <std::size_t... Is> void build_prefix(std::index_sequence<Is...>) {
            ((build_prefix_one<Is>()), ...);
        }

        template <std::size_t I> void build_prefix_one() {
            const double d = std::get<I>(segments_).duration();
            if (!std::isfinite(d) || d <= 0.0) {
                throw std::invalid_argument(
                    "ConcatenatedTask: all segment durations must be finite and positive "
                    "(segment " +
                    std::to_string(I) + " has duration " + std::to_string(d) + ")");
            }
            prefix_[I + 1] = prefix_[I] + d;
        }

        template <std::size_t... Is>
        [[nodiscard]] auto dispatch_evaluate(std::size_t idx, double local_t, TaskTarget &target,
                                             std::index_sequence<Is...>) const -> TrajectoryStatus {
            TrajectoryStatus result = TrajectoryStatus::ERROR;
            ((Is == idx ? (result = std::get<Is>(segments_).evaluate(local_t, target), true)
                        : false),
             ...);
            return result;
        }
    };

    // CTAD deduction guide.
    template <TaskTrajectory... Ts> ConcatenatedTask(Ts...) -> ConcatenatedTask<Ts...>;

    // --- ConcatenatedJoint<Ts...> ---
    //
    // Joint-space counterpart of ConcatenatedTask. All segments must have the
    // same dof(); an std::invalid_argument is thrown at construction otherwise.

    template <JointTrajectory... Ts> class ConcatenatedJoint {
        static constexpr std::size_t N = sizeof...(Ts);
        static_assert(N >= 2, "ConcatenatedJoint requires at least 2 segments");

    public:
        explicit ConcatenatedJoint(Ts... segs) : segments_(std::move(segs)...) {
            prefix_[0] = 0.0;
            build_prefix(std::index_sequence_for<Ts...>{});
            total_duration_ = prefix_[N];
            dof_ = std::get<0>(segments_).dof();
            validate_dof(std::index_sequence_for<Ts...>{});
        }

        [[nodiscard]] auto evaluate(double t, JointTarget &target) const -> TrajectoryStatus {
            if (t < 0.0 || t > total_duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

            std::size_t idx = N - 1;
            for (std::size_t i = 0; i < N - 1; ++i) {
                if (t < prefix_[i + 1]) {
                    idx = i;
                    break;
                }
            }

            const double local_t = t - prefix_[idx];
            return dispatch_evaluate(idx, local_t, target, std::index_sequence_for<Ts...>{});
        }

        [[nodiscard]] auto duration() const noexcept -> double { return total_duration_; }
        [[nodiscard]] auto dof() const noexcept -> int { return dof_; }

    private:
        std::tuple<Ts...> segments_;
        std::array<double, N + 1> prefix_{};
        double total_duration_ = 0.0;
        int dof_ = 0;

        template <std::size_t... Is> void build_prefix(std::index_sequence<Is...>) {
            ((build_prefix_one<Is>()), ...);
        }

        template <std::size_t I> void build_prefix_one() {
            const double d = std::get<I>(segments_).duration();
            if (!std::isfinite(d) || d <= 0.0) {
                throw std::invalid_argument(
                    "ConcatenatedJoint: all segment durations must be finite and positive "
                    "(segment " +
                    std::to_string(I) + " has duration " + std::to_string(d) + ")");
            }
            prefix_[I + 1] = prefix_[I] + d;
        }

        template <std::size_t... Is> void validate_dof(std::index_sequence<Is...>) {
            ((Is > 0 && std::get<Is>(segments_).dof() != dof_
                  ? throw std::invalid_argument("ConcatenatedJoint: dof mismatch at segment " +
                                                std::to_string(Is) + " (expected " +
                                                std::to_string(dof_) + ", got " +
                                                std::to_string(std::get<Is>(segments_).dof()) + ")")
                  : (void)0),
             ...);
        }

        template <std::size_t... Is>
        [[nodiscard]] auto dispatch_evaluate(std::size_t idx, double local_t, JointTarget &target,
                                             std::index_sequence<Is...>) const -> TrajectoryStatus {
            TrajectoryStatus result = TrajectoryStatus::ERROR;
            ((Is == idx ? (result = std::get<Is>(segments_).evaluate(local_t, target), true)
                        : false),
             ...);
            return result;
        }
    };

    template <JointTrajectory... Ts> ConcatenatedJoint(Ts...) -> ConcatenatedJoint<Ts...>;

    // --- TimeScaledTask<T> ---
    //
    // Plays a TaskTrajectory at a different speed. A scale > 1 slows down
    // (longer duration); a scale < 1 speeds up (shorter duration).
    //
    //   duration_out = inner.duration() * scale
    //   t_inner      = t_out / scale
    //
    // Derivatives are chain-rule corrected:
    //   twist_out        = twist_inner        / scale
    //   spatial_acc_out  = spatial_acc_inner  / (scale * scale)

    template <TaskTrajectory T> class TimeScaledTask {
    public:
        TimeScaledTask(T inner, double scale) : inner_(std::move(inner)), scale_(scale) {
            if (scale_ <= 0.0) { throw std::invalid_argument("TimeScaledTask: scale must be > 0"); }
        }

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(t / scale_, target);
            if (status != TrajectoryStatus::OK) { return status; }
            target.twist /= scale_;
            target.spatial_acc /= (scale_ * scale_);
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double {
            return inner_.duration() * scale_;
        }

    private:
        T inner_;
        double scale_;
    };

    // --- TimeScaledJoint<T> ---
    //
    // Joint-space counterpart of TimeScaledTask.
    //   v_out = v_inner / scale
    //   a_out = a_inner / (scale * scale)

    template <JointTrajectory T> class TimeScaledJoint {
    public:
        TimeScaledJoint(T inner, double scale) : inner_(std::move(inner)), scale_(scale) {
            if (scale_ <= 0.0) {
                throw std::invalid_argument("TimeScaledJoint: scale must be > 0");
            }
        }

        [[nodiscard]] auto evaluate(double t, JointTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(t / scale_, target);
            if (status != TrajectoryStatus::OK) { return status; }
            target.v /= scale_;
            target.a /= (scale_ * scale_);
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double {
            return inner_.duration() * scale_;
        }

        [[nodiscard]] auto dof() const noexcept -> int { return inner_.dof(); }

    private:
        T inner_;
        double scale_;
    };

    // --- OffsetTask<T> ---
    //
    // Applies a fixed SE(3) transform to every pose produced by a
    // TaskTrajectory. Two conventions are supported:
    //
    //   Left (default): pose_out = transform * pose_inner
    //     Moves the trajectory to a different world-frame location.
    //     Body twist and spatial acceleration are unchanged because the
    //     end-effector body frame is not rotated relative to the original.
    //
    //   Right: pose_out = pose_inner * transform
    //     Applies a fixed tool-frame offset (e.g. extended reach, sensor
    //     offset). The body frame moves with the trailing transform, so
    //     twist and spatial_acc must be pulled back:
    //       twist_out       = Ad_{transform^{-1}} * twist_inner
    //       spatial_acc_out = Ad_{transform^{-1}} * spatial_acc_inner

    enum class OffsetSide { Left, Right };

    template <TaskTrajectory T> class OffsetTask {
    public:
        OffsetTask(T inner, manifold::SE3 transform, OffsetSide side = OffsetSide::Left)
            : inner_(std::move(inner)), transform_(std::move(transform)), side_(side) {}

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(t, target);
            if (status != TrajectoryStatus::OK) { return status; }

            if (side_ == OffsetSide::Left) {
                // Pose moves to a different world location; body frame is
                // unchanged, so body-frame twist and spatial_acc are unaffected.
                target.pose = transform_ * target.pose;
            } else {
                // Tool-frame offset: body frame rotates with the transform.
                // Pull back via Ad_{transform^{-1}} = Ad of the inverse.
                const auto Ad_inv = transform_.inverse().Ad();
                target.pose = target.pose * transform_;
                target.twist = Ad_inv * target.twist;
                target.spatial_acc = Ad_inv * target.spatial_acc;
            }

            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return inner_.duration(); }

    private:
        T inner_;
        manifold::SE3 transform_;
        OffsetSide side_;
    };

    // --- ReversedTask<T> ---
    //
    // Plays a TaskTrajectory backwards in time.
    //
    //   t_inner = duration - t_out
    //
    // Body-frame derivative corrections:
    //   twist_out       = -twist_inner      (velocity reverses sign)
    //   spatial_acc_out =  spatial_acc_inner (second derivative sign unchanged)
    //
    // Note: Be aware that a reversed trajectory starting at non-zero velocity
    // will present a non-zero initial twist to the controller.

    template <TaskTrajectory T> class ReversedTask {
    public:
        explicit ReversedTask(T inner) : inner_(std::move(inner)) {}

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(inner_.duration() - t, target);
            if (status != TrajectoryStatus::OK) { return status; }
            target.twist = -target.twist;
            // spatial_acc sign is unchanged: d²/d(-t)² = d²/dt²
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return inner_.duration(); }

    private:
        T inner_;
    };

    // --- ReversedJoint<T> ---
    //
    // Joint-space counterpart of ReversedTask.
    //   v_out = -v_inner
    //   a_out =  a_inner  (second derivative; sign unchanged under time reversal)

    template <JointTrajectory T> class ReversedJoint {
    public:
        explicit ReversedJoint(T inner) : inner_(std::move(inner)) {}

        [[nodiscard]] auto evaluate(double t, JointTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(inner_.duration() - t, target);
            if (status != TrajectoryStatus::OK) { return status; }
            target.v = -target.v;
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return inner_.duration(); }
        [[nodiscard]] auto dof() const noexcept -> int { return inner_.dof(); }

    private:
        T inner_;
    };

}  // namespace xarm_geo
