#pragma once

#include <array>
#include <stdexcept>
#include <utility>

#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- Concatenated Task Trajectory ---
    //
    // Chains >= 2 TaskTrajectory segments end-to-end; total duration is the
    // sum of segment durations. Constructed via CTAD: `ConcatenatedTask{seg0, seg1, ...}`.
    //
    // All segment durations must be finite and positive; infinite-duration
    // segments throw std::invalid_argument. Derivatives are not smoothed at
    // segment seams -- expect step changes in target.twist if adjacent
    // segments don't share boundary velocities.

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

    // --- Concatenated Joint Trajectory ---
    //
    // Joint-space counterpart of ConcatenatedTask. All segments must share dof();
    // mismatches throw std::invalid_argument at construction.

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

    // --- Time-Scaled Task Trajectory ---
    //
    // Plays a TaskTrajectory at a different speed (scale > 1 slows down,
    // scale < 1 speeds up). Derivatives are chain-rule corrected:
    //   duration  = inner.duration() * scale
    //   twist     = twist_inner       / scale
    //   spat_acc  = spat_acc_inner    / (scale * scale)

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

    // --- Time-Scaled Joint Trajectory ---
    //
    // Joint-space counterpart of TimeScaledTask:
    //   v = v_inner / scale,  a = a_inner / (scale * scale)

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

    // --- Offset Task Trajectory ---
    //
    // Applies a fixed SE(3) transform to every pose, in one of two conventions:
    //   Left  (default): pose = transform * pose_inner. World-frame relocation;
    //                    body twist / spatial_acc unchanged.
    //   Right         : pose = pose_inner * transform. Tool-frame offset
    //                    (e.g. extended reach); body frame rotates with the
    //                    trailing transform, so derivatives are pulled back:
    //                      twist     = Ad_{transform^{-1}} * twist_inner
    //                      spat_acc  = Ad_{transform^{-1}} * spat_acc_inner

    enum class OffsetSide { Left, Right };

    template <TaskTrajectory T> class OffsetTask {
    public:
        OffsetTask(T inner, manifold::SE3 transform, OffsetSide side = OffsetSide::Left)
            : inner_(std::move(inner)), transform_(std::move(transform)), side_(side) {}

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            const auto status = inner_.evaluate(t, target);
            if (status != TrajectoryStatus::OK) { return status; }

            if (side_ == OffsetSide::Left) {
                // World-frame relocation; body twist / spatial_acc unaffected.
                target.pose = transform_ * target.pose;
            } else {
                // Tool-frame offset: pull back via Ad_{transform^{-1}}.
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

    // --- Reversed Task Trajectory ---
    //
    // Plays a TaskTrajectory backwards: t_inner = duration - t_out. Velocity
    // reverses sign; spatial acceleration sign is preserved (d²/d(-t)² = d²/dt²).
    // A reversed trajectory starting at non-zero velocity will present a
    // non-zero initial twist to the controller.

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

    // --- Reversed Joint Trajectory ---
    //
    // Joint-space counterpart of ReversedTask: v = -v_inner, a = a_inner.

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
