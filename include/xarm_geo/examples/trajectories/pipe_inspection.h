#pragma once

#include <cmath>
#include <numbers>
#include <string_view>
#include <utility>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Pipe Inspection Trajectory ---
    //
    // Circular arc in the YZ plane simulating travel around a pipe interior.
    // Pitch tracks the arc angle to keep the tool normal to the pipe wall.
    // Sampled at 100 points and fitted to a degree-5 B-spline.

    class PipeInspection final : public AnalyticTaskTrajectory {
    public:
        static constexpr std::string_view kName = "PipeInspection";

        explicit PipeInspection(const manifold::SE3 &anchor, double duration = 15.0,
                                double omega = 0.4, double radius = 0.06)
            : AnalyticTaskTrajectory(anchor, duration), omega_(omega), radius_(radius) {
            build_spline();
        }

    protected:
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {
            // Circular Arc in YZ Plane Simulating Travel Around Pipe Interior.
            const double y_pipe = radius_ * std::sin(omega_ * t);
            const double z_pipe = radius_ * (1.0 - std::cos(omega_ * t));
            const Eigen::Vector3d local_pos(0.0, y_pipe, -z_pipe);

            // Pitch Tracks Arc Angle to Keep Tool Normal to Pipe Wall.
            const double pitch_target = std::numbers::pi - (omega_ * t);
            const manifold::SO3 local_rot =
                manifold::SO3::exp(pitch_target * Eigen::Vector3d::UnitY());

            return {local_rot, local_pos};
        }

    private:
        double omega_, radius_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<PipeInspection>);

}  // namespace xarm_geo::trajectories
