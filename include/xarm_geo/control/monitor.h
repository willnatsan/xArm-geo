#pragma once

#include <algorithm>

namespace xarm_geo {

    // --- Convergence Monitor ---
    //
    // Settling-time heuristic: converged() returns true once the error norm
    // has stayed at or below `threshold` for `min_consecutive` ticks. A single
    // above-threshold tick resets the counter. Not a stability proof.

    struct ConvergenceMonitor {
        // --- Public Configuration ---
        double threshold = 1e-3;
        int min_consecutive = 50;

        void update(double error_norm) noexcept {
            if (error_norm <= threshold) {
                counter_ = std::min(counter_ + 1, min_consecutive);
            } else {
                counter_ = 0;
            }
        }

        void reset() noexcept { counter_ = 0; }

        [[nodiscard]] auto converged() const noexcept -> bool {
            return counter_ >= min_consecutive;
        }

    private:
        int counter_ = 0;
    };

}  // namespace xarm_geo
