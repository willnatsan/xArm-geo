#pragma once

#include <algorithm>

namespace xarm_geo {

    // --- ConvergenceMonitor ---
    //
    // Counts consecutive ticks where a scalar error norm is at or below
    // `threshold`. After `min_consecutive` such ticks, `converged()` returns
    // true. A single above-threshold tick resets the counter to zero.
    //
    // This is a settling-time heuristic, NOT a stability proof. Use it for
    // detecting that the controller has arrived at a setpoint or that tracking
    // error has settled. Do NOT equate `converged()` with Lyapunov stability.

    struct ConvergenceMonitor {
        // --- Public Configuration ---
        double threshold = 1e-3;   // Error norm below this counts as near-zero.
        int min_consecutive = 50;  // Consecutive below-threshold ticks required.

        // Increments the counter if error_norm <= threshold, resets it to zero otherwise.
        void update(double error_norm) noexcept {
            if (error_norm <= threshold) {
                counter_ = std::min(counter_ + 1, min_consecutive);
            } else {
                counter_ = 0;
            }
        }

        // Reset the counter (call from the controller's reset()).
        void reset() noexcept { counter_ = 0; }

        // True once the counter has reached min_consecutive.
        [[nodiscard]] auto converged() const noexcept -> bool {
            return counter_ >= min_consecutive;
        }

    private:
        int counter_ = 0;
    };

}  // namespace xarm_geo
