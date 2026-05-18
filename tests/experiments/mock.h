#pragma once

// --- Mock Hardware Augmenter ---
//
// Wraps the read/write boundary of a Simulation interface with hardware-style
// non-idealities, so a sim-domain control loop can stand in for the missing
// hardware variant of an experiment.  Used (with supervisor approval) where
// physical hardware is unavailable for a specific variant.
//
// Augmentations are layered into four tiers, applied in this order on write:
//   Tier 2  per-joint gain error (multiplicative, sampled once)
//   Tier 2  per-joint DC bias    (additive,       sampled once)
//   Tier 1  Gaussian command noise on commanded velocity (per-tick)
//   Tier 3  direction-reversal backlash mask (per-joint tick countdown)
//   Tier 3  deadband (zero |v| below threshold)
//   --      servo low-pass (existing)
//   --      quantisation (existing)
//
// On read:
//   Tier 4  Ornstein-Uhlenbeck slow drift added to observed q
//   --      Gaussian encoder noise on q and v
//
// Tier 1 + 2 propagate through the physics (they perturb the actual command),
// so they alter the *real* trajectory.  Tier 4 stays in the observation channel.
//
// Magnitudes were calibrated from the existing hardware reference trace; Tier
// 3/4 default off.  RNG is seedable for reproducibility.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include <xarm_geo/core/motion.h>

namespace xarm_geo::experiments {

    struct MockHardwareOptions {
        // Master switch -- when false, all apply_* calls are no-ops.
        bool enabled = false;

        // ---- Tier 0: observation noise ----------------------------------------
        // Encoder noise (Gaussian, per-joint scalar).  Calibrated from
        // hardware reference trace (median across joints).
        double q_noise_std = 1.5e-4;  // rad
        double v_noise_std = 4.0e-3;  // rad/s

        // ---- Tier 1: command-side process noise -------------------------------
        // Gaussian noise added to the commanded velocity before it reaches the
        // physics.  Propagates through the closed loop and produces real
        // tracking error (unlike observation noise).
        double cmd_noise_std = 2.0e-3;  // rad/s

        // ---- Tier 2: static per-joint command imperfections -------------------
        // Sampled once at construction with the seeded RNG, then held constant.
        // Multiplicative gain error per joint: vel[i] *= (1 + N(0, gain_error_std)).
        double gain_error_std = 5.0e-3;
        // Additive DC offset per joint: vel[i] += N(0, cmd_bias_std).
        double cmd_bias_std = 5.0e-4;  // rad/s

        // ---- Tier 3: nonlinear servo behaviour (off by default) ---------------
        // Velocity deadband -- zero out |vel[i]| below this threshold.
        double deadband = 0.0;  // rad/s
        // Direction-reversal backlash -- suppress commanded vel for this many
        // ticks on any joint that just changed sign.
        int backlash_ticks = 0;

        // ---- Tier 4: slow observation drift (off by default) ------------------
        // Ornstein-Uhlenbeck process added to observed q.  Stationary std =
        // drift_std; mean-reversion time constant = drift_tau_s.
        double drift_std = 0.0;    // rad
        double drift_tau_s = 5.0;  // s

        // ---- Servo / interface ------------------------------------------------
        // Velocity-servo 1st-order low-pass time constant.
        double servo_tau_s = 0.025;
        // Quantisation step on commanded velocity (rounding before write).
        double v_quantum = 1e-4;  // rad/s
        // Sleep-deadline jitter, uniform in [-jitter, +jitter].
        double timing_jitter_s = 2e-4;

        // RNG seed.  Identical seed -> identical mock trace.
        std::uint64_t seed = 0xC0FFEEULL;
    };

    // -------------------------------------------------------------------------
    // MockHardwareAugmenter
    // -------------------------------------------------------------------------
    // Stateful: owns RNG, per-joint static error draws, per-joint LP state,
    // backlash countdowns, OU drift state.  Construct once per Phase-2 loop.

    class MockHardwareAugmenter {
    public:
        MockHardwareAugmenter(int dof, const MockHardwareOptions &opts, double dt_s)
            : dof_(dof), dt_s_(dt_s), opts_(opts), rng_(opts.seed),
              v_lp_(Eigen::VectorXd::Zero(dof)), drift_(Eigen::VectorXd::Zero(dof)),
              prev_vel_(Eigen::VectorXd::Zero(dof)), backlash_remaining_(dof, 0),
              v_lp_seeded_(false), prev_vel_seeded_(false) {
            // Discrete 1st-order LP: alpha = exp(-dt/tau); 0 disables.
            lp_alpha_ =
                (opts_.servo_tau_s > 0.0 && dt_s > 0.0) ? std::exp(-dt_s / opts_.servo_tau_s) : 0.0;

            // OU discrete update: x[k+1] = a * x[k] + sigma_d * sqrt(1 - a^2) * N(0,1)
            // with a = exp(-dt/tau).  Pre-compute coefficients.
            if (opts_.drift_tau_s > 0.0 && dt_s > 0.0) {
                ou_a_ = std::exp(-dt_s / opts_.drift_tau_s);
                ou_sigma_step_ = opts_.drift_std * std::sqrt(1.0 - ou_a_ * ou_a_);
            }

            // Pre-sample per-joint static gain and bias.
            gain_.setOnes(dof);
            bias_.setZero(dof);
            if (opts_.gain_error_std > 0.0) {
                std::normal_distribution<double> ng(0.0, opts_.gain_error_std);
                for (int i = 0; i < dof; ++i) { gain_[i] = 1.0 + ng(rng_); }
            }
            if (opts_.cmd_bias_std > 0.0) {
                std::normal_distribution<double> nb(0.0, opts_.cmd_bias_std);
                for (int i = 0; i < dof; ++i) { bias_[i] = nb(rng_); }
            }
        }

        // Inject encoder noise (+ optional OU drift) into a freshly-read JointState.
        void apply_to_read(JointState &state) {
            if (!opts_.enabled) { return; }
            std::normal_distribution<double> nq(0.0, opts_.q_noise_std);
            std::normal_distribution<double> nv(0.0, opts_.v_noise_std);
            std::normal_distribution<double> nd(0.0, 1.0);
            const bool drift_on = opts_.drift_std > 0.0 && ou_sigma_step_ > 0.0;
            for (int i = 0; i < dof_; ++i) {
                if (drift_on) {
                    drift_[i] = ou_a_ * drift_[i] + ou_sigma_step_ * nd(rng_);
                    state.q[i] += drift_[i];
                }
                state.q[i] += nq(rng_);
                state.v[i] += nv(rng_);
            }
        }

        // Apply layered command-side perturbations + servo LP + quantisation.
        void apply_to_write(JointVelocity &vel) {
            if (!opts_.enabled) { return; }

            // Initialise prev_vel on first call so the first tick can't trigger
            // a spurious reversal/backlash event.
            if (!prev_vel_seeded_) {
                prev_vel_ = vel.v;
                prev_vel_seeded_ = true;
            }

            // Tier 2: static gain + bias (per-joint, set once at construction).
            for (int i = 0; i < dof_; ++i) { vel.v[i] = gain_[i] * vel.v[i] + bias_[i]; }

            // Tier 1: per-tick Gaussian command noise.
            if (opts_.cmd_noise_std > 0.0) {
                std::normal_distribution<double> nc(0.0, opts_.cmd_noise_std);
                for (int i = 0; i < dof_; ++i) { vel.v[i] += nc(rng_); }
            }

            // Tier 3: direction-reversal backlash mask.
            if (opts_.backlash_ticks > 0) {
                for (int i = 0; i < dof_; ++i) {
                    const double prev = prev_vel_[i];
                    const double cur = vel.v[i];
                    // Sign reversal detected (both magnitudes above tiny epsilon).
                    if (prev * cur < 0.0 && std::abs(prev) > 1e-6 && std::abs(cur) > 1e-6) {
                        backlash_remaining_[i] = opts_.backlash_ticks;
                    }
                    if (backlash_remaining_[i] > 0) {
                        vel.v[i] = 0.0;
                        --backlash_remaining_[i];
                    }
                }
            }
            // Cache pre-LP command for next tick's reversal detection.
            prev_vel_ = vel.v;

            // Tier 3: deadband.
            if (opts_.deadband > 0.0) {
                for (int i = 0; i < dof_; ++i) {
                    if (std::abs(vel.v[i]) < opts_.deadband) { vel.v[i] = 0.0; }
                }
            }

            // Servo LP + quantise.
            if (!v_lp_seeded_) {
                v_lp_ = vel.v;
                v_lp_seeded_ = true;
            }
            for (int i = 0; i < dof_; ++i) {
                v_lp_[i] = lp_alpha_ * v_lp_[i] + (1.0 - lp_alpha_) * vel.v[i];
                const double q = opts_.v_quantum;
                vel.v[i] = (q > 0.0) ? (std::round(v_lp_[i] / q) * q) : v_lp_[i];
            }
        }

        // Random sub-tick offset to add to a sleep deadline.
        [[nodiscard]] auto jitter_offset() -> std::chrono::nanoseconds {
            if (!opts_.enabled || opts_.timing_jitter_s <= 0.0) {
                return std::chrono::nanoseconds::zero();
            }
            std::uniform_real_distribution<double> u(-opts_.timing_jitter_s, opts_.timing_jitter_s);
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(u(rng_)));
        }

        [[nodiscard]] auto enabled() const -> bool { return opts_.enabled; }
        [[nodiscard]] auto options() const -> const MockHardwareOptions & { return opts_; }

    private:
        int dof_;
        double dt_s_;
        MockHardwareOptions opts_;
        std::mt19937_64 rng_;

        // Servo LP state.
        Eigen::VectorXd v_lp_;
        double lp_alpha_ = 0.0;

        // OU drift state.
        Eigen::VectorXd drift_;
        double ou_a_ = 0.0;
        double ou_sigma_step_ = 0.0;

        // Static per-joint command imperfections (sampled once).
        Eigen::VectorXd gain_;
        Eigen::VectorXd bias_;

        // Backlash bookkeeping.
        Eigen::VectorXd prev_vel_;
        std::vector<int> backlash_remaining_;

        bool v_lp_seeded_;
        bool prev_vel_seeded_;
    };

}  // namespace xarm_geo::experiments
