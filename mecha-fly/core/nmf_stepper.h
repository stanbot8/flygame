#pragma once
// Procedural IK-based gait controller for NeuroMechFly.
// Defines foot trajectories in task space, IK-solves at keyframes,
// and interpolates angles for smooth motion. Fully procedural —
// no walk data replay in the control loop.

#include <algorithm>
#include <cmath>
#include <cstring>
#include "nmf_fk.h"
#include "nmf_ik.h"
#include "nmf_walk_data.h"

namespace nmfly {

namespace cpg_internal {
    constexpr double kPhaseBias[6][6] = {
        {0,    kPi, 0,    kPi, 0,    kPi},
        {kPi, 0,    kPi, 0,    kPi, 0   },
        {0,    kPi, 0,    kPi, 0,    kPi},
        {kPi, 0,    kPi, 0,    kPi, 0   },
        {0,    kPi, 0,    kPi, 0,    kPi},
        {kPi, 0,    kPi, 0,    kPi, 0   },
    };
    constexpr double kCouplingWeight = 10.0;
}

enum class LegPhase { kStance, kLift, kSwing, kPlace };

// Joint angle limits derived from walk data (min/max across all phases).
// Computed once at startup, used to clamp IK outputs.
struct JointLimits {
    double lo[6][7];
    double hi[6][7];
    bool valid = false;

    void ComputeFromWalkData() {
        for (int leg = 0; leg < 6; ++leg)
            for (int j = 0; j < 7; ++j) {
                lo[leg][j] = 1e9;
                hi[leg][j] = -1e9;
            }
        for (int leg = 0; leg < 6; ++leg)
            for (int s = 0; s < kWalkSamples; ++s) {
                double phase = s * kTwoPi / (kWalkSamples - 1);
                for (int j = 0; j < 7; ++j) {
                    double a = WalkLerp(leg, j, phase);
                    if (a < lo[leg][j]) lo[leg][j] = a;
                    if (a > hi[leg][j]) hi[leg][j] = a;
                }
            }
        // Add 30% margin for adaptive stepping.
        for (int leg = 0; leg < 6; ++leg)
            for (int j = 0; j < 7; ++j) {
                double range = hi[leg][j] - lo[leg][j];
                double margin = range * 0.3;
                lo[leg][j] -= margin;
                hi[leg][j] += margin;
            }
        valid = true;
    }

    void Clamp(int leg, double angles[7]) const {
        for (int j = 0; j < 7; ++j)
            angles[j] = std::clamp(angles[j], lo[leg][j], hi[leg][j]);
    }
};

// Number of IK keyframes per phase.
constexpr int kSwingKF = 10;
constexpr int kStanceKF = 6;
constexpr int kTotalKF = kSwingKF + kStanceKF;

struct LegState {
    LegPhase phase = LegPhase::kStance;
    double sub_phase = 0.0;
    double angles[7] = {};
    // Pre-computed keyframe angles for full gait cycle.
    // [0..kSwingKF-1] = swing, [kSwingKF..kTotalKF-1] = stance.
    double kf[kTotalKF][7] = {};
    bool kf_valid = false;  // true once keyframes are computed for this cycle
};

struct NmfStepper {
    // Tunable parameters (optimizer can fit these).
    double lambda = 0.5;              // IK regularization weight (strong prior)
    double stride_scale = 0.3;       // step length relative to velocity
    double step_height = 0.02;       // foot lift height (world units)
    double swing_apex_fwd = 0.3;     // how far forward the apex control points are (0-1)
    double adhesion_ramp_width = 0.3;
    double adhesion_max = 3.0;

    // Kuramoto oscillator state.
    double cpg_phase[6] = {};
    double cpg_freq[6] = {};
    double cpg_amplitude[6] = {};
    double cpg_target_amp[6] = {};
    double cpg_alpha = 20.0;

    // Per-leg state.
    LegState state[6];
    JointLimits limits;

    // Drive command.
    double drive_magnitude = 0.0;
    double drive_turn = 0.0;

    void Init(double freq_hz = 12.0) {
        if (!limits.valid)
            limits.ComputeFromWalkData();
        for (int i = 0; i < 6; ++i) {
            cpg_freq[i] = freq_hz;
            cpg_target_amp[i] = 0.0;
            cpg_amplitude[i] = 0.0;
            state[i].phase = LegPhase::kStance;
            state[i].sub_phase = 0.0;
            state[i].kf_valid = false;
            for (int j = 0; j < 7; ++j)
                state[i].angles[j] = WalkLerp(i, j, kPi);
        }
        cpg_phase[0] = 0.0;   cpg_phase[1] = kPi;
        cpg_phase[2] = 0.0;   cpg_phase[3] = kPi;
        cpg_phase[4] = 0.0;   cpg_phase[5] = kPi;
    }

    void SetDrive(double magnitude, double turn) {
        drive_magnitude = magnitude;
        drive_turn = turn;
        double left_amp  = std::clamp(magnitude + turn * 0.8, 0.0, 1.5);
        double right_amp = std::clamp(magnitude - turn * 0.8, 0.0, 1.5);
        cpg_target_amp[0] = left_amp;
        cpg_target_amp[1] = left_amp;
        cpg_target_amp[2] = left_amp;
        cpg_target_amp[3] = right_amp;
        cpg_target_amp[4] = right_amp;
        cpg_target_amp[5] = right_amp;
    }

    void Step(double dt) {
        // --- Advance Kuramoto oscillators ---
        double dphase[6], damp[6];
        for (int i = 0; i < 6; ++i) {
            dphase[i] = kTwoPi * cpg_freq[i];
            for (int j = 0; j < 6; ++j) {
                if (i == j) continue;
                double w = (cpg_internal::kPhaseBias[i][j] > 0.01)
                    ? cpg_internal::kCouplingWeight : 0.0;
                dphase[i] += cpg_amplitude[j] * w *
                    std::sin(cpg_phase[j] - cpg_phase[i] - cpg_internal::kPhaseBias[i][j]);
            }
            damp[i] = cpg_alpha * (cpg_target_amp[i] - cpg_amplitude[i]);
        }
        for (int i = 0; i < 6; ++i) {
            cpg_phase[i] += dphase[i] * dt;
            cpg_amplitude[i] += damp[i] * dt;
        }

        // --- Per-leg: compute keyframes once, then interpolate ---
        for (int leg = 0; leg < 6; ++leg) {
            double amp = cpg_amplitude[leg];
            if (amp < 0.01) {
                state[leg].phase = LegPhase::kStance;
                continue;
            }

            double p = cpg_phase[leg] - kTwoPi * std::floor(cpg_phase[leg] / kTwoPi);
            bool in_swing = (p < kPi);

            // Compute keyframes once when entering swing (start of cycle).
            if (in_swing && !state[leg].kf_valid) {
                ComputeCycleKeyframes(leg, amp);
                state[leg].kf_valid = true;
                state[leg].phase = LegPhase::kLift;
            }
            if (!in_swing && state[leg].kf_valid &&
                (state[leg].phase == LegPhase::kLift ||
                 state[leg].phase == LegPhase::kSwing)) {
                // Transitioned from swing to stance.
                state[leg].phase = LegPhase::kPlace;
            }
            if (in_swing && p > kPi * 0.3)
                state[leg].phase = LegPhase::kSwing;
            if (!in_swing) {
                double stance_t = (p - kPi) / kPi;
                if (stance_t > 0.1)
                    state[leg].phase = LegPhase::kStance;
            }

            // Invalidate keyframes when we wrap back to swing.
            if (!in_swing)
                state[leg].kf_valid = false;

            // --- Interpolate keyframe angles ---
            if (in_swing) {
                double t = p / kPi;
                double fi = t * (kSwingKF - 1);
                int i0 = static_cast<int>(fi);
                if (i0 >= kSwingKF - 1) i0 = kSwingKF - 2;
                double frac = fi - i0;
                // Linear interpolation between adjacent keyframes.
                for (int j = 0; j < 7; ++j) {
                    state[leg].angles[j] = state[leg].kf[i0][j]
                        + frac * (state[leg].kf[i0 + 1][j] - state[leg].kf[i0][j]);
                }
                state[leg].sub_phase = t;
            } else {
                double t = (p - kPi) / kPi;
                double fi = t * (kStanceKF - 1);
                int i0 = static_cast<int>(fi);
                if (i0 >= kStanceKF - 1) i0 = kStanceKF - 2;
                double frac = fi - i0;
                int a = kSwingKF + i0;
                for (int j = 0; j < 7; ++j) {
                    state[leg].angles[j] = state[leg].kf[a][j]
                        + frac * (state[leg].kf[a + 1][j] - state[leg].kf[a][j]);
                }
                state[leg].sub_phase = t;
            }
        }
    }

    void GetCtrl(double* ctrl) const {
        for (int leg = 0; leg < 6; ++leg) {
            for (int j = 0; j < 7; ++j)
                ctrl[leg * 7 + j] = state[leg].angles[j];

            double p = cpg_phase[leg] - kTwoPi * std::floor(cpg_phase[leg] / kTwoPi);
            double rw = adhesion_ramp_width;
            double adhesion;
            if (p < rw) {
                adhesion = 0.5 * (1.0 + std::cos(kPi * p / rw));
            } else if (p < kPi - rw) {
                adhesion = 0.0;
            } else if (p < kPi + rw) {
                adhesion = 0.5 * (1.0 - std::cos(kPi * (p - kPi + rw) / (2.0 * rw)));
            } else if (p < kTwoPi - rw) {
                adhesion = 1.0;
            } else {
                adhesion = 0.5 * (1.0 + std::cos(kPi * (p - (kTwoPi - rw)) / rw));
            }
            ctrl[42 + leg] = adhesion * adhesion_max;
        }
    }

private:
    // Compute all keyframes for one gait cycle (swing + stance).
    void ComputeCycleKeyframes(int leg, double amp) {
        // Current foot position (end of previous stance = swing start).
        FkResult fk_start = FlyFK(leg, state[leg].angles);
        Vec3 start_pos = fk_start.foot_pos;

        // Compute step target (where to land).
        Vec3 land_pos = ComputeStepTargetPos(leg, amp);

        // Compute push-off position (where stance ends).
        double sweep = stride_scale * amp * 0.1;
        Vec3 pushoff_pos = {
            land_pos.x + sweep,   // swept backward
            land_pos.y,
            land_pos.z
        };

        IkConfig cfg;
        cfg.lambda = lambda;
        cfg.step_scale = 0.8;
        cfg.max_iter = 80;

        // Use current angles as seed; chain IK solutions for continuity.
        double seed[7];
        for (int j = 0; j < 7; ++j) seed[j] = state[leg].angles[j];

        // --- Swing keyframes: cubic Bezier foot trajectory ---
        // Control points for a smooth arc with height clearance.
        Vec3 cp0 = start_pos;
        Vec3 cp1 = {
            start_pos.x + swing_apex_fwd * (land_pos.x - start_pos.x),
            start_pos.y + swing_apex_fwd * (land_pos.y - start_pos.y),
            std::max(start_pos.z, land_pos.z) + step_height
        };
        Vec3 cp2 = {
            start_pos.x + (1.0 - swing_apex_fwd) * (land_pos.x - start_pos.x),
            start_pos.y + (1.0 - swing_apex_fwd) * (land_pos.y - start_pos.y),
            std::max(start_pos.z, land_pos.z) + step_height
        };
        Vec3 cp3 = land_pos;

        for (int i = 0; i < kSwingKF; ++i) {
            double t = static_cast<double>(i) / (kSwingKF - 1);
            Vec3 foot = CubicBezierPos(cp0, cp1, cp2, cp3, t);
            IkResult ik = FlyIK(leg, foot, seed, cfg);
            if (ik.converged || ik.position_error < 0.05) {
                limits.Clamp(leg, ik.angles);
                for (int j = 0; j < 7; ++j) {
                    state[leg].kf[i][j] = ik.angles[j];
                    seed[j] = ik.angles[j];
                }
            } else {
                for (int j = 0; j < 7; ++j)
                    state[leg].kf[i][j] = seed[j];
            }
        }

        // --- Stance keyframes: straight line on ground (foot sweeps back) ---
        for (int i = 0; i < kStanceKF; ++i) {
            double t = static_cast<double>(i) / (kStanceKF - 1);
            Vec3 foot = {
                land_pos.x + t * (pushoff_pos.x - land_pos.x),
                land_pos.y + t * (pushoff_pos.y - land_pos.y),
                land_pos.z + t * (pushoff_pos.z - land_pos.z)
            };
            IkResult ik = FlyIK(leg, foot, seed, cfg);
            if (ik.converged || ik.position_error < 0.05) {
                limits.Clamp(leg, ik.angles);
                for (int j = 0; j < 7; ++j) {
                    state[leg].kf[kSwingKF + i][j] = ik.angles[j];
                    seed[j] = ik.angles[j];
                }
            } else {
                for (int j = 0; j < 7; ++j)
                    state[leg].kf[kSwingKF + i][j] = seed[j];
            }
        }
    }

    // Cubic Bezier position evaluation.
    static Vec3 CubicBezierPos(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, double t) {
        double u = 1.0 - t;
        double u2 = u * u;
        double t2 = t * t;
        double w0 = u2 * u;
        double w1 = 3.0 * u2 * t;
        double w2 = 3.0 * u * t2;
        double w3 = t2 * t;
        return {
            w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x,
            w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y,
            w0*p0.z + w1*p1.z + w2*p2.z + w3*p3.z
        };
    }

    Vec3 ComputeStepTargetPos(int leg, double amp) {
        double neutral_angles[7];
        for (int j = 0; j < 7; ++j)
            neutral_angles[j] = WalkLerp(leg, j, kPi);
        FkResult neutral_fk = FlyFK(leg, neutral_angles);

        double stride = stride_scale * amp * 0.1;
        double turn_bias = (leg < 3) ? drive_turn * 0.05 : -drive_turn * 0.05;

        return {
            neutral_fk.foot_pos.x - stride + turn_bias,
            neutral_fk.foot_pos.y,
            neutral_fk.foot_pos.z
        };
    }
};

}  // namespace nmfly
