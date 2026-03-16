#pragma once
// Hybrid locomotion controller for NeuroMechFly.
// Ported from flygym's hybrid_controller.py and rule_based_controller.py.
//
// Architecture: CPG (Kuramoto) → cubic spline walk data → correction vectors.
// Same GetCtrl(double* ctrl) interface as NmfCpg — drop-in replacement.
//
// Corrections (from flygym):
//   - Retraction: lifts leg when foot z drops too low (hole in terrain)
//   - Stumbling: retracts leg on horizontal contact during swing
// Both use small per-joint correction vectors, NOT full IK.

#include <algorithm>
#include <cmath>
#include <cstring>
#include "nmf_walk_spline.h"

namespace nmfly {

// ── Constants ───────────────────────────────────────────────────────────────

constexpr double kHybPi    = 3.14159265358979323846;
constexpr double kHybTwoPi = 2.0 * kHybPi;

// ── Cubic spline interpolation (periodic) ───────────────────────────────────

// Precomputed cubic spline coefficients for one joint trajectory.
struct CubicCoeffs {
    // For interval [i, i+1]: y(t) = a + b*t + c*t^2 + d*t^3
    // where t = (phase - phase[i]) / (phase[i+1] - phase[i])
    double a[kWalkSplineSamples];
    double b[kWalkSplineSamples];
    double c[kWalkSplineSamples];
    double d[kWalkSplineSamples];
};

// Build periodic cubic spline coefficients for a single joint.
// Natural (periodic) cubic spline via tridiagonal solve.
inline CubicCoeffs BuildSplineCoeffs(const double y[kWalkSplineSamples]) {
    constexpr int N = kWalkSplineSamples;
    constexpr int M = N - 1;  // number of intervals (last point == first)
    CubicCoeffs co = {};

    // Uniform spacing
    double h = kWalkPhase[1] - kWalkPhase[0];

    // Solve for second derivatives (m[i]) with periodic boundary.
    // For uniform spacing: m[i-1] + 4*m[i] + m[i+1] = 6/h^2 * (y[i-1] - 2*y[i] + y[i+1])
    // With periodic wrap: index modulo M (since y[0]==y[M]).
    double m[N] = {};
    // Use iterative method (SOR) — simple and robust for 45 points.
    for (int iter = 0; iter < 200; ++iter) {
        double max_change = 0.0;
        for (int i = 0; i < M; ++i) {
            int ip = (i + 1) % M;
            int im = (i - 1 + M) % M;
            double rhs = (6.0 / (h * h)) * (y[im] - 2.0 * y[i] + y[ip]);
            double new_m = (rhs - m[im] - m[ip]) / 4.0;
            double change = std::abs(new_m - m[i]);
            if (change > max_change) max_change = change;
            m[i] = new_m;
        }
        m[M] = m[0];  // periodic
        if (max_change < 1e-12) break;
    }

    // Build cubic coefficients for each interval.
    for (int i = 0; i < M; ++i) {
        int ip = i + 1;
        co.a[i] = y[i];
        co.b[i] = (y[ip] - y[i]) / h - h * (2.0 * m[i] + m[ip]) / 6.0;
        co.c[i] = m[i] / 2.0;
        co.d[i] = (m[ip] - m[i]) / (6.0 * h);
    }
    return co;
}

// Evaluate cubic spline at given phase [0, 2*pi].
inline double SplineEval(const CubicCoeffs& co, double phase) {
    // Wrap to [0, 2*pi)
    phase = phase - kHybTwoPi * std::floor(phase / kHybTwoPi);
    double h = kWalkPhase[1] - kWalkPhase[0];
    int i = static_cast<int>(phase / h);
    if (i >= kWalkSplineSamples - 1) i = kWalkSplineSamples - 2;
    double t = phase - kWalkPhase[i];
    return co.a[i] + t * (co.b[i] + t * (co.c[i] + t * co.d[i]));
}

// ── Pre-built spline table (all legs, all joints) ───────────────────────────

struct WalkSplineTable {
    CubicCoeffs joint[6][7];
    bool ready = false;

    void Build() {
        for (int leg = 0; leg < 6; ++leg)
            for (int j = 0; j < 7; ++j)
                joint[leg][j] = BuildSplineCoeffs(kWalkSplineData[leg][j]);
        ready = true;
    }

    double Eval(int leg, int joint_idx, double phase) const {
        return SplineEval(this->joint[leg][joint_idx], phase);
    }
};

// ── Hybrid correction vectors (from flygym) ─────────────────────────────────

// Per-leg-position correction vectors: applied when retraction or stumbling
// corrections are triggered. Values from flygym hybrid_controller.py.
// Index: [leg_position][joint], where leg_position: 0=Front, 1=Mid, 2=Hind.
// Joint order: Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1.
constexpr double kCorrectionVec[3][7] = {
    {-0.03,   0.0,   0.0,  -0.03,  0.0,   0.03,  0.03 },  // Front (LF, RF)
    {-0.015,  0.001, 0.025,-0.02,  0.0,  -0.02,  0.0  },  // Mid   (LM, RM)
    { 0.0,    0.0,   0.0,  -0.02,  0.0,   0.01, -0.02 },  // Hind  (LH, RH)
};

// Map leg index → correction vector index.
// LF=0→Front(0), LM=1→Mid(1), LH=2→Hind(2), RF=3→Front(0), RM=4→Mid(1), RH=5→Hind(2)
constexpr int kLegPosition[6] = {0, 1, 2, 0, 1, 2};

// ── Phase biases (Kuramoto coupling) ────────────────────────────────────────

constexpr double kPhaseBiasHyb[6][6] = {
    {0,       kHybPi, 0,       kHybPi, 0,       kHybPi},
    {kHybPi, 0,       kHybPi, 0,       kHybPi, 0      },
    {0,       kHybPi, 0,       kHybPi, 0,       kHybPi},
    {kHybPi, 0,       kHybPi, 0,       kHybPi, 0      },
    {0,       kHybPi, 0,       kHybPi, 0,       kHybPi},
    {kHybPi, 0,       kHybPi, 0,       kHybPi, 0      },
};

constexpr double kCouplingWeightHyb = 10.0;

// ── Sensor input structure ──────────────────────────────────────────────────

struct HybridSensors {
    double end_effector_z[6];     // foot z position per leg (world frame)
    double fly_z;                 // body z position
    double contact_force_fwd[6];  // forward-projected contact force per leg (mN)
    bool   leg_contact[6];        // binary ground contact per leg
};

// ── Hybrid controller ───────────────────────────────────────────────────────

struct NmfHybrid {
    // CPG state
    double phase[6]      = {};
    double amplitude[6]  = {};
    double freq[6]       = {};
    double target_amp[6] = {};
    double alpha          = 20.0;  // amplitude convergence rate [1/s]

    // Correction state
    double retraction[6]  = {};  // retraction magnitude per leg [rad]
    double stumbling[6]   = {};  // stumbling correction magnitude per leg [rad]

    // Spline table
    WalkSplineTable spline;

    // Correction parameters (from flygym)
    double retract_lift_rate    = 800.0;   // rad/s
    double retract_lower_rate   = 700.0;   // rad/s
    double stumble_activate_rate = 2200.0; // rad/s
    double stumble_deactivate_rate = 1800.0; // rad/s
    double retract_z_threshold  = 0.05;    // mm below neighbor mean to trigger
    double stumble_force_threshold = -1.0; // mN (negative = against motion)

    void Init(double freq_hz = 12.0, double amp = 1.0) {
        spline.Build();
        for (int i = 0; i < 6; ++i) {
            freq[i] = freq_hz;
            target_amp[i] = amp;
            amplitude[i] = 0.0;
            retraction[i] = 0.0;
            stumbling[i] = 0.0;
        }
        // Tripod initial phases
        phase[0] = 0.0;        // LF — Tripod A
        phase[1] = kHybPi;     // LM — Tripod B
        phase[2] = 0.0;        // LH — Tripod A
        phase[3] = kHybPi;     // RF — Tripod B
        phase[4] = 0.0;        // RM — Tripod A
        phase[5] = kHybPi;     // RH — Tripod B
    }

    // Advance CPG oscillators (same as NmfCpg).
    void Step(double dt) {
        double dphase[6], damp[6];
        for (int i = 0; i < 6; ++i) {
            dphase[i] = kHybTwoPi * freq[i];
            for (int j = 0; j < 6; ++j) {
                if (i == j) continue;
                double w = (kPhaseBiasHyb[i][j] > 0.01) ? kCouplingWeightHyb : 0.0;
                dphase[i] += amplitude[j] * w *
                    std::sin(phase[j] - phase[i] - kPhaseBiasHyb[i][j]);
            }
            damp[i] = alpha * (target_amp[i] - amplitude[i]);
        }
        for (int i = 0; i < 6; ++i) {
            phase[i] += dphase[i] * dt;
            amplitude[i] += damp[i] * dt;
        }
    }

    // Update corrections from sensor input.
    void UpdateCorrections(double dt, const HybridSensors& sens) {
        for (int leg = 0; leg < 6; ++leg) {
            double p = phase[leg] - kHybTwoPi * std::floor(phase[leg] / kHybTwoPi);

            // Is this leg in swing phase?
            double sw_start = kWalkSwingPeriod[leg][0];
            double sw_end   = kWalkSwingPeriod[leg][1];
            bool in_swing = (sw_start <= sw_end)
                ? (p >= sw_start && p <= sw_end)
                : (p >= sw_start || p <= sw_end);

            // ── Retraction correction ──
            // Check if foot is too low relative to body.
            double rel_z = sens.fly_z - sens.end_effector_z[leg];
            if (rel_z > retract_z_threshold && in_swing) {
                retraction[leg] += retract_lift_rate * dt;
            } else {
                retraction[leg] -= retract_lower_rate * dt;
            }
            retraction[leg] = std::clamp(retraction[leg], 0.0, 1.5);

            // ── Stumbling correction ──
            // Only during swing: retract if horizontal contact.
            if (in_swing && sens.contact_force_fwd[leg] < stumble_force_threshold) {
                stumbling[leg] += stumble_activate_rate * dt;
            } else {
                stumbling[leg] -= stumble_deactivate_rate * dt;
            }
            stumbling[leg] = std::clamp(stumbling[leg], 0.0, 1.5);
        }
    }

    void SetDrive(double magnitude, double turn) {
        double left_amp  = std::clamp(magnitude + turn * 0.8, 0.0, 1.5);
        double right_amp = std::clamp(magnitude - turn * 0.8, 0.0, 1.5);
        target_amp[0] = left_amp;
        target_amp[1] = left_amp;
        target_amp[2] = left_amp;
        target_amp[3] = right_amp;
        target_amp[4] = right_amp;
        target_amp[5] = right_amp;
    }

    // Get joint ctrl values for leg and adhesion actuators.
    // ctrl[0..41] = joint angles, ctrl[42..47] = adhesion.
    // (Wing actuators at 42..47 are handled separately by NmfWing.)
    //
    // Uses cubic spline interpolation of flygym walk data,
    // with optional hybrid corrections layered on top.
    void GetCtrl(double* ctrl) const {
        for (int leg = 0; leg < 6; ++leg) {
            double p = phase[leg] - kHybTwoPi * std::floor(phase[leg] / kHybTwoPi);
            double amp = amplitude[leg];
            int pos = kLegPosition[leg];

            for (int j = 0; j < 7; ++j) {
                // Cubic spline interpolation of walk data.
                double neutral = kWalkNeutralPose[leg][j];
                double current = spline.Eval(leg, j, p);
                double angle = neutral + amp * (current - neutral);

                // Apply hybrid corrections (additive).
                double corr = (retraction[leg] + stumbling[leg]) * kCorrectionVec[pos][j];
                angle += corr;

                ctrl[leg * 7 + j] = angle;
            }

            // Adhesion: ON during stance, OFF during swing.
            constexpr double kRampWidth = 0.3;
            constexpr double kAdhesionMax = 3.0;
            double adhesion;
            if (p < kRampWidth) {
                adhesion = 0.5 * (1.0 + std::cos(kHybPi * p / kRampWidth));
            } else if (p < kHybPi - kRampWidth) {
                adhesion = 0.0;
            } else if (p < kHybPi + kRampWidth) {
                adhesion = 0.5 * (1.0 - std::cos(kHybPi * (p - kHybPi + kRampWidth) / (2.0 * kRampWidth)));
            } else if (p < kHybTwoPi - kRampWidth) {
                adhesion = 1.0;
            } else {
                adhesion = 0.5 * (1.0 + std::cos(kHybPi * (p - (kHybTwoPi - kRampWidth)) / kRampWidth));
            }
            ctrl[42 + leg] = adhesion * kAdhesionMax;
        }
    }
};

}  // namespace nmfly
