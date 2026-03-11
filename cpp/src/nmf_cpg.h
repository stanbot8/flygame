#pragma once
// Kuramoto-coupled CPG for NeuroMechFly tripod walking.
// Vendored from flygym's CPG controller.

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include "nmf_walk_data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nmfly {

// Tripod phase biases: legs in same tripod group sync (0),
// legs in different groups anti-phase (pi).
// Leg order: LF=0, LM=1, LH=2, RF=3, RM=4, RH=5
// Tripod A: LF(0), LH(2), RM(4)  — same phase
// Tripod B: LM(1), RF(3), RH(5)  — opposite phase
constexpr double kPhaseBias[6][6] = {
    {0,    M_PI, 0,    M_PI, 0,    M_PI}, // LF
    {M_PI, 0,    M_PI, 0,    M_PI, 0   }, // LM
    {0,    M_PI, 0,    M_PI, 0,    M_PI}, // LH
    {M_PI, 0,    M_PI, 0,    M_PI, 0   }, // RF
    {0,    M_PI, 0,    M_PI, 0,    M_PI}, // RM
    {M_PI, 0,    M_PI, 0,    M_PI, 0   }, // RH
};

// Coupling weight: 10 for coupled pairs (those with pi bias).
constexpr double kCouplingWeight = 10.0;

struct NmfCpg {
    double phase[6]     = {};  // current phase per leg [rad]
    double amplitude[6] = {};  // current amplitude per leg
    double freq[6]      = {};  // intrinsic frequency [Hz]
    double target_amp[6]= {};  // target amplitude
    double alpha         = 20.0; // amplitude convergence rate [1/s]

    void Init(double freq_hz = 12.0, double amp = 1.0) {
        // Tripod A starts at 0, Tripod B starts at pi.
        for (int i = 0; i < 6; ++i) {
            freq[i] = freq_hz;
            target_amp[i] = amp;
            amplitude[i] = 0.0;  // ramp up from zero
        }
        // Initial phases for tripod gait.
        phase[0] = 0.0;       // LF — Tripod A
        phase[1] = M_PI;      // LM — Tripod B
        phase[2] = 0.0;       // LH — Tripod A
        phase[3] = M_PI;      // RF — Tripod B
        phase[4] = 0.0;       // RM — Tripod A
        phase[5] = M_PI;      // RH — Tripod B
    }

    // Advance CPG by dt seconds (Euler integration of Kuramoto oscillators).
    void Step(double dt) {
        double dphase[6], damp[6];
        for (int i = 0; i < 6; ++i) {
            // Intrinsic frequency term.
            dphase[i] = 2.0 * M_PI * freq[i];
            // Coupling term: sum over neighbors.
            for (int j = 0; j < 6; ++j) {
                if (i == j) continue;
                double w = (kPhaseBias[i][j] > 0.01) ? kCouplingWeight : 0.0;
                dphase[i] += amplitude[j] * w *
                    std::sin(phase[j] - phase[i] - kPhaseBias[i][j]);
            }
            // Amplitude convergence.
            damp[i] = alpha * (target_amp[i] - amplitude[i]);
        }
        for (int i = 0; i < 6; ++i) {
            phase[i] += dphase[i] * dt;
            amplitude[i] += damp[i] * dt;
        }
    }

    // Set walking speed: magnitude [0, 1.5] and turning [-1, 1].
    // Positive turn = turn right (left legs faster).
    // Turn coupling 0.8 matches flygym's demo range (1.2 vs 0.4 at full turn).
    void SetDrive(double magnitude, double turn) {
        double left_amp  = std::clamp(magnitude + turn * 0.8, 0.0, 1.5);
        double right_amp = std::clamp(magnitude - turn * 0.8, 0.0, 1.5);
        // Left legs: 0(LF), 1(LM), 2(LH)
        target_amp[0] = left_amp;
        target_amp[1] = left_amp;
        target_amp[2] = left_amp;
        // Right legs: 3(RF), 4(RM), 5(RH)
        target_amp[3] = right_amp;
        target_amp[4] = right_amp;
        target_amp[5] = right_amp;
    }

    // Get joint ctrl values for all 48 actuators.
    // ctrl[0..41] = joint angles, ctrl[42..47] = adhesion on/off.
    //
    // Matches flygym's PreprogrammedSteps.get_joint_angles exactly:
    //   neutral = trajectory(pi)
    //   ctrl = neutral + amplitude * (trajectory(phase) - neutral)
    // When amplitude=1, ctrl = trajectory(phase) (the full recorded walk).
    // When amplitude=0, ctrl = trajectory(pi) (neutral standing pose).
    void GetCtrl(double ctrl[48]) const {
        constexpr double kTwoPi = 2.0 * M_PI;
        for (int leg = 0; leg < 6; ++leg) {
            // Wrap phase to [0, 2pi).
            double p = phase[leg] - kTwoPi * std::floor(phase[leg] / kTwoPi);
            double amp = amplitude[leg];
            for (int j = 0; j < 7; ++j) {
                double neutral = WalkLerp(leg, j, M_PI);
                double current = WalkLerp(leg, j, p);
                ctrl[leg * 7 + j] = neutral + amp * (current - neutral);
            }
            // Adhesion: ON during stance, OFF during swing.
            // flygym uses boolean (0 or 1); we add a smooth ramp
            // to avoid discontinuous contact forces.
            double swing_end = M_PI;  // swing is [0, pi), stance is [pi, 2pi)
            bool in_stance = (p >= swing_end);
            ctrl[42 + leg] = in_stance ? 1.0 : 0.0;
        }
    }
};

}  // namespace nmfly
