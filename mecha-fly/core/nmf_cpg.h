#pragma once
// Kuramoto-coupled CPG for NeuroMechFly tripod walking.
// Vendored from flygym's CPG controller.

#include <algorithm>
#include <cmath>
#include <cstring>
#include "nmf_walk_data.h"

namespace nmfly {

// Tripod phase biases: legs in same tripod group sync (0),
// legs in different groups anti-phase (pi).
// Leg order: LF=0, LM=1, LH=2, RF=3, RM=4, RH=5
// Tripod A: LF(0), LH(2), RM(4)  — same phase
// Tripod B: LM(1), RF(3), RH(5)  — opposite phase
constexpr double kPhaseBias[6][6] = {
    {0,    kPi, 0,    kPi, 0,    kPi}, // LF
    {kPi, 0,    kPi, 0,    kPi, 0   }, // LM
    {0,    kPi, 0,    kPi, 0,    kPi}, // LH
    {kPi, 0,    kPi, 0,    kPi, 0   }, // RF
    {0,    kPi, 0,    kPi, 0,    kPi}, // RM
    {kPi, 0,    kPi, 0,    kPi, 0   }, // RH
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
        phase[1] = kPi;      // LM — Tripod B
        phase[2] = 0.0;       // LH — Tripod A
        phase[3] = kPi;      // RF — Tripod B
        phase[4] = 0.0;       // RM — Tripod A
        phase[5] = kPi;      // RH — Tripod B
    }

    // Advance CPG by dt seconds (Euler integration of Kuramoto oscillators).
    void Step(double dt) {
        double dphase[6], damp[6];
        for (int i = 0; i < 6; ++i) {
            // Intrinsic frequency term.
            dphase[i] = kTwoPi * freq[i];
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

    // Get joint ctrl values for leg and adhesion actuators.
    // ctrl[0..41] = joint angles, ctrl[42..47] = adhesion on/off.
    // (Wing actuators at 42..47 are handled separately by NmfWing.)
    //
    // Matches flygym's PreprogrammedSteps.get_joint_angles exactly:
    //   neutral = trajectory(pi)
    //   ctrl = neutral + amplitude * (trajectory(phase) - neutral)
    // When amplitude=1, ctrl = trajectory(phase) (the full recorded walk).
    // When amplitude=0, ctrl = trajectory(pi) (neutral standing pose).
    void GetCtrl(double* ctrl) const {
        for (int leg = 0; leg < 6; ++leg) {
            // Wrap phase to [0, 2pi).
            double p = phase[leg] - kTwoPi * std::floor(phase[leg] / kTwoPi);
            double amp = amplitude[leg];
            for (int j = 0; j < 7; ++j) {
                double neutral = WalkLerp(leg, j, kPi);
                double current = WalkLerp(leg, j, p);
                ctrl[leg * 7 + j] = neutral + amp * (current - neutral);
            }
            // Adhesion: ON during stance, OFF during swing.
            // Drosophila tarsal pads use wet adhesion (van der Waals +
            // capillary; Niederegger & Gorb 2003). With gainprm=40 in the
            // XML, ctrl=3 gives 120 force units per leg, enough to pin
            // stance legs without torquing the body over. ctrl=20 flipped
            // the fly; ctrl=1 (flygym default) allowed too much bounce.
            // Smooth cosine ramp avoids discontinuous contact forces.
            constexpr double kRampWidth = 0.3;    // ~14 deg ramp zone
            constexpr double kAdhesionMax = 3.0;   // stance adhesion strength
            double adhesion;
            if (p < kRampWidth) {
                // Ramp down from stance->swing at phase 0
                adhesion = 0.5 * (1.0 + std::cos(kPi * p / kRampWidth));
            } else if (p < kPi - kRampWidth) {
                // Mid-swing: fully off
                adhesion = 0.0;
            } else if (p < kPi + kRampWidth) {
                // Ramp up from swing->stance at phase pi
                adhesion = 0.5 * (1.0 - std::cos(kPi * (p - kPi + kRampWidth) / (2.0 * kRampWidth)));
            } else if (p < kTwoPi - kRampWidth) {
                // Mid-stance: fully on
                adhesion = 1.0;
            } else {
                // Ramp down from stance->swing at phase 2pi
                adhesion = 0.5 * (1.0 + std::cos(kPi * (p - (kTwoPi - kRampWidth)) / kRampWidth));
            }
            ctrl[42 + leg] = adhesion * kAdhesionMax;
        }
    }
};

}  // namespace nmfly
