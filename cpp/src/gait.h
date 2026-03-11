#pragma once
// Tripod gait CPG for 6-legged insect locomotion.
//
// Converts MotorCommand into per-leg joint angle targets.
// Joint order per leg (NeuroMechFly v2 naming):
//   0: ThC_yaw, 1: ThC_pitch, 2: ThC_roll, 3: CTr_pitch,
//   4: CTr_roll, 5: FTi_pitch, 6: TiTa_pitch

#include <algorithm>
#include <cmath>

#include "types.h"

namespace nmfly {

constexpr int kLegs = 6;
constexpr int kJointsPerLeg = 7;
constexpr int kTotalJoints = kLegs * kJointsPerLeg;  // 42
constexpr float kPi = 3.14159265358979f;

// Tripod phase offsets: Group A (0): L1,R2,L3. Group B (pi): R1,L2,R3.
constexpr float kTripodPhase[kLegs] = {
    0.0f, kPi, kPi, 0.0f, 0.0f, kPi
};

// Rest pose (radians) for each joint.
constexpr float kRestPose[kJointsPerLeg] = {
    0.0f,   // ThC_yaw
    0.3f,   // ThC_pitch (slightly forward)
    0.0f,   // ThC_roll
   -0.8f,   // CTr_pitch (femur down)
    0.0f,   // CTr_roll
    1.2f,   // FTi_pitch (tibia bent)
    0.5f,   // TiTa_pitch (tarsus on ground)
};

// Swing amplitude per joint during walking.
constexpr float kSwingAmplitude[kJointsPerLeg] = {
    0.10f,  // ThC_yaw   (slight lateral)
    0.40f,  // ThC_pitch (main swing, big)
    0.05f,  // ThC_roll  (minimal)
    0.20f,  // CTr_pitch (femur lift during swing)
    0.00f,  // CTr_roll
    0.15f,  // FTi_pitch (tibia extends during swing)
    0.10f,  // TiTa_pitch
};

struct GaitState {
    float phase     = 0.0f;   // gait cycle [0, 2*pi]
    float frequency = 8.0f;   // Hz
    float fwd_vel   = 0.0f;   // smoothed forward velocity
    float ang_vel   = 0.0f;   // smoothed angular velocity
    bool  frozen    = false;
    float tau       = 0.05f;  // EMA smoothing time constant (seconds)

    // Per-leg contact state for stumble correction.
    // When a swing-phase leg unexpectedly contacts ground, we briefly
    // increase its lift height to step over the obstacle.
    float stumble_lift[kLegs] = {};
    float stumble_decay = 10.0f;  // decay rate (1/s)

    void Update(const MotorCommand& cmd, float dt) {
        float alpha = 1.0f - std::exp(-dt / tau);
        fwd_vel += alpha * (cmd.forward_velocity - fwd_vel);
        ang_vel += alpha * (cmd.angular_velocity - ang_vel);
        frozen = cmd.freeze > 0.5f;

        if (frozen) return;

        // Phase rate proportional to forward speed.
        float speed = std::abs(fwd_vel);
        frequency = std::clamp(speed * 0.4f, 2.0f, 15.0f);
        float direction = (fwd_vel >= 0.0f) ? 1.0f : -1.0f;

        phase += direction * 2.0f * kPi * frequency * dt;
        phase = std::fmod(phase, 2.0f * kPi);
        if (phase < 0.0f) phase += 2.0f * kPi;

        // Decay stumble lift.
        for (int i = 0; i < kLegs; ++i) {
            stumble_lift[i] *= std::max(0.0f, 1.0f - stumble_decay * dt);
        }
    }

    // Call when a swing-phase leg unexpectedly touches ground.
    void TriggerStumble(int leg_idx) {
        if (leg_idx >= 0 && leg_idx < kLegs) {
            stumble_lift[leg_idx] = 1.0f;
        }
    }
};

// Compute joint targets from gait state. Output: targets[leg][joint].
inline void ComputeJointTargets(const GaitState& gait,
                                float targets[kLegs][kJointsPerLeg]) {
    // Start from rest pose.
    for (int leg = 0; leg < kLegs; ++leg)
        for (int j = 0; j < kJointsPerLeg; ++j)
            targets[leg][j] = kRestPose[j];

    if (gait.frozen) return;

    float speed_scale = std::min(1.0f, std::abs(gait.fwd_vel) / 30.0f);

    for (int leg = 0; leg < kLegs; ++leg) {
        float leg_phase = gait.phase + kTripodPhase[leg];
        float swing = std::sin(leg_phase);
        float lift  = std::max(0.0f, std::sin(leg_phase));

        // Stumble correction: extra lift.
        lift += gait.stumble_lift[leg] * 0.5f;

        // Turn bias: reduce stride on inner side, increase on outer.
        bool is_left = (leg % 2 == 0);
        float turn_bias = 1.0f;
        if (gait.ang_vel > 0.1f) {       // turning left
            turn_bias = is_left ? 0.5f : 1.5f;
        } else if (gait.ang_vel < -0.1f) {  // turning right
            turn_bias = is_left ? 1.5f : 0.5f;
        }

        for (int j = 0; j < kJointsPerLeg; ++j) {
            float amp = kSwingAmplitude[j] * speed_scale * turn_bias;

            if (j == 1) {        // ThC_pitch: main swing
                targets[leg][j] += amp * swing;
            } else if (j == 3) { // CTr_pitch: lift during swing
                targets[leg][j] += amp * lift;
            } else if (j == 5) { // FTi_pitch: extend during swing
                targets[leg][j] -= amp * lift * 0.5f;
            } else {
                targets[leg][j] += amp * swing * 0.3f;
            }
        }
    }
}

// Detect which legs are in swing phase (should be in the air).
inline bool IsSwingPhase(const GaitState& gait, int leg_idx) {
    float leg_phase = gait.phase + kTripodPhase[leg_idx];
    return std::sin(leg_phase) > 0.0f;
}

// Wing state for flight mode.
struct FlightState {
    float phase     = 0.0f;
    float frequency = 200.0f;  // Hz (Drosophila wing beat)
    float amplitude = 2.5f;    // radians, full stroke
    float roll_bias = 0.0f;    // asymmetric for turning

    void Update(const MotorCommand& cmd, float dt) {
        phase += 2.0f * kPi * frequency * dt;
        phase = std::fmod(phase, 2.0f * kPi);

        float alpha = 1.0f - std::exp(-dt / 0.02f);
        roll_bias += alpha * (cmd.angular_velocity * 0.1f - roll_bias);
    }

    void WingAngles(float& left, float& right) const {
        float base = std::sin(phase) * amplitude;
        left  = base + roll_bias;
        right = base - roll_bias;
    }
};

}  // namespace nmfly
