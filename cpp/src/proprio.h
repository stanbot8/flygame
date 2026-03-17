#pragma once
// Proprioception encoder: maps body state to normalized sensory activations.
//
// Channel layout (matches Python nmfly.proprioception):
//   0..41   joint angle sensors (6 legs x 7 joints)
//  42..83   joint velocity sensors
//  84..89   leg contact sensors (binary ground contact)
//  90..92   body velocity (forward, lateral, angular)
//  93..94   wing angle sensors (left, right)

#include <cmath>
#include <cstring>

#include "types.h"  // mjgame::BodyState, mjgame::SensoryReading

namespace nmfly {

using mjgame::BodyState;
using mjgame::SensoryReading;

constexpr int kJointAngleStart = 0;
constexpr int kJointVelStart   = 42;
constexpr int kContactStart    = 84;
constexpr int kBodyVelStart    = 90;
constexpr int kWingAngleStart  = 93;
constexpr int kTotalChannels   = 95;

struct ProprioConfig {
    float angle_gain    = 0.3f;   // radians -> activation
    float velocity_gain = 0.1f;   // rad/s -> activation
    float contact_prob  = 0.8f;   // activation when leg is on ground
    float body_vel_gain = 0.05f;  // mm/s -> activation
};

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x + 2.0f));
}

// Encode body state into sensory readings array.
// Returns number of readings written.
inline int EncodeProprioception(
    const BodyState& state,
    SensoryReading* out,
    int max_out,
    const ProprioConfig& cfg = {})
{
    int n = 0;
    auto emit = [&](uint32_t ch, float activation, float raw) {
        if (n < max_out) {
            out[n].channel    = ch;
            out[n].activation = activation;
            out[n].raw_value  = raw;
            ++n;
        }
    };

    // Joint angles.
    for (int i = 0; i < 42; ++i) {
        float v = state.joint_angles[i];
        emit(kJointAngleStart + i, Sigmoid(std::abs(v) * cfg.angle_gain), v);
    }

    // Joint velocities.
    for (int i = 0; i < 42; ++i) {
        float v = state.joint_velocities[i];
        emit(kJointVelStart + i, Sigmoid(std::abs(v) * cfg.velocity_gain), v);
    }

    // Contact sensors.
    for (int i = 0; i < 6; ++i) {
        float v = state.contacts[i];
        float act = (v > 0.01f) ? cfg.contact_prob : 0.02f;
        emit(kContactStart + i, act, v);
    }

    // Body velocity.
    for (int i = 0; i < 3; ++i) {
        float v = state.body_velocity[i];
        emit(kBodyVelStart + i, Sigmoid(std::abs(v) * cfg.body_vel_gain), v);
    }

    return n;
}

}  // namespace nmfly
