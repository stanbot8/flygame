#pragma once
// mujoco-game core types. Framework-agnostic: no dependency on MuJoCo,
// network protocol, or any specific spiking simulator.

#include <cstdint>

namespace mjgame {

constexpr int kMaxJoints   = 64;
constexpr int kMaxContacts = 8;

struct MotorCommand {
    float forward_velocity = 0.0f;  // mm/s (positive = forward)
    float angular_velocity = 0.0f;  // rad/s (positive = left turn)
    float approach_drive   = 0.0f;  // positive = approach stimulus
    float freeze           = 0.0f;  // [0, 1], 1 = stop all motion
};

struct SensoryReading {
    uint32_t channel    = 0;
    float    activation = 0.0f;  // [0, 1] normalized
    float    raw_value  = 0.0f;  // original physical value
};

// Body state returned by the simulation each step.
// Uses generous maximums — each animal sets n_joints / n_contacts
// to its actual count.
struct BodyState {
    float joint_angles[kMaxJoints]     = {};  // radians
    float joint_velocities[kMaxJoints] = {};  // rad/s
    float contacts[kMaxContacts]       = {};  // per-limb ground contact [0,1]
    float body_velocity[3]             = {};  // [fwd mm/s, lat mm/s, yaw rad/s]
    float position[3]                  = {};  // world position [x, y, z] meters
    float heading                      = 0.0f; // yaw angle radians
    int   step                         = 0;
    float sim_time                     = 0.0f; // seconds
    int   n_joints                     = 0;    // actual joint count
    int   n_contacts                   = 0;    // actual contact count
};

}  // namespace mjgame
