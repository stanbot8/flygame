#pragma once
// Fly-specific TCP extensions: body state wire format for the FWMC protocol.

#include <cstdint>
#include <cstring>

#include "tcp.h"    // mjgame::TcpClient
#include "types.h"  // mjgame::BodyState

namespace nmfly {

// Fly body state wire format (396 bytes, matches fwmc protocol::BodyState).
#pragma pack(push, 1)
struct WireBodyState {
    float    joint_angles[42];
    float    joint_velocities[42];
    float    contacts[6];
    float    body_velocity[3];
    float    position[3];
    float    heading;
    float    sim_time;
    uint32_t step;
};
static_assert(sizeof(WireBodyState) == 396);
#pragma pack(pop)

inline bool SendFlyBodyState(mjgame::TcpClient& tcp,
                             const mjgame::BodyState& bs) {
    WireBodyState w = {};
    std::memcpy(w.joint_angles, bs.joint_angles, sizeof(w.joint_angles));
    std::memcpy(w.joint_velocities, bs.joint_velocities,
                sizeof(w.joint_velocities));
    std::memcpy(w.contacts, bs.contacts, 6 * sizeof(float));
    std::memcpy(w.body_velocity, bs.body_velocity, sizeof(w.body_velocity));
    std::memcpy(w.position, bs.position, sizeof(w.position));
    w.heading  = bs.heading;
    w.sim_time = bs.sim_time;
    w.step     = static_cast<uint32_t>(bs.step);
    return tcp.SendBodyStateRaw(&w, sizeof(w));
}

inline bool ExchangeFlyBodyState(mjgame::TcpClient& tcp,
                                 const mjgame::BodyState& bs,
                                 mjgame::MotorCommand& motor_out) {
    if (!SendFlyBodyState(tcp, bs)) return false;
    return tcp.RecvMotor(motor_out);
}

}  // namespace nmfly
