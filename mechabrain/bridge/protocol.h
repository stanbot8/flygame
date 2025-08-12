#ifndef FWMC_BRIDGE_PROTOCOL_H_
#define FWMC_BRIDGE_PROTOCOL_H_

// FWMC Bridge Protocol v1
//
// Binary TCP protocol for communication between a brain simulator
// (spiking connectome) and a body simulator (physics engine).
//
// Wire format:
//   Each message is [uint32 type][uint32 payload_size][payload...]
//   All integers are little-endian. All floats are IEEE 754.
//
// Connection flow:
//   1. Body sim connects to brain sim on TCP port (default 9100)
//   2. Body sim sends MSG_HELLO with protocol version
//   3. Brain sim replies MSG_HELLO with its protocol version
//   4. Main loop: body sends readings, brain sends commands
//
// Message types (body → brain):
//   0x00  HELLO          uint32 version (currently 1)
//   0x01  BIO_READINGS   BioReading[]  (neural/sensory data)
//   0x02  CONFIG         JSON string
//   0x03  PING           empty
//   0x04  BODY_STATE     BodyState     (proprioception, contacts)
//
// Message types (brain → body):
//   0x80  HELLO          uint32 version
//   0x81  STIM_COMMANDS  StimCommand[] (optogenetic stimulation)
//   0x82  STATUS         JSON string   (legacy motor command)
//   0x83  PONG           empty
//   0x84  MOTOR          MotorCommand  (binary motor command)
//
// Implementors: to build a body sim that connects to FWMC, you only
// need this header. Implement a TCP client that:
//   1. Connects and exchanges HELLO
//   2. Sends BODY_STATE every N steps
//   3. Receives MOTOR commands
//   4. Optionally sends BIO_READINGS for neural feedback

#include <cstdint>
#include <cstring>

namespace mechabrain {
namespace protocol {

constexpr uint32_t kVersion = 1;
constexpr uint16_t kDefaultPort = 9100;

// Message types
enum MsgType : uint32_t {
  // Body → Brain
  kHelloClient  = 0x00,
  kBioReadings  = 0x01,
  kConfig       = 0x02,
  kPing         = 0x03,
  kBodyState    = 0x04,

  // Brain → Body
  kHelloServer  = 0x80,
  kStimCommands = 0x81,
  kStatus       = 0x82,  // JSON motor command (legacy)
  kPong         = 0x83,
  kMotor        = 0x84,  // Binary motor command (single)
  kMotorBatch   = 0x85,  // Binary motor command batch (N commands)
  kBrainFrame   = 0x86,  // Brain visualization frame (RGBA thumbnail)
};

// Message header (8 bytes)
#pragma pack(push, 1)
struct Header {
  uint32_t type;
  uint32_t payload_size;
};

// Neural reading from biological or simulated tissue.
struct BioReading {
  uint32_t neuron_idx;
  float    spike_prob;   // [0, 1] deconvolved firing probability
  float    calcium_raw;  // raw dF/F signal
  float    voltage_mv;   // voltage if available, else NaN
};
static_assert(sizeof(BioReading) == 16, "BioReading must be 16 bytes");

// Optogenetic stimulation command.
struct StimCommand {
  uint32_t neuron_idx;
  float    intensity;    // [0, 1] normalized laser power
  uint8_t  excitatory;   // 1 = activation, 0 = inhibition
  float    duration_ms;  // pulse duration
};
static_assert(sizeof(StimCommand) == 13, "StimCommand must be 13 bytes");

// Motor command from brain to body.
struct MotorCommand {
  float forward_velocity;  // mm/s (positive = forward)
  float angular_velocity;  // rad/s (positive = left turn)
  float approach_drive;    // positive = approach, negative = avoid
  float freeze;            // [0, 1] freeze probability
};
static_assert(sizeof(MotorCommand) == 16, "MotorCommand must be 16 bytes");

// Body state from body sim to brain (proprioception + contacts).
struct BodyState {
  float joint_angles[42];      // 6 legs x 7 joints, radians
  float joint_velocities[42];  // rad/s
  float contacts[6];           // per-leg ground contact [0, 1]
  float body_velocity[3];      // [fwd mm/s, lat mm/s, yaw rad/s]
  float position[3];           // world [x, y, z] mm
  float heading;               // yaw angle, radians
  float sim_time;              // seconds since start
  uint32_t step;               // simulation step count
};
static_assert(sizeof(BodyState) == 396, "BodyState must be 396 bytes");

#pragma pack(pop)

// Helper: build a message buffer (header + payload).
inline void PackMessage(uint8_t* buf, uint32_t type,
                        const void* payload, uint32_t size) {
  Header hdr;
  hdr.type = type;
  hdr.payload_size = size;
  std::memcpy(buf, &hdr, sizeof(hdr));
  if (size > 0 && payload)
    std::memcpy(buf + sizeof(hdr), payload, size);
}

}  // namespace protocol
}  // namespace mechabrain

#endif  // FWMC_BRIDGE_PROTOCOL_H_
