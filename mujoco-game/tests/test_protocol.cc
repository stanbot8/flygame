// mujoco-game protocol and types tests.
// Tests pure functions and data structures that don't require MuJoCo runtime.

#include "types.h"
#include "tcp.h"
#include "test_harness.h"

// ---- types.h tests ----

TEST(constants) {
    assert(kMaxJoints == 64);
    assert(kMaxContacts == 8);
}

TEST(motor_command_defaults) {
    MotorCommand cmd{};
    assert(cmd.forward_velocity == 0.0f);
    assert(cmd.angular_velocity == 0.0f);
    assert(cmd.approach_drive == 0.0f);
    assert(cmd.freeze == 0.0f);
}

TEST(sensory_reading_defaults) {
    SensoryReading sr{};
    assert(sr.channel == 0);
    assert(sr.activation == 0.0f);
    assert(sr.raw_value == 0.0f);
}

TEST(body_state_defaults) {
    BodyState bs{};
    assert(bs.n_joints == 0);
    assert(bs.n_contacts == 0);
    assert(bs.step == 0);
    assert(bs.sim_time == 0.0f);
    assert(bs.heading == 0.0f);
    for (int i = 0; i < kMaxJoints; ++i) {
        assert(bs.joint_angles[i] == 0.0f);
        assert(bs.joint_velocities[i] == 0.0f);
    }
    for (int i = 0; i < kMaxContacts; ++i)
        assert(bs.contacts[i] == 0.0f);
}

// ---- tcp.h protocol tests ----

TEST(protocol_constants) {
    assert(kProtocolVersion == 1);
    assert(kDefaultPort == 9100);
}

TEST(msg_header_size) {
    static_assert(sizeof(MsgHeader) == 8);
    MsgHeader hdr{};
    hdr.type = kMsgHelloClient;
    hdr.payload_size = 42;
    assert(hdr.type == 0x00);
    assert(hdr.payload_size == 42);
}

TEST(bio_reading_size) {
    static_assert(sizeof(BioReading) == 16);
}

TEST(wire_motor_command_size) {
    static_assert(sizeof(WireMotorCommand) == 16);
}

TEST(bio_reading_from_sensory) {
    SensoryReading sr{};
    sr.channel = 42;
    sr.activation = 0.75f;
    sr.raw_value = 1.5f;

    BioReading bio = BioReading::FromSensory(sr);
    assert(bio.neuron_idx == 42);
    assert(std::fabs(bio.spike_prob - 0.75f) < 1e-6f);
    assert(std::fabs(bio.calcium_raw - 0.6f) < 1e-6f);  // 0.75 * 0.8
    assert(bio.voltage_mv == 0.0f);
}

TEST(bio_reading_from_sensory_zero) {
    SensoryReading sr{};
    BioReading bio = BioReading::FromSensory(sr);
    assert(bio.neuron_idx == 0);
    assert(bio.spike_prob == 0.0f);
    assert(bio.calcium_raw == 0.0f);
}

TEST(bio_reading_from_sensory_full) {
    SensoryReading sr{};
    sr.channel = 1000;
    sr.activation = 1.0f;
    sr.raw_value = 99.0f;

    BioReading bio = BioReading::FromSensory(sr);
    assert(bio.neuron_idx == 1000);
    assert(std::fabs(bio.spike_prob - 1.0f) < 1e-6f);
    assert(std::fabs(bio.calcium_raw - 0.8f) < 1e-6f);
}

TEST(parse_motor_json_basic) {
    const char* json = R"({"motor":{"forward_velocity":100.5,"angular_velocity":0.3,"approach_drive":0.0,"freeze":0.0}})";
    MotorCommand cmd{};
    bool ok = ParseMotorJSON(json, static_cast<int>(strlen(json)), cmd);
    assert(ok);
    assert(std::fabs(cmd.forward_velocity - 100.5f) < 0.1f);
    assert(std::fabs(cmd.angular_velocity - 0.3f) < 0.01f);
    assert(cmd.approach_drive == 0.0f);
    assert(cmd.freeze == 0.0f);
}

TEST(parse_motor_json_all_fields) {
    const char* json = R"({"motor":{"forward_velocity":-50.0,"angular_velocity":1.5,"approach_drive":0.7,"freeze":0.9}})";
    MotorCommand cmd{};
    bool ok = ParseMotorJSON(json, static_cast<int>(strlen(json)), cmd);
    assert(ok);
    assert(std::fabs(cmd.forward_velocity - (-50.0f)) < 0.1f);
    assert(std::fabs(cmd.angular_velocity - 1.5f) < 0.01f);
    assert(std::fabs(cmd.approach_drive - 0.7f) < 0.01f);
    assert(std::fabs(cmd.freeze - 0.9f) < 0.01f);
}

TEST(parse_motor_json_no_motor_key) {
    const char* json = R"({"status":"ok"})";
    MotorCommand cmd{};
    bool ok = ParseMotorJSON(json, static_cast<int>(strlen(json)), cmd);
    assert(!ok);
}

TEST(parse_motor_json_empty) {
    MotorCommand cmd{};
    bool ok = ParseMotorJSON("", 0, cmd);
    assert(!ok);
}

TEST(parse_motor_json_partial) {
    // Only forward_velocity present.
    const char* json = R"({"motor":{"forward_velocity":42.0}})";
    MotorCommand cmd{};
    bool ok = ParseMotorJSON(json, static_cast<int>(strlen(json)), cmd);
    assert(ok);
    assert(std::fabs(cmd.forward_velocity - 42.0f) < 0.1f);
    // Missing fields default to 0.
    assert(cmd.angular_velocity == 0.0f);
}

TEST(msg_type_values) {
    // Body -> Brain range.
    assert(kMsgHelloClient == 0x00);
    assert(kMsgBioReadings == 0x01);
    assert(kMsgConfig == 0x02);
    assert(kMsgPing == 0x03);
    assert(kMsgBodyState == 0x04);

    // Brain -> Body range.
    assert(kMsgHelloServer == 0x80);
    assert(kMsgStimCommands == 0x81);
    assert(kMsgStatus == 0x82);
    assert(kMsgPong == 0x83);
    assert(kMsgMotor == 0x84);
    assert(kMsgMotorBatch == 0x85);
}

TEST(wire_motor_command_layout) {
    WireMotorCommand w{};
    w.forward_velocity = 100.0f;
    w.angular_velocity = 0.5f;
    w.approach_drive = 0.3f;
    w.freeze = 0.1f;

    // Verify we can round-trip through raw bytes.
    uint8_t buf[sizeof(WireMotorCommand)];
    std::memcpy(buf, &w, sizeof(w));

    WireMotorCommand w2;
    std::memcpy(&w2, buf, sizeof(w2));
    assert(w2.forward_velocity == 100.0f);
    assert(w2.angular_velocity == 0.5f);
    assert(w2.approach_drive == 0.3f);
    assert(w2.freeze == 0.1f);
}

TEST(body_state_joint_storage) {
    BodyState bs{};
    bs.n_joints = 20;
    for (int i = 0; i < 20; ++i) {
        bs.joint_angles[i] = static_cast<float>(i) * 0.1f;
        bs.joint_velocities[i] = static_cast<float>(i) * -0.05f;
    }
    assert(std::fabs(bs.joint_angles[10] - 1.0f) < 1e-6f);
    assert(std::fabs(bs.joint_velocities[10] - (-0.5f)) < 1e-6f);
}

TEST(body_state_contacts) {
    BodyState bs{};
    bs.n_contacts = 4;
    bs.contacts[0] = 1.0f;  // LF on ground
    bs.contacts[1] = 0.0f;  // RF in air
    bs.contacts[2] = 0.5f;  // LH partial
    bs.contacts[3] = 1.0f;  // RH on ground
    assert(bs.contacts[0] == 1.0f);
    assert(bs.contacts[1] == 0.0f);
    assert(std::fabs(bs.contacts[2] - 0.5f) < 1e-6f);
}

TEST(body_state_velocity) {
    BodyState bs{};
    bs.body_velocity[0] = 80.0f;   // forward mm/s
    bs.body_velocity[1] = -5.0f;   // lateral mm/s
    bs.body_velocity[2] = 0.1f;    // yaw rad/s
    assert(bs.body_velocity[0] == 80.0f);
    assert(bs.body_velocity[1] == -5.0f);
    assert(std::fabs(bs.body_velocity[2] - 0.1f) < 1e-6f);
}

int main() {
    printf("mujoco-game protocol tests\n");
    return RunAllTests();
}
