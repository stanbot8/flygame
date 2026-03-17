#pragma once
// FWMC Bridge Protocol v1: TCP client for body simulators.
//
// Wire format: [uint32_le type] [uint32_le payload_size] [payload...]
//
// Connection flow:
//   1. Connect to brain sim on TCP (default port 9100)
//   2. Send HELLO with protocol version
//   3. Receive HELLO reply with server version
//   4. Main loop: send body state / readings, receive motor commands

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  using SocketType = SOCKET;
  constexpr SocketType kInvalidSocket = INVALID_SOCKET;
#else
  #include <arpa/inet.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using SocketType = int;
  constexpr SocketType kInvalidSocket = -1;
#endif

#include "types.h"

namespace mjgame {

// ---------------------------------------------------------------------------
// FWMC Bridge Protocol v1 constants and wire types.
// These must match mechabrain/io/protocol.h on the brain side.
// ---------------------------------------------------------------------------

constexpr uint32_t kProtocolVersion = 1;
constexpr uint16_t kDefaultPort     = 9100;

// Message types.
constexpr uint32_t kMsgHelloClient  = 0x00;  // Body -> Brain
constexpr uint32_t kMsgBioReadings  = 0x01;
constexpr uint32_t kMsgConfig       = 0x02;
constexpr uint32_t kMsgPing         = 0x03;
constexpr uint32_t kMsgBodyState    = 0x04;

constexpr uint32_t kMsgHelloServer  = 0x80;  // Brain -> Body
constexpr uint32_t kMsgStimCommands = 0x81;
constexpr uint32_t kMsgStatus       = 0x82;  // JSON motor (legacy v0)
constexpr uint32_t kMsgPong         = 0x83;
constexpr uint32_t kMsgMotor        = 0x84;  // Binary motor (v1+)
constexpr uint32_t kMsgMotorBatch   = 0x85;  // Binary motor batch (v1+)
constexpr uint32_t kMsgBrainFrame   = 0x86;  // Visualization frame (RGBA)

// Wire header (8 bytes).
#pragma pack(push, 1)
struct MsgHeader {
    uint32_t type;
    uint32_t payload_size;
};

// Neural reading (16 bytes).
struct BioReading {
    uint32_t neuron_idx = 0;
    float    spike_prob = 0.0f;
    float    calcium_raw = 0.0f;
    float    voltage_mv = 0.0f;

    static BioReading FromSensory(const SensoryReading& sr) {
        BioReading b;
        b.neuron_idx  = sr.channel;
        b.spike_prob  = sr.activation;
        b.calcium_raw = sr.activation * 0.8f;
        return b;
    }
};
static_assert(sizeof(BioReading) == 16);

// Binary motor command (16 bytes).
struct WireMotorCommand {
    float forward_velocity;
    float angular_velocity;
    float approach_drive;
    float freeze;
};
static_assert(sizeof(WireMotorCommand) == 16);
#pragma pack(pop)

// Simple JSON parser for legacy MotorCommand.
// Expects: {"motor":{"forward_velocity":X,"angular_velocity":Y,...}}
inline bool ParseMotorJSON(const char* json, int len, MotorCommand& out) {
    std::string s(json, len);
    auto getf = [&](const char* key) -> float {
        auto pos = s.find(key);
        if (pos == std::string::npos) return 0.0f;
        pos = s.find(':', pos);
        if (pos == std::string::npos) return 0.0f;
        return std::strtof(s.c_str() + pos + 1, nullptr);
    };

    if (s.find("motor") == std::string::npos) return false;

    out.forward_velocity = getf("forward_velocity");
    out.angular_velocity = getf("angular_velocity");
    out.approach_drive   = getf("approach_drive");
    out.freeze           = getf("freeze");
    return true;
}

// ---------------------------------------------------------------------------
// TCP Client
// ---------------------------------------------------------------------------

class TcpClient {
public:
    TcpClient() {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    }

    ~TcpClient() { Close(); }

    /// Connect and perform the v1 HELLO handshake.
    /// Returns true if connected (even if server is v0).
    bool Connect(const std::string& host, int port, float timeout_sec = 5.0f) {
        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ == kInvalidSocket) return false;

        int flag = 1;
        setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&flag), sizeof(flag));

        int ms = static_cast<int>(timeout_sec * 1000);
#ifdef _WIN32
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&ms), sizeof(ms));
        setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO,
                   reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
        struct timeval tv = {};
        tv.tv_sec = static_cast<int>(timeout_sec);
        tv.tv_usec = static_cast<int>((timeout_sec - tv.tv_sec) * 1e6);
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

        if (connect(sock_, reinterpret_cast<sockaddr*>(&addr),
                    sizeof(addr)) != 0) {
            Close();
            return false;
        }

        connected_ = true;

        // Send HELLO with our protocol version.
        uint32_t ver = kProtocolVersion;
        if (!SendMsg(kMsgHelloClient,
                     reinterpret_cast<const uint8_t*>(&ver), 4)) {
            Close();
            return false;
        }

        // Wait for server HELLO.
        std::vector<uint8_t> payload;
        uint32_t reply_type = RecvMsg(payload);
        if (reply_type == kMsgHelloServer && payload.size() >= 4) {
            std::memcpy(&server_version_, payload.data(), 4);
        } else {
            server_version_ = 0;  // Legacy server, no handshake support.
        }

        return true;
    }

    void Close() {
        if (sock_ != kInvalidSocket) {
#ifdef _WIN32
            closesocket(sock_);
#else
            close(sock_);
#endif
            sock_ = kInvalidSocket;
        }
        connected_ = false;
        server_version_ = 0;
    }

    bool IsConnected() const { return connected_; }
    uint32_t ServerVersion() const { return server_version_; }
    bool IsV1() const { return server_version_ >= 1; }

    // -- sending -----------------------------------------------------------

    bool SendReadings(const SensoryReading* readings, int count) {
        std::vector<BioReading> bio(count);
        for (int i = 0; i < count; ++i)
            bio[i] = BioReading::FromSensory(readings[i]);
        return SendMsg(kMsgBioReadings,
                       reinterpret_cast<const uint8_t*>(bio.data()),
                       count * sizeof(BioReading));
    }

    /// Send raw body state bytes over the protocol.
    bool SendBodyStateRaw(const void* data, uint32_t size) {
        return SendMsg(kMsgBodyState,
                       reinterpret_cast<const uint8_t*>(data), size);
    }

    bool SendPing() {
        return SendMsg(kMsgPing, nullptr, 0);
    }

    // -- receiving ---------------------------------------------------------

    /// Receive a message. Returns message type (0 on error).
    uint32_t RecvMsg(std::vector<uint8_t>& payload) {
        uint32_t header[2];
        if (!RecvExact(reinterpret_cast<uint8_t*>(header), 8))
            return 0;

        uint32_t msg_type = header[0];
        uint32_t size     = header[1];

        if (size > 1 << 20) return 0;  // Reject oversized payloads.

        if (size > 0) {
            payload.resize(size);
            if (!RecvExact(payload.data(), size))
                return 0;
        } else {
            payload.clear();
        }
        return msg_type;
    }

    // -- high-level exchange -----------------------------------------------

    /// Send readings, receive motor command. Handles v0 JSON and v1 binary.
    bool Exchange(const SensoryReading* readings, int count,
                  MotorCommand& motor_out) {
        if (!SendReadings(readings, count)) {
            connected_ = false;
            return false;
        }

        return RecvMotor(motor_out);
    }

    /// Receive a motor command (call after sending any request).
    bool RecvMotor(MotorCommand& motor_out) {
        std::vector<uint8_t> payload;
        uint32_t msg_type = RecvMsg(payload);
        if (msg_type == 0) {
            connected_ = false;
            return false;
        }

        // v1 motor batch: N commands, use last (most recent state).
        if (msg_type == kMsgMotorBatch &&
            payload.size() >= sizeof(WireMotorCommand)) {
            size_t n = payload.size() / sizeof(WireMotorCommand);
            size_t offset = (n - 1) * sizeof(WireMotorCommand);
            WireMotorCommand w;
            std::memcpy(&w, payload.data() + offset, sizeof(w));
            motor_out.forward_velocity = w.forward_velocity;
            motor_out.angular_velocity = w.angular_velocity;
            motor_out.approach_drive   = w.approach_drive;
            motor_out.freeze           = w.freeze;
            return true;
        }

        // v1 binary motor command (single).
        if (msg_type == kMsgMotor &&
            payload.size() >= sizeof(WireMotorCommand)) {
            WireMotorCommand w;
            std::memcpy(&w, payload.data(), sizeof(w));
            motor_out.forward_velocity = w.forward_velocity;
            motor_out.angular_velocity = w.angular_velocity;
            motor_out.approach_drive   = w.approach_drive;
            motor_out.freeze           = w.freeze;
            return true;
        }

        // Legacy JSON motor command.
        if (msg_type == kMsgStatus && !payload.empty()) {
            return ParseMotorJSON(
                reinterpret_cast<const char*>(payload.data()),
                static_cast<int>(payload.size()), motor_out);
        }

        return false;
    }

private:
    SocketType sock_ = kInvalidSocket;
    bool connected_ = false;
    uint32_t server_version_ = 0;

    bool SendMsg(uint32_t type, const uint8_t* data, uint32_t size) {
        MsgHeader hdr;
        hdr.type = type;
        hdr.payload_size = size;
        if (send(sock_, reinterpret_cast<const char*>(&hdr),
                 sizeof(hdr), 0) != sizeof(hdr))
            return false;
        if (size > 0 && data) {
            int sent = send(sock_, reinterpret_cast<const char*>(data),
                            static_cast<int>(size), 0);
            return sent == static_cast<int>(size);
        }
        return true;
    }

    bool RecvExact(uint8_t* buf, int n) {
        int received = 0;
        while (received < n) {
            int r = recv(sock_, reinterpret_cast<char*>(buf + received),
                         n - received, 0);
            if (r <= 0) return false;
            received += r;
        }
        return true;
    }
};

}  // namespace mjgame
