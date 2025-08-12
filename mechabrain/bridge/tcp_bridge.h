#ifndef FWMC_TCP_BRIDGE_H_
#define FWMC_TCP_BRIDGE_H_

// TCP-based bridge for hardware integration and body sim communication.
//
// See bridge/protocol.h for the full protocol specification.
// This header provides ReadChannel/WriteChannel adapters that wrap
// a TCP socket into the bridge's abstract I/O interfaces.

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "bridge/bridge_channel.h"
#include "bridge/protocol.h"

namespace mechabrain {

// Legacy aliases (kept for existing code; prefer protocol:: types directly).
using TcpMsgType   = protocol::MsgType;
using TcpMsgHeader = protocol::Header;

// Server configuration
struct TcpBridgeConfig {
  uint16_t port = protocol::kDefaultPort;
  float timeout_ms = 100.0f;      // read timeout per frame (ms)
  size_t max_payload = 1 << 20;   // max payload size (1 MB)
  bool loopback_only = true;      // only accept connections from localhost
};

// Serialize BioReadings to binary payload (header + data).
inline std::vector<uint8_t> SerializeBioReadings(
    const std::vector<BioReading>& readings) {
  size_t size = readings.size() * sizeof(BioReading);
  std::vector<uint8_t> buf(sizeof(protocol::Header) + size);
  protocol::Header hdr;
  hdr.type = static_cast<uint32_t>(protocol::MsgType::kBioReadings);
  hdr.payload_size = static_cast<uint32_t>(size);
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  std::memcpy(buf.data() + sizeof(hdr), readings.data(), size);
  return buf;
}

// Deserialize BioReadings from raw payload bytes.
inline std::vector<BioReading> DeserializeBioReadings(
    const uint8_t* payload, uint32_t size) {
  size_t count = size / sizeof(BioReading);
  std::vector<BioReading> readings(count);
  std::memcpy(readings.data(), payload, count * sizeof(BioReading));
  return readings;
}

// Serialize StimCommands to binary payload (header + data).
inline std::vector<uint8_t> SerializeStimCommands(
    const std::vector<StimCommand>& commands) {
  size_t size = commands.size() * sizeof(StimCommand);
  std::vector<uint8_t> buf(sizeof(protocol::Header) + size);
  protocol::Header hdr;
  hdr.type = static_cast<uint32_t>(protocol::MsgType::kStimCommands);
  hdr.payload_size = static_cast<uint32_t>(size);
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  std::memcpy(buf.data() + sizeof(hdr), commands.data(), size);
  return buf;
}

// Deserialize StimCommands from raw payload bytes.
inline std::vector<StimCommand> DeserializeStimCommands(
    const uint8_t* payload, uint32_t size) {
  size_t count = size / sizeof(StimCommand);
  std::vector<StimCommand> commands(count);
  std::memcpy(commands.data(), payload, count * sizeof(StimCommand));
  return commands;
}

// TCP read channel: receives BioReadings from a connected client.
// Uses a callback for the actual socket recv (platform-dependent).
class TcpReadChannel : public ReadChannel {
 public:
  using RecvFn = std::function<int(uint8_t* buf, size_t len)>;

  TcpReadChannel(RecvFn recv_fn, size_t n_monitored, float sample_rate_hz)
      : recv_fn_(std::move(recv_fn)),
        n_monitored_(n_monitored),
        sample_rate_hz_(sample_rate_hz) {}

  std::vector<BioReading> ReadFrame(float) override {
    // Read header
    protocol::Header hdr{};
    int n = recv_fn_(reinterpret_cast<uint8_t*>(&hdr), sizeof(hdr));
    if (n != sizeof(hdr)) return {};

    if (hdr.type != static_cast<uint32_t>(protocol::MsgType::kBioReadings)) return {};
    if (hdr.payload_size > 1 << 20) return {};  // safety limit

    // Read payload
    std::vector<uint8_t> payload(hdr.payload_size);
    n = recv_fn_(payload.data(), hdr.payload_size);
    if (n != static_cast<int>(hdr.payload_size)) return {};

    return DeserializeBioReadings(payload.data(), hdr.payload_size);
  }

  size_t NumMonitored() const override { return n_monitored_; }
  float SampleRateHz() const override { return sample_rate_hz_; }

 private:
  RecvFn recv_fn_;
  size_t n_monitored_;
  float sample_rate_hz_;
};

// TCP write channel: sends StimCommands to a connected client.
class TcpWriteChannel : public WriteChannel {
 public:
  using SendFn = std::function<int(const uint8_t* buf, size_t len)>;

  TcpWriteChannel(SendFn send_fn, size_t max_targets, float min_isi)
      : send_fn_(std::move(send_fn)),
        max_targets_(max_targets),
        min_isi_(min_isi) {}

  void WriteFrame(const std::vector<StimCommand>& commands) override {
    auto buf = SerializeStimCommands(commands);
    send_fn_(buf.data(), buf.size());
  }

  size_t MaxTargets() const override { return max_targets_; }
  float MinISI() const override { return min_isi_; }

 private:
  SendFn send_fn_;
  size_t max_targets_;
  float min_isi_;
};

}  // namespace mechabrain

#endif  // FWMC_TCP_BRIDGE_H_
