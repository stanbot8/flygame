#ifndef FWMC_BRIDGE_CHANNEL_H_
#define FWMC_BRIDGE_CHANNEL_H_

#include <cstdint>
#include <vector>
#include "bridge/protocol.h"

namespace mechabrain {

// Wire-compatible types re-exported from protocol.h for convenience.
// These are the canonical definitions; do not redefine them.
using BioReading = protocol::BioReading;
using StimCommand = protocol::StimCommand;

// Abstract read channel: biological tissue to digital twin.
// Implementations: CalciumDecoder (GCaMP), VoltageDecoder (ASAP4),
// ElectrophysDecoder (Neuropixels), or SimulatedRead (for testing).
class ReadChannel {
 public:
  virtual ~ReadChannel() = default;
  virtual std::vector<BioReading> ReadFrame(float sim_time_ms) = 0;
  virtual size_t NumMonitored() const = 0;
  virtual float SampleRateHz() const = 0;
};

// Abstract write channel: digital twin to biological tissue.
// Implementations: HolographicOptogenetics, SingleFiberOpto,
// or SimulatedWrite (for testing).
class WriteChannel {
 public:
  virtual ~WriteChannel() = default;
  virtual void WriteFrame(const std::vector<StimCommand>& commands) = 0;
  virtual size_t MaxTargets() const = 0;
  virtual float MinISI() const = 0;
};

// Simulated read channel: reads from a NeuronArray directly.
// Used for development and validation (no biological hardware needed).
class SimulatedRead : public ReadChannel {
 public:
  explicit SimulatedRead(float sample_rate_hz = 1000.0f)
      : sample_rate_hz_(sample_rate_hz) {}

  void SetSpikeData(const std::vector<BioReading>& data) {
    latest_ = data;
  }

  std::vector<BioReading> ReadFrame(float) override { return latest_; }
  size_t NumMonitored() const override { return latest_.size(); }
  float SampleRateHz() const override { return sample_rate_hz_; }

 private:
  std::vector<BioReading> latest_;
  float sample_rate_hz_;
};

// Simulated write channel: records commands for inspection.
class SimulatedWrite : public WriteChannel {
 public:
  void WriteFrame(const std::vector<StimCommand>& commands) override {
    last_commands_ = commands;
  }
  size_t MaxTargets() const override { return 1000; }
  float MinISI() const override { return 0.1f; }

  const std::vector<StimCommand>& LastCommands() const {
    return last_commands_;
  }

 private:
  std::vector<StimCommand> last_commands_;
};

}  // namespace mechabrain

#endif  // FWMC_BRIDGE_CHANNEL_H_
