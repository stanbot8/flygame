#ifndef FWMC_FILE_READ_CHANNEL_H_
#define FWMC_FILE_READ_CHANNEL_H_

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "bridge/bridge_channel.h"
#include "core/log.h"

namespace mechabrain {

// Reads pre-recorded neural data from a binary file.
// Used for offline validation: replay a real recording through the bridge
// and compare digital twin predictions against ground truth.
//
// File format (recording.bin):
//   [n_neurons:u32] [n_frames:u32] [sample_rate_hz:f32]
//   then per frame:
//     [time_ms:f32] [neuron_idx:u32, spike_prob:f32, calcium_raw:f32, voltage_mv:f32] * n_neurons
//
// If voltage is unavailable, write NaN for voltage_mv.
class FileReadChannel : public ReadChannel {
 public:
  bool Open(const std::string& path) {
    file_ = fopen(path.c_str(), "rb");
    if (!file_) {
      Log(LogLevel::kError, "FileReadChannel: cannot open %s", path.c_str());
      return false;
    }

    uint32_t n_neurons, n_frames;
    if (fread(&n_neurons, sizeof(uint32_t), 1, file_) != 1 ||
        fread(&n_frames, sizeof(uint32_t), 1, file_) != 1 ||
        fread(&sample_rate_hz_, sizeof(float), 1, file_) != 1) {
      Log(LogLevel::kError, "FileReadChannel: truncated header in %s", path.c_str());
      Close();
      return false;
    }

    if (n_neurons == 0 || n_neurons > 10000000) {
      Log(LogLevel::kError, "FileReadChannel: invalid neuron count %u in %s",
          n_neurons, path.c_str());
      Close();
      return false;
    }

    n_neurons_ = n_neurons;
    n_frames_ = n_frames;
    current_frame_ = 0;
    read_errors_ = 0;

    Log(LogLevel::kInfo, "FileReadChannel: %s (%u neurons, %u frames, %.0f Hz)",
        path.c_str(), n_neurons, n_frames, sample_rate_hz_);
    return true;
  }

  std::vector<BioReading> ReadFrame(float) override {
    std::vector<BioReading> readings;
    if (!file_ || current_frame_ >= n_frames_) return readings;

    float time_ms;
    if (fread(&time_ms, sizeof(float), 1, file_) != 1) {
      Log(LogLevel::kWarn, "FileReadChannel: EOF at frame %u/%u",
          current_frame_, n_frames_);
      n_frames_ = current_frame_;  // mark exhausted
      return readings;
    }

    readings.resize(n_neurons_);
    for (uint32_t i = 0; i < n_neurons_; ++i) {
      size_t read = 0;
      read += fread(&readings[i].neuron_idx, sizeof(uint32_t), 1, file_);
      read += fread(&readings[i].spike_prob, sizeof(float), 1, file_);
      read += fread(&readings[i].calcium_raw, sizeof(float), 1, file_);
      read += fread(&readings[i].voltage_mv, sizeof(float), 1, file_);
      if (read != 4) {
        Log(LogLevel::kWarn, "FileReadChannel: truncated frame %u at neuron %u",
            current_frame_, i);
        readings.resize(i);
        read_errors_++;
        break;
      }
    }

    current_frame_++;
    return readings;
  }

  size_t NumMonitored() const override { return n_neurons_; }
  float SampleRateHz() const override { return sample_rate_hz_; }

  bool IsExhausted() const { return current_frame_ >= n_frames_; }
  uint32_t FramesRemaining() const { return n_frames_ - current_frame_; }
  int ReadErrors() const { return read_errors_; }

  void Close() {
    if (file_) { fclose(file_); file_ = nullptr; }
  }

  ~FileReadChannel() { Close(); }

 private:
  FILE* file_ = nullptr;
  uint32_t n_neurons_ = 0;
  uint32_t n_frames_ = 0;
  uint32_t current_frame_ = 0;
  float sample_rate_hz_ = 1000.0f;
  int read_errors_ = 0;
};

}  // namespace mechabrain

#endif  // FWMC_FILE_READ_CHANNEL_H_
