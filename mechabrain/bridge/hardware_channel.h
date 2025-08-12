#ifndef FWMC_HARDWARE_CHANNEL_H_
#define FWMC_HARDWARE_CHANNEL_H_

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "bridge/bridge_channel.h"

namespace mechabrain {

// ============================================================
// Hardware channel adapters for real experimental rigs.
//
// Each adapter implements ReadChannel or WriteChannel using
// callbacks that the user wires to their acquisition/control
// library. This keeps FWMC dependency-free while providing
// a clear integration surface.
//
// Supported paradigms:
//   Open Ephys    : electrophysiology (Neuropixels, tetrodes)
//   ScanImage     : two-photon calcium imaging (GCaMP, jRGECO)
//   Bonsai        : reactive visual programming (any data source)
//   ZeroMQ        : generic pub/sub transport (language-agnostic)
// ============================================================

// --- Callback-based read channel ---
// User provides a function that fills a vector of BioReadings.
// This is the simplest integration path: wrap your SDK calls
// in a lambda and hand it to CallbackReadChannel.
class CallbackReadChannel : public ReadChannel {
 public:
  using ReadFn = std::function<std::vector<BioReading>(float sim_time_ms)>;

  CallbackReadChannel(ReadFn fn, size_t n_monitored, float sample_rate_hz)
      : fn_(std::move(fn)),
        n_monitored_(n_monitored),
        sample_rate_hz_(sample_rate_hz) {}

  std::vector<BioReading> ReadFrame(float sim_time_ms) override {
    return fn_(sim_time_ms);
  }
  size_t NumMonitored() const override { return n_monitored_; }
  float SampleRateHz() const override { return sample_rate_hz_; }

 private:
  ReadFn fn_;
  size_t n_monitored_;
  float sample_rate_hz_;
};

// --- Callback-based write channel ---
class CallbackWriteChannel : public WriteChannel {
 public:
  using WriteFn = std::function<void(const std::vector<StimCommand>&)>;

  CallbackWriteChannel(WriteFn fn, size_t max_targets, float min_isi)
      : fn_(std::move(fn)),
        max_targets_(max_targets),
        min_isi_(min_isi) {}

  void WriteFrame(const std::vector<StimCommand>& commands) override {
    fn_(commands);
  }
  size_t MaxTargets() const override { return max_targets_; }
  float MinISI() const override { return min_isi_; }

 private:
  WriteFn fn_;
  size_t max_targets_;
  float min_isi_;
};

// --- Shared memory read channel ---
// Reads BioReading data from a shared memory buffer.
// Layout: [uint32_t count] [BioReading * count]
// The external process (Open Ephys plugin, Bonsai node, etc.)
// writes to this buffer; FWMC reads from it each frame.
class SharedMemoryReadChannel : public ReadChannel {
 public:
  SharedMemoryReadChannel(void* shm_ptr, size_t shm_size,
                          size_t n_monitored, float sample_rate_hz)
      : shm_ptr_(static_cast<uint8_t*>(shm_ptr)),
        shm_size_(shm_size),
        n_monitored_(n_monitored),
        sample_rate_hz_(sample_rate_hz) {}

  std::vector<BioReading> ReadFrame(float) override {
    std::vector<BioReading> readings;
    if (!shm_ptr_ || shm_size_ < sizeof(uint32_t)) return readings;

    uint32_t count = 0;
    std::memcpy(&count, shm_ptr_, sizeof(uint32_t));

    size_t needed = sizeof(uint32_t) + count * sizeof(BioReading);
    if (needed > shm_size_) return readings;

    readings.resize(count);
    std::memcpy(readings.data(), shm_ptr_ + sizeof(uint32_t),
                count * sizeof(BioReading));
    return readings;
  }

  size_t NumMonitored() const override { return n_monitored_; }
  float SampleRateHz() const override { return sample_rate_hz_; }

 private:
  uint8_t* shm_ptr_;
  size_t shm_size_;
  size_t n_monitored_;
  float sample_rate_hz_;
};

// --- Shared memory write channel ---
// Writes StimCommands to a shared memory buffer.
// Layout: [uint32_t count] [StimCommand * count]
class SharedMemoryWriteChannel : public WriteChannel {
 public:
  SharedMemoryWriteChannel(void* shm_ptr, size_t shm_size,
                           size_t max_targets, float min_isi)
      : shm_ptr_(static_cast<uint8_t*>(shm_ptr)),
        shm_size_(shm_size),
        max_targets_(max_targets),
        min_isi_(min_isi) {}

  void WriteFrame(const std::vector<StimCommand>& commands) override {
    if (!shm_ptr_) return;
    uint32_t count = static_cast<uint32_t>(commands.size());
    size_t needed = sizeof(uint32_t) + count * sizeof(StimCommand);
    if (needed > shm_size_) {
      // Truncate to fit
      count = static_cast<uint32_t>((shm_size_ - sizeof(uint32_t)) / sizeof(StimCommand));
    }
    std::memcpy(shm_ptr_, &count, sizeof(uint32_t));
    std::memcpy(shm_ptr_ + sizeof(uint32_t), commands.data(),
                count * sizeof(StimCommand));
  }

  size_t MaxTargets() const override { return max_targets_; }
  float MinISI() const override { return min_isi_; }

 private:
  uint8_t* shm_ptr_;
  size_t shm_size_;
  size_t max_targets_;
  float min_isi_;
};

// --- Ring buffer for streaming data ---
// Lock-free single-producer single-consumer ring buffer for
// continuous data streaming between acquisition thread and
// simulation thread.
template <typename T>
struct RingBuffer {
  std::vector<T> buffer;
  size_t capacity = 0;
  size_t write_pos = 0;  // written by producer
  size_t read_pos = 0;   // written by consumer

  void Init(size_t cap) {
    capacity = cap;
    buffer.resize(cap);
    write_pos = 0;
    read_pos = 0;
  }

  bool Push(const T& item) {
    size_t next = (write_pos + 1) % capacity;
    if (next == read_pos) return false;  // full
    buffer[write_pos] = item;
    write_pos = next;
    return true;
  }

  bool Pop(T& item) {
    if (read_pos == write_pos) return false;  // empty
    item = buffer[read_pos];
    read_pos = (read_pos + 1) % capacity;
    return true;
  }

  size_t Available() const {
    if (write_pos >= read_pos) return write_pos - read_pos;
    return capacity - read_pos + write_pos;
  }
};

// --- Open Ephys integration ---
// Open Ephys uses a plugin architecture. The typical integration
// path is to write a custom processor plugin that:
//   1. Receives spike-sorted data from Neuropixels
//   2. Writes BioReadings to shared memory
//   3. Reads StimCommands from shared memory
//   4. Forwards commands to the stimulation hardware
//
// This struct holds the configuration for an Open Ephys session.
struct OpenEphysConfig {
  std::string host = "localhost";
  int data_port = 5556;       // ZMQ port for spike data
  int command_port = 5557;    // ZMQ port for stim commands
  float sample_rate_hz = 30000.0f;  // Neuropixels sample rate
  size_t n_channels = 384;          // Neuropixels 1.0 channel count
  float spike_threshold_uv = -50.0f; // threshold for spike detection

  // Spike sorting output: neuron index mapping
  // Maps sorted unit IDs to FWMC neuron indices
  std::vector<std::pair<uint32_t, uint32_t>> unit_to_neuron;
};

// --- ScanImage integration ---
// ScanImage (Vidrio Technologies) controls two-photon microscopes.
// Integration uses ScanImage's external trigger and
// MATLAB/Python API for hologram upload.
struct ScanImageConfig {
  std::string host = "localhost";
  int api_port = 5558;
  float frame_rate_hz = 30.0f;       // imaging frame rate
  float pixel_dwell_us = 1.0f;       // pixel dwell time
  int pixels_x = 512;
  int pixels_y = 512;
  float fov_um = 500.0f;             // field of view (um)
  float zoom = 1.0f;

  // ROI definitions: each ROI maps to a neuron index
  struct ROI {
    uint32_t neuron_idx;
    float center_x_um;
    float center_y_um;
    float radius_um;
  };
  std::vector<ROI> rois;
};

// --- Bonsai integration ---
// Bonsai is a reactive visual programming framework for
// neuroscience experiments. Integration via its OSC/ZMQ nodes.
struct BonsaiConfig {
  std::string osc_address = "/fwmc/bio";
  int osc_port = 9000;
  std::string stim_address = "/fwmc/stim";
  int stim_port = 9001;
  float sample_rate_hz = 1000.0f;
};

}  // namespace mechabrain

#endif  // FWMC_HARDWARE_CHANNEL_H_
