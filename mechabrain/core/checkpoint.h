#ifndef FWMC_CHECKPOINT_H_
#define FWMC_CHECKPOINT_H_

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Binary checkpoint format for saving and restoring simulation state.
//
// The core checkpoint handles neuron/synapse state only.
// Bridge-specific state (replacer, shadow tracker) is serialized via
// an opaque extension blob, so core/ never depends on bridge/.
//
// File layout:
//   [magic:u32]  "FWCK" = 0x4B435746
//   [version:u32]
//   [sim_time_ms:f32]
//   [total_steps:i32]
//   [total_resyncs:i32]
//   [n_neurons:u32]
//   [n_synapses:u32]
//   --- Neuron state ---
//   [v, u, i_syn, spiked, dopamine, serotonin, octopamine, last_spike_time]
//   --- Synapse weights ---
//   [weights[0..s]:f32]
//   --- Extension blob (bridge state, opaque to core) ---
//   [ext_size:u32]
//   [ext_data[0..ext_size]:u8]
struct Checkpoint {
  static constexpr uint32_t kMagic = 0x4B435746;  // "FWCK"
  static constexpr uint32_t kVersion = 3;  // v3: adds STP state, release probs, neuron metadata

  // Save core simulation state + optional extension blob.
  static bool Save(const std::string& path,
                   float sim_time_ms, int total_steps, int total_resyncs,
                   const NeuronArray& neurons,
                   const SynapseTable& synapses,
                   const std::vector<uint8_t>& extension = {}) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
      Log(LogLevel::kError, "Checkpoint: cannot open %s for writing", path.c_str());
      return false;
    }

    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    uint32_t n = static_cast<uint32_t>(neurons.n);
    uint32_t s = static_cast<uint32_t>(synapses.Size());

    bool ok = true;
    ok &= Wr(f, &magic, 1);
    ok &= Wr(f, &version, 1);
    ok &= Wr(f, &sim_time_ms, 1);
    ok &= Wr(f, &total_steps, 1);
    ok &= Wr(f, &total_resyncs, 1);
    ok &= Wr(f, &n, 1);
    ok &= Wr(f, &s, 1);

    // Neuron state
    ok &= Wr(f, neurons.v.data(), n);
    ok &= Wr(f, neurons.u.data(), n);
    ok &= Wr(f, neurons.i_syn.data(), n);
    ok &= Wr(f, neurons.spiked.data(), n);
    ok &= Wr(f, neurons.dopamine.data(), n);
    ok &= Wr(f, neurons.serotonin.data(), n);
    ok &= Wr(f, neurons.octopamine.data(), n);
    ok &= Wr(f, neurons.last_spike_time.data(), n);

    // Neuron metadata
    ok &= Wr(f, neurons.type.data(), n);
    ok &= Wr(f, neurons.region.data(), n);
    ok &= Wr(f, neurons.i_ext.data(), n);

    // Synapse weights
    ok &= Wr(f, synapses.weight.data(), s);

    // STP state (write flag + data if present)
    uint8_t has_stp = synapses.HasSTP() ? 1 : 0;
    ok &= Wr(f, &has_stp, 1);
    if (has_stp) {
      ok &= Wr(f, synapses.stp_u.data(), s);
      ok &= Wr(f, synapses.stp_x.data(), s);
      ok &= Wr(f, synapses.stp_U_se.data(), s);
      ok &= Wr(f, synapses.stp_tau_d.data(), s);
      ok &= Wr(f, synapses.stp_tau_f.data(), s);
    }

    // Stochastic release probabilities
    uint8_t has_release = synapses.HasStochasticRelease() ? 1 : 0;
    ok &= Wr(f, &has_release, 1);
    if (has_release) {
      ok &= Wr(f, synapses.p_release.data(), s);
    }

    // Extension blob (bridge state, opaque to core)
    uint32_t ext_size = static_cast<uint32_t>(extension.size());
    ok &= Wr(f, &ext_size, 1);
    if (ext_size > 0) {
      ok &= Wr(f, extension.data(), ext_size);
    }

    fclose(f);

    if (!ok) {
      Log(LogLevel::kError, "Checkpoint: write error to %s", path.c_str());
      return false;
    }

    Log(LogLevel::kInfo, "Checkpoint saved: %s (t=%.1fms, step=%d, %u neurons, %u synapses)",
        path.c_str(), sim_time_ms, total_steps, n, s);
    return true;
  }

  // Load core simulation state + optional extension blob.
  // Neurons and synapses must already be sized correctly.
  static bool Load(const std::string& path,
                   float& sim_time_ms, int& total_steps, int& total_resyncs,
                   NeuronArray& neurons,
                   SynapseTable& synapses,
                   std::vector<uint8_t>& extension) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      Log(LogLevel::kError, "Checkpoint: cannot open %s for reading", path.c_str());
      return false;
    }

    uint32_t magic = 0, version = 0;
    uint32_t n = 0, s = 0;

    bool ok = true;
    ok &= Rd(f, &magic, 1);
    ok &= Rd(f, &version, 1);

    if (!ok || magic != kMagic) {
      Log(LogLevel::kError, "Checkpoint: invalid magic in %s", path.c_str());
      fclose(f);
      return false;
    }
    if (version < 3 || version > kVersion) {
      Log(LogLevel::kError, "Checkpoint: version %u, expected %u", version, kVersion);
      fclose(f);
      return false;
    }

    ok &= Rd(f, &sim_time_ms, 1);
    ok &= Rd(f, &total_steps, 1);
    ok &= Rd(f, &total_resyncs, 1);
    ok &= Rd(f, &n, 1);
    ok &= Rd(f, &s, 1);

    if (n != static_cast<uint32_t>(neurons.n)) {
      Log(LogLevel::kError, "Checkpoint: neuron count mismatch (%u vs %zu)", n, neurons.n);
      fclose(f);
      return false;
    }
    if (s != static_cast<uint32_t>(synapses.Size())) {
      Log(LogLevel::kError, "Checkpoint: synapse count mismatch (%u vs %zu)", s, synapses.Size());
      fclose(f);
      return false;
    }

    // Neuron state
    ok &= Rd(f, neurons.v.data(), n);
    ok &= Rd(f, neurons.u.data(), n);
    ok &= Rd(f, neurons.i_syn.data(), n);
    ok &= Rd(f, neurons.spiked.data(), n);
    ok &= Rd(f, neurons.dopamine.data(), n);
    ok &= Rd(f, neurons.serotonin.data(), n);
    ok &= Rd(f, neurons.octopamine.data(), n);
    ok &= Rd(f, neurons.last_spike_time.data(), n);

    // Neuron metadata
    ok &= Rd(f, neurons.type.data(), n);
    ok &= Rd(f, neurons.region.data(), n);
    ok &= Rd(f, neurons.i_ext.data(), n);

    // Synapse weights
    ok &= Rd(f, synapses.weight.data(), s);

    // STP state
    uint8_t has_stp = 0;
    ok &= Rd(f, &has_stp, 1);
    if (has_stp && s > 0) {
      synapses.stp_u.resize(s);
      synapses.stp_x.resize(s);
      synapses.stp_U_se.resize(s);
      synapses.stp_tau_d.resize(s);
      synapses.stp_tau_f.resize(s);
      ok &= Rd(f, synapses.stp_u.data(), s);
      ok &= Rd(f, synapses.stp_x.data(), s);
      ok &= Rd(f, synapses.stp_U_se.data(), s);
      ok &= Rd(f, synapses.stp_tau_d.data(), s);
      ok &= Rd(f, synapses.stp_tau_f.data(), s);
    }

    // Stochastic release probabilities
    uint8_t has_release = 0;
    ok &= Rd(f, &has_release, 1);
    if (has_release && s > 0) {
      synapses.p_release.resize(s);
      ok &= Rd(f, synapses.p_release.data(), s);
    }

    // Extension blob
    uint32_t ext_size = 0;
    ok &= Rd(f, &ext_size, 1);
    if (ext_size > 100 * 1024 * 1024) {  // sanity check: 100MB max
      Log(LogLevel::kError, "Checkpoint: extension blob too large (%u bytes)", ext_size);
      fclose(f);
      return false;
    }
    extension.resize(ext_size);
    if (ext_size > 0) {
      ok &= Rd(f, extension.data(), ext_size);
    }

    fclose(f);

    if (!ok) {
      Log(LogLevel::kError, "Checkpoint: read error from %s", path.c_str());
      return false;
    }

    Log(LogLevel::kInfo, "Checkpoint loaded: %s (t=%.1fms, step=%d)",
        path.c_str(), sim_time_ms, total_steps);
    return true;
  }

 private:
  template <typename T>
  static bool Wr(FILE* f, const T* data, size_t count) {
    return fwrite(data, sizeof(T), count, f) == count;
  }

  template <typename T>
  static bool Rd(FILE* f, T* data, size_t count) {
    return fread(data, sizeof(T), count, f) == count;
  }
};

}  // namespace mechabrain

#endif  // FWMC_CHECKPOINT_H_
