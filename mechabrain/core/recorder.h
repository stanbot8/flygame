#ifndef FWMC_RECORDER_H_
#define FWMC_RECORDER_H_

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include "core/log.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Drift metrics snapshot for recording. This is a plain struct with no
// dependencies on bridge types; bridge code populates it from ShadowTracker.
struct DriftMetrics {
  float spike_correlation = 0.0f;
  float population_rmse = 0.0f;
  float mean_v_error = 0.0f;
  int n_false_positive = 0;
  int n_false_negative = 0;
};

// Records simulation data to binary files for offline analysis.
struct Recorder {
  Recorder() = default;
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  Recorder(Recorder&&) = delete;
  Recorder& operator=(Recorder&&) = delete;

  FILE* spike_file = nullptr;
  FILE* voltage_file = nullptr;
  FILE* metrics_file = nullptr;
  FILE* per_neuron_file = nullptr;

  uint32_t n_neurons = 0;
  uint32_t n_recorded_steps = 0;
  std::string output_dir;

  bool record_spikes = true;
  bool record_voltages = false;
  bool record_shadow_metrics = true;
  bool record_per_neuron_error = true;
  int recording_interval = 1;

  bool Open(const std::string& dir, uint32_t num_neurons) {
    output_dir = dir;
    n_neurons = num_neurons;
    n_recorded_steps = 0;

    std::filesystem::create_directories(dir);

    auto fail = [&](const char* msg, const std::string& path) {
      Log(LogLevel::kError, "%s: %s", msg, path.c_str());
      Close();
      return false;
    };

    if (record_spikes) {
      std::string path = dir + "/spikes.bin";
      spike_file = fopen(path.c_str(), "wb");
      if (!spike_file) return fail("Cannot open", path);
      if (fwrite(&n_neurons, sizeof(uint32_t), 1, spike_file) != 1 ||
          fwrite(&n_recorded_steps, sizeof(uint32_t), 1, spike_file) != 1)
        return fail("Cannot write header to", path);
    }

    if (record_voltages) {
      std::string path = dir + "/voltages.bin";
      voltage_file = fopen(path.c_str(), "wb");
      if (!voltage_file) return fail("Cannot open", path);
      if (fwrite(&n_neurons, sizeof(uint32_t), 1, voltage_file) != 1 ||
          fwrite(&n_recorded_steps, sizeof(uint32_t), 1, voltage_file) != 1)
        return fail("Cannot write header to", path);
    }

    if (record_shadow_metrics) {
      std::string path = dir + "/metrics.csv";
      metrics_file = fopen(path.c_str(), "w");
      if (!metrics_file) return fail("Cannot open", path);
      fprintf(metrics_file,
              "time_ms,spike_count,correlation,rmse,mean_v_error,"
              "false_pos,false_neg,resyncs,replaced_pct\n");
    }

    if (record_per_neuron_error) {
      std::string path = dir + "/per_neuron_error.bin";
      per_neuron_file = fopen(path.c_str(), "wb");
      if (!per_neuron_file) return fail("Cannot open", path);
      if (fwrite(&n_neurons, sizeof(uint32_t), 1, per_neuron_file) != 1 ||
          fwrite(&n_recorded_steps, sizeof(uint32_t), 1, per_neuron_file) != 1)
        return fail("Cannot write header to", path);
    }

    Log(LogLevel::kInfo, "Recorder opened: %s (%u neurons)", dir.c_str(), n_neurons);
    return true;
  }

  void RecordStep(float time_ms, const NeuronArray& neurons,
                  const DriftMetrics* drift,
                  int total_resyncs, float replaced_pct,
                  const std::vector<float>* per_neuron_err) {
    n_recorded_steps++;

    if (spike_file) {
      if (fwrite(&time_ms, sizeof(float), 1, spike_file) != 1 ||
          fwrite(neurons.spiked.data(), sizeof(uint8_t), n_neurons, spike_file) != n_neurons) {
        Log(LogLevel::kError, "Recorder: write error in spikes.bin at step %u", n_recorded_steps);
      }
    }

    if (voltage_file) {
      if (fwrite(&time_ms, sizeof(float), 1, voltage_file) != 1 ||
          fwrite(neurons.v.data(), sizeof(float), n_neurons, voltage_file) != n_neurons) {
        Log(LogLevel::kError, "Recorder: write error in voltages.bin at step %u", n_recorded_steps);
      }
    }

    if (metrics_file) {
      int spike_count = 0;
      for (uint32_t i = 0; i < n_neurons; ++i) spike_count += neurons.spiked[i];

      if (drift) {
        fprintf(metrics_file, "%.4f,%d,%.6f,%.6f,%.6f,%d,%d,%d,%.4f\n",
                time_ms, spike_count,
                drift->spike_correlation, drift->population_rmse,
                drift->mean_v_error,
                drift->n_false_positive, drift->n_false_negative,
                total_resyncs, replaced_pct);
      } else {
        fprintf(metrics_file, "%.4f,%d,,,,,,,%0.4f\n",
                time_ms, spike_count, replaced_pct);
      }
    }

    if (per_neuron_file && per_neuron_err && per_neuron_err->size() >= n_neurons) {
      if (fwrite(&time_ms, sizeof(float), 1, per_neuron_file) != 1 ||
          fwrite(per_neuron_err->data(), sizeof(float), n_neurons, per_neuron_file) != n_neurons) {
        Log(LogLevel::kError, "Recorder: write error in per_neuron_error.bin at step %u", n_recorded_steps);
      }
    }
  }

  void Close() {
    if (spike_file) {
      fseek(spike_file, sizeof(uint32_t), SEEK_SET);
      fwrite(&n_recorded_steps, sizeof(uint32_t), 1, spike_file);
      fclose(spike_file);
      spike_file = nullptr;
    }
    if (voltage_file) {
      fseek(voltage_file, sizeof(uint32_t), SEEK_SET);
      fwrite(&n_recorded_steps, sizeof(uint32_t), 1, voltage_file);
      fclose(voltage_file);
      voltage_file = nullptr;
    }
    if (metrics_file) {
      fclose(metrics_file);
      metrics_file = nullptr;
    }
    if (per_neuron_file) {
      fseek(per_neuron_file, sizeof(uint32_t), SEEK_SET);
      fwrite(&n_recorded_steps, sizeof(uint32_t), 1, per_neuron_file);
      fclose(per_neuron_file);
      per_neuron_file = nullptr;
    }

    Log(LogLevel::kInfo, "Recorder closed: %u steps recorded", n_recorded_steps);
  }

  ~Recorder() { Close(); }
};

}  // namespace mechabrain

#endif  // FWMC_RECORDER_H_
