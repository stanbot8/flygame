#ifndef FWMC_CONNECTOME_STATS_H_
#define FWMC_CONNECTOME_STATS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "core/log.h"
#include "core/synapse_table.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Connectome validation and statistics.
// Run after loading to verify data integrity and log structural properties.
struct ConnectomeStats {
  size_t n_neurons = 0;
  size_t n_synapses = 0;

  // Degree statistics
  uint32_t min_out_degree = 0;
  uint32_t max_out_degree = 0;
  float mean_out_degree = 0;
  float median_out_degree = 0;

  uint32_t min_in_degree = 0;
  uint32_t max_in_degree = 0;
  float mean_in_degree = 0;

  // NT type distribution
  size_t n_ach = 0;
  size_t n_gaba = 0;
  size_t n_glut = 0;
  size_t n_da = 0;
  size_t n_5ht = 0;
  size_t n_oa = 0;
  size_t n_unknown_nt = 0;

  // Weight statistics
  float min_weight = 0;
  float max_weight = 0;
  float mean_weight = 0;

  // Validation results
  int n_self_loops = 0;
  int n_out_of_bounds = 0;
  int n_nan_weights = 0;
  int n_zero_weights = 0;
  int n_isolated_neurons = 0;  // no incoming or outgoing synapses

  // Compute all statistics and validate the connectome.
  // Returns false if critical errors found (out-of-bounds indices).
  bool Compute(const SynapseTable& synapses, const NeuronArray& /*neurons*/) {
    n_neurons = synapses.n_neurons;
    n_synapses = synapses.Size();

    if (n_neurons == 0 || n_synapses == 0) {
      Log(LogLevel::kWarn, "ConnectomeStats: empty connectome (%zu neurons, %zu synapses)",
          n_neurons, n_synapses);
      return n_neurons > 0;  // empty synapses is ok, zero neurons is not
    }

    // Out-degree from CSR row_ptr
    std::vector<uint32_t> out_degrees(n_neurons);
    for (size_t i = 0; i < n_neurons; ++i) {
      out_degrees[i] = synapses.row_ptr[i + 1] - synapses.row_ptr[i];
    }

    // In-degree by scanning post targets
    std::vector<uint32_t> in_degrees(n_neurons, 0);
    for (size_t s = 0; s < n_synapses; ++s) {
      if (synapses.post[s] < n_neurons) {
        in_degrees[synapses.post[s]]++;
      }
    }

    // Degree stats
    auto [out_min, out_max] = std::minmax_element(out_degrees.begin(), out_degrees.end());
    min_out_degree = *out_min;
    max_out_degree = *out_max;
    float sum_out = 0;
    for (auto d : out_degrees) sum_out += d;
    mean_out_degree = sum_out / n_neurons;

    std::sort(out_degrees.begin(), out_degrees.end());
    median_out_degree = (n_neurons % 2 == 0)
        ? (out_degrees[n_neurons / 2 - 1] + out_degrees[n_neurons / 2]) / 2.0f
        : static_cast<float>(out_degrees[n_neurons / 2]);

    auto [in_min, in_max] = std::minmax_element(in_degrees.begin(), in_degrees.end());
    min_in_degree = *in_min;
    max_in_degree = *in_max;
    float sum_in = 0;
    for (auto d : in_degrees) sum_in += d;
    mean_in_degree = sum_in / n_neurons;

    // Isolated neurons (no in or out connections)
    for (size_t i = 0; i < n_neurons; ++i) {
      uint32_t out_d = synapses.row_ptr[i + 1] - synapses.row_ptr[i];
      if (out_d == 0 && in_degrees[i] == 0) {
        n_isolated_neurons++;
      }
    }

    // NT type distribution
    for (size_t s = 0; s < n_synapses; ++s) {
      switch (synapses.nt_type[s]) {
        case kACh:  n_ach++; break;
        case kGABA: n_gaba++; break;
        case kGlut: n_glut++; break;
        case kDA:   n_da++; break;
        case k5HT:  n_5ht++; break;
        case kOA:   n_oa++; break;
        default:    n_unknown_nt++; break;
      }
    }

    // Weight stats and validation
    min_weight = synapses.weight[0];
    max_weight = synapses.weight[0];
    float sum_w = 0;
    for (size_t s = 0; s < n_synapses; ++s) {
      float w = synapses.weight[s];
      if (std::isnan(w)) { n_nan_weights++; continue; }
      if (w == 0.0f) n_zero_weights++;
      if (w < min_weight) min_weight = w;
      if (w > max_weight) max_weight = w;
      sum_w += w;
    }
    size_t valid_weights = n_synapses - n_nan_weights;
    mean_weight = (valid_weights > 0) ? sum_w / static_cast<float>(valid_weights) : 0.0f;

    // Validate: check for self-loops and out-of-bounds post indices
    for (size_t pre = 0; pre < n_neurons; ++pre) {
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        if (synapses.post[s] == static_cast<uint32_t>(pre)) n_self_loops++;
        if (synapses.post[s] >= n_neurons) n_out_of_bounds++;
      }
    }

    return n_out_of_bounds == 0 && n_nan_weights == 0;
  }

  void LogSummary() const {
    Log(LogLevel::kInfo, "=== Connectome Statistics ===");
    Log(LogLevel::kInfo, "Neurons: %zu  Synapses: %zu  Density: %.4f%%",
        n_neurons, n_synapses,
        100.0 * n_synapses / (static_cast<double>(n_neurons) * n_neurons));

    Log(LogLevel::kInfo, "Out-degree: min=%u max=%u mean=%.1f median=%.1f",
        min_out_degree, max_out_degree, mean_out_degree, median_out_degree);
    Log(LogLevel::kInfo, "In-degree:  min=%u max=%u mean=%.1f",
        min_in_degree, max_in_degree, mean_in_degree);

    Log(LogLevel::kInfo, "NT types: ACh=%zu GABA=%zu Glut=%zu DA=%zu 5HT=%zu OA=%zu unknown=%zu",
        n_ach, n_gaba, n_glut, n_da, n_5ht, n_oa, n_unknown_nt);
    if (n_synapses > 0) {
      Log(LogLevel::kInfo, "NT ratio: %.1f%% excitatory, %.1f%% inhibitory, %.1f%% modulatory",
          100.0 * (n_ach + n_glut) / n_synapses,
          100.0 * n_gaba / n_synapses,
          100.0 * (n_da + n_5ht + n_oa) / n_synapses);
    }

    Log(LogLevel::kInfo, "Weights: min=%.4f max=%.4f mean=%.4f",
        min_weight, max_weight, mean_weight);

    Log(LogLevel::kInfo, "Isolated neurons: %d  Self-loops: %d  Zero-weight: %d",
        n_isolated_neurons, n_self_loops, n_zero_weights);

    if (n_out_of_bounds > 0) {
      Log(LogLevel::kError, "VALIDATION FAILED: %d out-of-bounds post indices", n_out_of_bounds);
    }
    if (n_nan_weights > 0) {
      Log(LogLevel::kError, "VALIDATION FAILED: %d NaN weights", n_nan_weights);
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_CONNECTOME_STATS_H_
