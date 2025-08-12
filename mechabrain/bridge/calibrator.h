#ifndef FWMC_CALIBRATOR_H_
#define FWMC_CALIBRATOR_H_

#include <cmath>
#include <vector>
#include "bridge/bridge_channel.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Supervised calibration: adjusts synaptic weights to minimize
// prediction error against ground truth biological recordings.
//
// Phase 1 of the research protocol requires per-fly calibration.
// This implements a simple gradient-free approach:
//   1. Run a trial, measure per-synapse prediction error
//   2. For each synapse, if pre spiked and post prediction was wrong,
//      adjust weight in the direction that would fix it
//   3. Repeat for multiple trials
//
// This is NOT backpropagation (can't differentiate through Izhikevich).
// It's a biologically-plausible perturbation-based method.
struct Calibrator {
  float learning_rate = 0.001f;
  float momentum = 0.9f;
  float weight_decay = 1e-5f;

  // Per-synapse error accumulator and velocity (for momentum)
  std::vector<float> weight_velocity;
  std::vector<float> error_accum;
  int n_samples = 0;

  void Init(size_t n_synapses) {
    weight_velocity.assign(n_synapses, 0.0f);
    error_accum.assign(n_synapses, 0.0f);
    n_samples = 0;
  }

  // Accumulate error for one timestep.
  // For each synapse where the pre-neuron spiked, compute the
  // prediction error at the post-neuron and attribute it to this synapse.
  void AccumulateError(const SynapseTable& synapses,
                       const NeuronArray& digital,
                       const std::vector<BioReading>& bio) {
    if (bio.empty()) return;

    // Build a lookup: neuron_idx -> bio spike_prob
    std::vector<float> bio_prob(digital.n, -1.0f);
    for (const auto& b : bio) {
      if (b.neuron_idx < digital.n) {
        bio_prob[b.neuron_idx] = b.spike_prob;
      }
    }

    for (size_t pre = 0; pre < synapses.n_neurons; ++pre) {
      if (!digital.spiked[pre]) continue;

      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];

      for (uint32_t s = start; s < end; ++s) {
        uint32_t post_idx = synapses.post[s];
        if (bio_prob[post_idx] < 0.0f) continue;  // no bio data for this neuron

        // Error: digital prediction vs biological ground truth
        float digital_spike = digital.spiked[post_idx] ? 1.0f : 0.0f;
        float error = digital_spike - bio_prob[post_idx];

        // Attribute error to this synapse (signed: positive = too active)
        error_accum[s] += error;
      }
    }
    n_samples++;
  }

  // Apply accumulated gradients to synaptic weights.
  // Call this after accumulating errors over a trial.
  void ApplyGradients(SynapseTable& synapses) {
    if (n_samples == 0) return;

    float inv_n = 1.0f / static_cast<float>(n_samples);
    size_t n_updated = 0;

    for (size_t s = 0; s < synapses.Size(); ++s) {
      float grad = error_accum[s] * inv_n;

      // Momentum SGD
      weight_velocity[s] = momentum * weight_velocity[s] - learning_rate * grad;

      // Weight decay (L2 regularization)
      weight_velocity[s] -= weight_decay * synapses.weight[s];

      float new_w = synapses.weight[s] + weight_velocity[s];

      // Clamp magnitude to valid range (preserve sign for inhibitory synapses)
      new_w = std::clamp(new_w, -20.0f, 20.0f);

      if (new_w != synapses.weight[s]) n_updated++;
      synapses.weight[s] = new_w;
    }

    Log(LogLevel::kInfo, "Calibration: %zu/%zu weights updated (lr=%.4f, %d samples)",
        n_updated, synapses.Size(), learning_rate, n_samples);

    // Reset accumulators
    std::fill(error_accum.begin(), error_accum.end(), 0.0f);
    n_samples = 0;
  }

  // Compute mean prediction error across all observed neurons
  float MeanError(const NeuronArray& digital,
                  const std::vector<BioReading>& bio) const {
    if (bio.empty()) return 0.0f;
    float total = 0.0f;
    int count = 0;
    for (const auto& b : bio) {
      if (b.neuron_idx >= digital.n) continue;
      float pred = digital.spiked[b.neuron_idx] ? 1.0f : 0.0f;
      total += std::abs(pred - b.spike_prob);
      count++;
    }
    return count > 0 ? total / count : 0.0f;
  }
};

}  // namespace mechabrain

#endif  // FWMC_CALIBRATOR_H_
