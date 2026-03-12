#ifndef FWMC_STRUCTURAL_PLASTICITY_H_
#define FWMC_STRUCTURAL_PLASTICITY_H_

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/cell_types.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Structural plasticity: synapse pruning and sprouting.
// Expensive operations; only runs every update_interval steps.
// Pruning zeroes out weak weights in-place (CSR topology unchanged).
// Sprouting collects new COO entries and rebuilds the CSR.
struct StructuralPlasticity {
  struct Config {
    float prune_threshold = 0.05f;       // prune synapses with |weight| below this
    float sprout_rate = 0.001f;          // probability of new synapse per neuron pair per update
    int update_interval = 100000;        // steps between structural updates (~10s at 0.1ms dt)
                                         // structural plasticity operates on minutes-to-hours timescale
    uint32_t max_synapses_per_neuron = 100;
  };

  Config config;

  // Mark weak synapses by setting weight to 0.
  // Actual removal would require a CSR rebuild; zeroed weights are
  // effectively dead (contribute no current in PropagateSpikes).
  // Returns count of synapses pruned.
  size_t PruneWeak(SynapseTable& syn) {
    size_t pruned = 0;
    for (size_t i = 0; i < syn.Size(); ++i) {
      if (syn.weight[i] != 0.0f && std::fabs(syn.weight[i]) < config.prune_threshold) {
        syn.weight[i] = 0.0f;
        ++pruned;
      }
    }
    Log(LogLevel::kInfo, "structural_plasticity: pruned %llu weak synapses (threshold %.4f)",
        static_cast<unsigned long long>(pruned), config.prune_threshold);
    return pruned;
  }

  // Sprout new synapses between correlated neurons (both spiked recently).
  // Since CSR is immutable after build, we extract existing data into COO,
  // append new entries, and rebuild.  Returns count of synapses sprouted.
  size_t SproutNew(SynapseTable& syn, NeuronArray& neurons, std::mt19937& rng) {
    // Collect indices of neurons that spiked this step
    std::vector<uint32_t> active;
    active.reserve(256);
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.spiked[i]) {
        active.push_back(static_cast<uint32_t>(i));
      }
    }

    if (active.size() < 2) {
      Log(LogLevel::kDebug, "structural_plasticity: sprout skipped, <2 active neurons");
      return 0;
    }

    // Count existing outgoing synapses per neuron (exclude zeroed-out weights)
    std::vector<uint32_t> out_degree(syn.n_neurons, 0);
    for (size_t pre = 0; pre < syn.n_neurons; ++pre) {
      uint32_t start = syn.row_ptr[pre];
      uint32_t end = syn.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        if (syn.weight[s] != 0.0f) ++out_degree[pre];
      }
    }

    // Extract current COO from existing CSR (skip dead synapses)
    std::vector<uint32_t> coo_pre, coo_post;
    std::vector<float> coo_weight;
    std::vector<uint8_t> coo_nt;
    coo_pre.reserve(syn.Size());
    coo_post.reserve(syn.Size());
    coo_weight.reserve(syn.Size());
    coo_nt.reserve(syn.Size());

    for (size_t pre = 0; pre < syn.n_neurons; ++pre) {
      uint32_t start = syn.row_ptr[pre];
      uint32_t end = syn.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        if (syn.weight[s] != 0.0f) {
          coo_pre.push_back(static_cast<uint32_t>(pre));
          coo_post.push_back(syn.post[s]);
          coo_weight.push_back(syn.weight[s]);
          coo_nt.push_back(syn.nt_type[s]);
        }
      }
    }

    // Try to sprout between pairs of active neurons
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    size_t sprouted = 0;

    for (size_t i = 0; i < active.size(); ++i) {
      uint32_t a = active[i];
      if (out_degree[a] >= config.max_synapses_per_neuron) continue;

      for (size_t j = 0; j < active.size(); ++j) {
        if (i == j) continue;
        uint32_t b = active[j];

        if (dist(rng) > config.sprout_rate) continue;
        if (out_degree[a] >= config.max_synapses_per_neuron) break;

        // Add new excitatory synapse a -> b with small weight
        coo_pre.push_back(a);
        coo_post.push_back(b);
        coo_weight.push_back(config.prune_threshold);  // start at threshold
        coo_nt.push_back(kACh);
        ++out_degree[a];
        ++sprouted;
      }
    }

    if (sprouted > 0) {
      syn.BuildFromCOO(syn.n_neurons, coo_pre, coo_post, coo_weight, coo_nt);
    }

    Log(LogLevel::kInfo, "structural_plasticity: sprouted %llu new synapses (%llu active neurons)",
        static_cast<unsigned long long>(sprouted),
        static_cast<unsigned long long>(active.size()));
    return sprouted;
  }

  // Called each simulation step.  Only acts on update_interval boundaries.
  void Update(SynapseTable& syn, NeuronArray& neurons, int step, std::mt19937& rng) {
    if (config.update_interval <= 0) return;
    if (step % config.update_interval != 0) return;

    PruneWeak(syn);
    SproutNew(syn, neurons, rng);
  }
};

}  // namespace mechabrain

#endif  // FWMC_STRUCTURAL_PLASTICITY_H_
