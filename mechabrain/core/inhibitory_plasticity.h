#ifndef FWMC_INHIBITORY_PLASTICITY_H_
#define FWMC_INHIBITORY_PLASTICITY_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Inhibitory spike-timing-dependent plasticity (iSTDP).
// Vogels et al. 2011, Science 334:1569 (inhibitory plasticity rule).
//
// Standard excitatory STDP uses asymmetric windows (pre-before-post
// potentiates, post-before-pre depresses). This is wrong for inhibitory
// synapses: strengthening inhibition when the post-neuron fires too much
// is the correct homeostatic response.
//
// The Vogels rule uses symmetric potentiation for ANY temporal correlation
// between pre and post spikes, plus a constant depression offset (alpha)
// that sets the target postsynaptic firing rate:
//
//   When post fires: dw += eta * x_pre[s]
//   When pre fires:  dw += eta * (x_post[post] - alpha)
//
// where x_pre, x_post are exponential eligibility traces (tau ~20ms),
// and alpha = 2 * rho_0 * tau controls the target rate rho_0.
//
// This naturally drives inhibitory weights to balance excitation:
// - If post fires too fast: more pre-post coincidences -> inhibition grows
// - If post fires too slow: depression dominates -> inhibition shrinks
//
// Only applied to inhibitory synapses (GABA, glutamatergic inhibitory).
// Excitatory synapses continue to use standard STDP from stdp.h.
struct InhibitorySTDP {
  float eta = 0.001f;          // learning rate
  float tau = 20.0f;           // eligibility trace time constant (ms)
  float target_rate_hz = 5.0f; // target postsynaptic firing rate
  float w_min = 0.0f;          // minimum inhibitory weight
  float w_max = 10.0f;         // maximum inhibitory weight

  // Derived: alpha = 2 * rho_0 * tau (in per-ms units)
  // rho_0 in Hz = spikes/second; tau in ms
  // alpha = 2 * (target_rate_hz / 1000) * tau
  float Alpha() const {
    return 2.0f * (target_rate_hz / 1000.0f) * tau;
  }

  // Per-synapse pre-synaptic eligibility trace.
  // Jumps by 1 when pre fires, decays exponentially.
  std::vector<float> x_pre;

  // Per-neuron post-synaptic eligibility trace.
  // Jumps by 1 when post fires, decays exponentially.
  std::vector<float> x_post;

  void Init(size_t n_synapses, size_t n_neurons) {
    x_pre.assign(n_synapses, 0.0f);
    x_post.assign(n_neurons, 0.0f);
  }

  bool IsInitialized() const { return !x_pre.empty(); }

  float cached_decay = 0.0f;
  float cached_dt = -1.0f;
};

// Update inhibitory synaptic weights using the Vogels iSTDP rule.
// Only modifies synapses with inhibitory NT type (GABA, glutamatergic).
// Call after spike propagation each timestep.
//
// This is separate from STDPUpdate() (stdp.h) which handles excitatory
// plasticity. Both can run simultaneously on the same network.
inline void InhibitorySTDPUpdate(SynapseTable& synapses,
                                  const NeuronArray& neurons,
                                  float dt_ms,
                                  InhibitorySTDP& rule) {
  if (!rule.IsInitialized()) return;

  if (dt_ms != rule.cached_dt) {
    rule.cached_decay = std::exp(-dt_ms / rule.tau);
    rule.cached_dt = dt_ms;
  }
  const float decay = rule.cached_decay;
  const float alpha = rule.Alpha();
  const int n = static_cast<int>(synapses.n_neurons);

  // Decay post-synaptic traces
  for (int i = 0; i < n; ++i) {
    rule.x_post[i] *= decay;
    if (neurons.spiked[i]) {
      rule.x_post[i] += 1.0f;
    }
  }

  // Inhibitory NT types
  constexpr uint8_t kGABA = static_cast<uint8_t>(NTType::kGABA);
  constexpr uint8_t kGlut = static_cast<uint8_t>(NTType::kGlut);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
  #endif
  for (int pre = 0; pre < n; ++pre) {
    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];

    for (uint32_t s = start; s < end; ++s) {
      uint8_t nt = synapses.nt_type[s];
      if (nt != kGABA && nt != kGlut) continue;  // skip excitatory

      // Decay pre-synaptic trace
      rule.x_pre[s] *= decay;

      float dw = 0.0f;
      uint32_t post_idx = synapses.post[s];

      // Post spiked: potentiate based on pre trace
      if (neurons.spiked[post_idx]) {
        dw += rule.eta * rule.x_pre[s];
      }

      // Pre spiked: potentiate based on post trace, depress by alpha
      if (neurons.spiked[pre]) {
        rule.x_pre[s] += 1.0f;
        dw += rule.eta * (rule.x_post[post_idx] - alpha);
      }

      if (dw != 0.0f) {
        synapses.weight[s] = std::clamp(
            synapses.weight[s] + dw, rule.w_min, rule.w_max);
      }
    }
  }
}

}  // namespace mechabrain

#endif  // FWMC_INHIBITORY_PLASTICITY_H_
