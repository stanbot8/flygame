#ifndef FWMC_SPIKE_FREQUENCY_ADAPTATION_H_
#define FWMC_SPIKE_FREQUENCY_ADAPTATION_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "core/neuron_array.h"

namespace mechabrain {

// Spike-frequency adaptation via slow K+ current (sAHP).
//
// This is distinct from the Izhikevich recovery variable u:
// - u operates on the ~10ms timescale (fast adaptation)
// - sAHP operates on the ~100-1000ms timescale (slow adaptation)
//
// Biological basis: calcium-activated K+ channels (SK/BK channels)
// accumulate calcium with each spike. The resulting hyperpolarizing
// current reduces firing rate over sustained input, producing the
// classic "adaptation" where a neuron's response decays from an
// initial burst to a lower sustained rate.
//
// In Drosophila, SK channels (dSK/KCNMB) are expressed in many
// central neurons and contribute to interspike interval regulation
// (Abou Tayoun et al. 2011, Ping et al. 2011).
//
// Usage:
//   SpikeFrequencyAdaptation sfa;
//   sfa.Init(n_neurons);
//   // each timestep, after IzhikevichStep:
//   sfa.Update(neurons, dt_ms);
struct SpikeFrequencyAdaptation {
  // Per-neuron calcium-like state (arbitrary units, tracks spike history)
  std::vector<float> calcium;

  // Parameters
  float tau_calcium_ms = 300.0f;   // calcium decay time constant
                                    // ~300ms matches dSK channel kinetics
  float calcium_increment = 1.0f;  // calcium added per spike
  float g_sahp = 0.5f;             // sAHP conductance (pA per unit calcium)
  float max_current = 5.0f;        // clamp to prevent runaway inhibition

  bool initialized = false;
  float cached_decay = 0.0f;
  float cached_dt = -1.0f;

  void Init(size_t n_neurons) {
    calcium.assign(n_neurons, 0.0f);
    initialized = true;
  }

  // Update calcium and inject sAHP current.
  // Call after IzhikevichStep each timestep.
  void Update(NeuronArray& neurons, float dt_ms) {
    if (!initialized) return;

    if (dt_ms != cached_dt) {
      cached_decay = std::exp(-dt_ms / tau_calcium_ms);
      cached_dt = dt_ms;
    }
    float decay = cached_decay;

    for (size_t i = 0; i < neurons.n; ++i) {
      // Accumulate calcium on spikes
      if (neurons.spiked[i]) {
        calcium[i] += calcium_increment;
      }

      // Inject hyperpolarizing sAHP current
      float i_sahp = -g_sahp * calcium[i];
      i_sahp = std::max(-max_current, i_sahp);
      neurons.i_ext[i] += i_sahp;

      // Decay calcium
      calcium[i] *= decay;
    }
  }

  // Get mean calcium level (diagnostic)
  float MeanCalcium() const {
    if (calcium.empty()) return 0.0f;
    float sum = 0.0f;
    for (float c : calcium) sum += c;
    return sum / static_cast<float>(calcium.size());
  }

  void Clear() {
    std::fill(calcium.begin(), calcium.end(), 0.0f);
  }
};

}  // namespace mechabrain

#endif  // FWMC_SPIKE_FREQUENCY_ADAPTATION_H_
