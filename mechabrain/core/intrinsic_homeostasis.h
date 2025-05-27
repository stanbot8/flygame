#ifndef FWMC_INTRINSIC_HOMEOSTASIS_H_
#define FWMC_INTRINSIC_HOMEOSTASIS_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "core/neuron_array.h"

namespace mechabrain {

// Target firing rates by cell type (Hz).
// Drosophila neurons have widely varying intrinsic rates:
//   KCs: ~1-2 Hz (sparse coding, only 5-10% fire per odor; Turner et al. 2008)
//   ORNs: ~8 Hz spontaneous (Nagel & Wilson 2011)
//   PNs: ~5-10 Hz (Gouwens & Wilson 2009)
//   LNs: ~4-6 Hz (Chou et al. 2010)
//   MBONs: ~10-15 Hz (Prisco et al. 2021)
//   DANs: ~3-5 Hz baseline (Mao & Davis 2009)
//   FastSpiking: ~15-30 Hz (Chou et al. 2010)
//   Bursting: ~10-20 Hz (Green et al. 2017)
inline float TargetRateForCellType(uint8_t cell_type) {
  switch (cell_type) {
    case 1:  return 2.0f;   // KC: sparse coding
    case 2:  return 12.0f;  // MBON_cholinergic
    case 3:  return 10.0f;  // MBON_gabaergic
    case 4:  return 10.0f;  // MBON_glutamatergic
    case 5:  return 4.0f;   // DAN_PPL1
    case 6:  return 4.0f;   // DAN_PAM
    case 7:  return 8.0f;   // PN_excitatory
    case 8:  return 6.0f;   // PN_inhibitory
    case 9:  return 5.0f;   // LN_local
    case 10: return 8.0f;   // ORN
    case 11: return 20.0f;  // FastSpiking
    case 12: return 15.0f;  // Bursting
    case 13: return 3.0f;   // Serotonergic (tonic low-rate)
    case 14: return 5.0f;   // Octopaminergic
    // Mammalian cortical types (spontaneous rates from in vivo recordings)
    case 15: return 5.0f;   // L2/3 pyramidal (Barth & Bhatt 2012: ~2-8 Hz in vivo)
    case 16: return 8.0f;   // L4 stellate (de Kock et al 2007: ~5-10 Hz)
    case 17: return 8.0f;   // L5 pyramidal (de Kock & Sakmann 2009: ~5-15 Hz)
    case 18: return 3.0f;   // L6 pyramidal (Bortone et al 2014: ~2-5 Hz, sparse)
    case 19: return 40.0f;  // PV basket (Atallah et al 2012: ~30-60 Hz in vivo)
    case 20: return 10.0f;  // SST Martinotti (Gentet et al 2012: ~5-15 Hz)
    case 21: return 8.0f;   // VIP interneuron (Pi et al 2013: ~5-10 Hz)
    case 22: return 15.0f;  // Thalamocortical relay (Sherman 2001: ~10-20 Hz tonic)
    case 23: return 10.0f;  // TRN (Pinault 2004: ~5-15 Hz)
    case 24: return 3.0f;   // Cholinergic NBM (Zaborszky et al 2015: ~2-5 Hz)
    default: return 5.0f;   // Generic
  }
}

// Intrinsic excitability homeostasis (Marder & Goaillard 2006).
// Each neuron maintains a slow bias current that adjusts to keep
// its firing rate near a target. This prevents runaway excitation
// or permanent silence over long simulations.
//
// Mechanism: neurons that fire too much get a negative bias (harder
// to spike), neurons that are silent get a positive bias (easier
// to spike). The adjustment is slow (seconds to minutes) to avoid
// interfering with fast dynamics.
//
// Usage:
//   IntrinsicHomeostasis homeo;
//   homeo.Init(n_neurons, target_rate_hz, dt_ms);
//   // each timestep:
//   homeo.RecordSpikes(neurons);
//   // periodically:
//   homeo.Apply(neurons);
struct IntrinsicHomeostasis {
  float target_rate_hz = 5.0f;    // default target (overridden per-neuron if available)
  float learning_rate = 0.01f;    // pA per Hz deviation per update
  float max_bias = 5.0f;          // clamp magnitude of bias current
  float update_interval_ms = 1000.0f; // how often to adjust

  // Per-neuron state
  std::vector<float> bias_current;     // slow excitability adjustment (pA)
  std::vector<int> spike_count;        // accumulator within current window
  std::vector<float> per_neuron_target; // per-neuron target rate (Hz)
  float accumulated_ms = 0.0f;    // time in current measurement window
  float dt_ms = 1.0f;

  void Init(size_t n_neurons, float target_hz, float sim_dt_ms) {
    target_rate_hz = target_hz;
    dt_ms = sim_dt_ms;
    bias_current.assign(n_neurons, 0.0f);
    spike_count.assign(n_neurons, 0);
    per_neuron_target.assign(n_neurons, target_hz);
    accumulated_ms = 0.0f;
  }

  // Set per-neuron targets from cell type assignments.
  // Call after Init and after neuron types are assigned.
  void SetTargetsFromTypes(const NeuronArray& neurons) {
    per_neuron_target.resize(neurons.n);
    for (size_t i = 0; i < neurons.n; ++i) {
      per_neuron_target[i] = TargetRateForCellType(neurons.type[i]);
    }
  }

  // Call each timestep to accumulate spike counts.
  void RecordSpikes(const NeuronArray& neurons) {
    for (size_t i = 0; i < neurons.n; ++i) {
      spike_count[i] += neurons.spiked[i];
    }
    accumulated_ms += dt_ms;
  }

  // Check if enough time has elapsed and apply adjustment.
  // Returns true if an update was performed.
  bool MaybeApply(NeuronArray& neurons) {
    if (accumulated_ms < update_interval_ms) return false;
    Apply(neurons);
    return true;
  }

  // Apply the homeostatic adjustment now.
  void Apply(NeuronArray& neurons) {
    if (accumulated_ms <= 0.0f) return;
    float window_s = accumulated_ms / 1000.0f;

    for (size_t i = 0; i < neurons.n; ++i) {
      float rate = static_cast<float>(spike_count[i]) / window_s;
      float target = (i < per_neuron_target.size()) ? per_neuron_target[i] : target_rate_hz;
      float error = target - rate;

      // Adjust bias: positive error (too silent) -> increase bias
      //              negative error (too active) -> decrease bias
      bias_current[i] += learning_rate * error;
      bias_current[i] = std::clamp(bias_current[i], -max_bias, max_bias);

      // Inject bias as external current
      neurons.i_ext[i] += bias_current[i];
    }

    // Reset accumulators
    std::fill(spike_count.begin(), spike_count.end(), 0);
    accumulated_ms = 0.0f;
  }

  // Get the mean bias current across all neurons.
  float MeanBias() const {
    if (bias_current.empty()) return 0.0f;
    float sum = 0.0f;
    for (float b : bias_current) sum += b;
    return sum / static_cast<float>(bias_current.size());
  }

  // Get the fraction of neurons with positive bias (excitability up).
  float FractionExcited() const {
    if (bias_current.empty()) return 0.0f;
    int count = 0;
    for (float b : bias_current) {
      if (b > 0.0f) count++;
    }
    return static_cast<float>(count) / static_cast<float>(bias_current.size());
  }
};

}  // namespace mechabrain

#endif  // FWMC_INTRINSIC_HOMEOSTASIS_H_
