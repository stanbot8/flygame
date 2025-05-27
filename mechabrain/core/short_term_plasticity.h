#ifndef FWMC_SHORT_TERM_PLASTICITY_H_
#define FWMC_SHORT_TERM_PLASTICITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Preset STP parameter factories.
// STPParams is defined in synapse_table.h. These helpers return
// biologically motivated configurations.
//
// Drosophila presets grounded in NMJ electrophysiology:
//   Release probability ~0.5 at 1mM Ca2+ (Kittel et al. 2006 Science)
//   ~300 readily releasable vesicles (Hallermann et al. 2010 PNAS)
//   Rapid recovery tau ~40ms (Hallermann et al. 2010)
//   Facilitation prominent at low pr (0.4mM Ca2+), subsides at high pr
//
// Mammalian presets use longer tau_d (200ms) per Markram et al. 1998.

// --- Drosophila presets ---

// Drosophila facilitating: low release probability (sparse KC synapses),
// fast vesicle recovery, moderate facilitation decay. PN->KC contacts
// are sparse (~7 per PN, Caron et al. 2013) with low pr.
inline STPParams STPFacilitatingDrosophila() {
  return {.U_se = 0.15f, .tau_d = 40.0f, .tau_f = 200.0f};
}

// Drosophila depressing: high release probability (NMJ, LN synapses),
// fast vesicle replenishment matching Hallermann et al. 2010.
// KC->MBON also depression-dominant (Hige et al. 2015).
inline STPParams STPDepressingDrosophila() {
  return {.U_se = 0.5f, .tau_d = 40.0f, .tau_f = 30.0f};
}

// Drosophila combined: intermediate pr with both facilitation and
// depression. Seen at some central synapses with mixed dynamics.
inline STPParams STPCombinedDrosophila() {
  return {.U_se = 0.3f, .tau_d = 40.0f, .tau_f = 100.0f};
}

// --- Mammalian presets (Markram et al. 1998; Tsodyks & Markram 1997) ---

// Mammalian facilitating: cortical E-to-E, low pr, very slow
// facilitation decay (~1500ms residual calcium).
inline STPParams STPFacilitating() {
  return {.U_se = 0.15f, .tau_d = 200.0f, .tau_f = 1500.0f};
}

// Mammalian depressing: thalamocortical, sensory-to-interneuron.
// High pr, slow vesicle replenishment.
inline STPParams STPDepressing() {
  return {.U_se = 0.5f, .tau_d = 200.0f, .tau_f = 50.0f};
}

// Mammalian combined: cortical E-to-I, non-monotonic response.
inline STPParams STPCombined() {
  return {.U_se = 0.25f, .tau_d = 150.0f, .tau_f = 500.0f};
}

// Update STP state for all synapses in a SynapseTable.
//
// Call this once per timestep, after spike propagation. For each synapse:
//   1. Relax u toward U_se and x toward 1.0 (exponential decay)
//   2. If pre-neuron spiked: u += U_se*(1-u), x -= u*x
//
// Requires SynapseTable::HasSTP() == true (call InitSTP first).
// OpenMP parallelized for large synapse counts.
inline void UpdateSTP(SynapseTable& synapses, const NeuronArray& neurons,
                      float dt_ms) {
  if (!synapses.HasSTP()) return;
  const int n_pre = static_cast<int>(synapses.n_neurons);

  // Phase 1: exponential relaxation using cached alpha factors.
  // RecoverSTP caches exp(-dt/tau) per synapse, recomputing only on dt change.
  synapses.RecoverSTP(dt_ms);

  // Phase 2: spike updates (iterate by pre-neuron for CSR locality)
  for (int pre = 0; pre < n_pre; ++pre) {
    if (!neurons.spiked[pre]) continue;
    const uint32_t start = synapses.row_ptr[pre];
    const uint32_t end   = synapses.row_ptr[pre + 1];
    for (uint32_t s = start; s < end; ++s) {
      float U = synapses.stp_U_se[s];
      synapses.stp_u[s] += U * (1.0f - synapses.stp_u[s]);
      synapses.stp_u[s] = std::clamp(synapses.stp_u[s], 0.0f, 1.0f);
      float ux = synapses.stp_u[s] * synapses.stp_x[s];
      synapses.stp_x[s] = std::max(0.0f, synapses.stp_x[s] - ux);
    }
  }
}

// Reset all STP state to resting values without reallocating.
inline void ResetSTP(SynapseTable& synapses) {
  if (!synapses.HasSTP()) return;
  for (size_t s = 0; s < synapses.Size(); ++s) {
    synapses.stp_u[s] = synapses.stp_U_se[s];
    synapses.stp_x[s] = 1.0f;
  }
}

// Diagnostic: mean utilization across all synapses.
inline float MeanSTPUtilization(const SynapseTable& synapses) {
  if (!synapses.HasSTP() || synapses.Size() == 0) return 0.0f;
  float sum = 0.0f;
  for (size_t s = 0; s < synapses.Size(); ++s) sum += synapses.stp_u[s];
  return sum / static_cast<float>(synapses.Size());
}

// Diagnostic: mean available resources across all synapses.
inline float MeanSTPResources(const SynapseTable& synapses) {
  if (!synapses.HasSTP() || synapses.Size() == 0) return 0.0f;
  float sum = 0.0f;
  for (size_t s = 0; s < synapses.Size(); ++s) sum += synapses.stp_x[s];
  return sum / static_cast<float>(synapses.Size());
}

}  // namespace mechabrain

#endif  // FWMC_SHORT_TERM_PLASTICITY_H_
