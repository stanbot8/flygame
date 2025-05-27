#ifndef FWMC_NEUROMODULATOR_EFFECTS_H_
#define FWMC_NEUROMODULATOR_EFFECTS_H_

#include "core/neuron_array.h"

namespace mechabrain {

// Neuromodulatory effects on neuronal excitability.
//
// Neuromodulator concentrations (DA, 5HT, OA) are tracked per-neuron
// and updated by NeuromodulatorUpdate() in stdp.h. This function
// converts those concentrations into currents that modulate excitability.
//
// Without this, neuromodulators only gate STDP. In reality, they
// continuously modulate neuronal state:
//
// Octopamine (OA): Drosophila analog of norepinephrine.
//   Increases excitability via OAMB/Oct-beta receptors -> cAMP -> PKA.
//   Behavioral: arousal, flight initiation, aggression.
//   Effect: depolarizing current up to ~3 pA at saturating OA.
//   References: Roeder 2005; Claridge-Chang et al. 2009
//
// Serotonin (5HT): mixed effects depending on receptor subtype.
//   d5HT1A/1B (Gi-coupled): inhibitory, lowers excitability.
//   d5HT2A/7 (Gq/Gs-coupled): excitatory.
//   Net population effect in Drosophila MB: mild inhibition.
//   Behavioral: sleep/wake, feeding, aggression suppression.
//   Effect: mild hyperpolarizing current up to ~2 pA.
//   References: Pooryasin & Bhatt 2021; Yuan et al. 2006
//
// Dopamine (DA): modulates via D1-like (DopR1, excitatory) and
//   D2-like (DopR2/DD2R, inhibitory) receptors.
//   In MB Kenyon cells: primarily D1 (DopR1) -> slight excitation.
//   In MBONs: mixed, context-dependent.
//   Besides gating STDP, DA directly modulates membrane properties.
//   Effect: mild depolarizing current up to ~2 pA.
//   References: Kim et al. 2007; Berry et al. 2012
struct NeuromodulatorEffects {
  // Current injected per unit concentration (pA equivalent).
  // Positive = depolarizing (excitatory), negative = hyperpolarizing.
  float oa_current = 3.0f;     // octopamine: strong arousal signal
  float serotonin_current = -2.0f;  // serotonin: net inhibitory
  float da_current = 1.5f;     // dopamine: mild excitation (D1-like)

  // Threshold: ignore negligible concentrations to avoid FP noise
  float min_concentration = 0.01f;

  // Synaptic gain modulation: OA enhances synaptic transmission
  // (Farooqui et al. 2004; OA increases evoked PSP amplitude ~20-40%)
  float oa_syn_gain = 0.3f;  // fractional increase in i_syn per unit OA

  // Apply neuromodulatory currents to external input.
  // Call after NeuromodulatorUpdate() and before IzhikevichStep().
  void Apply(NeuronArray& neurons) const {
    for (size_t i = 0; i < neurons.n; ++i) {
      float oa = neurons.octopamine[i];
      float ser = neurons.serotonin[i];
      float da = neurons.dopamine[i];

      // Direct excitability modulation via tonic currents
      if (oa > min_concentration)
        neurons.i_ext[i] += oa * oa_current;
      if (ser > min_concentration)
        neurons.i_ext[i] += ser * serotonin_current;
      if (da > min_concentration)
        neurons.i_ext[i] += da * da_current;

      // OA-mediated synaptic gain: amplify existing synaptic input
      if (oa > min_concentration)
        neurons.i_syn[i] *= (1.0f + oa * oa_syn_gain);
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_NEUROMODULATOR_EFFECTS_H_
