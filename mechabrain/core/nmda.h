#ifndef FWMC_NMDA_H_
#define FWMC_NMDA_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// NMDA receptor dynamics with voltage-dependent Mg2+ block.
//
// At excitatory synapses, neurotransmitter activates both fast (AMPA-like)
// and slow (NMDA-like) ionotropic receptors. The NMDA receptor has three
// distinguishing properties:
//
//  1. Voltage-dependent Mg2+ block: at resting potential (~-65mV), Mg2+
//     ions plug the channel pore. Depolarization expels them.
//     B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
//     (Jahr & Stevens 1990, J Neurosci 10:1830)
//
//  2. Slow kinetics: NMDA channels decay 10-50x slower than AMPA
//     (tau_NMDA ~50-100ms vs tau_AMPA ~2-3ms). This provides temporal
//     integration over ~100ms windows.
//
//  3. Calcium permeability: open NMDA channels admit Ca2+, driving
//     CaMKII/calcineurin signaling cascades for LTP/LTD induction.
//
// Together these create a coincidence detector: NMDA current flows only
// when BOTH presynaptic activity (transmitter binding) AND postsynaptic
// depolarization (Mg block removed) are present. This is the molecular
// basis for Hebbian associative learning.
//
// Drosophila: Nmdar1 (dNR1) and Nmdar2 (dNR2) are expressed throughout
// the brain and required for long-term memory (Xia et al. 2005, Curr Biol;
// Wu et al. 2007). Mg2+ block is conserved. In the fly, ACh is the
// primary fast excitatory transmitter (analogous to glutamate in mammals),
// and cholinergic synapses have both fast (nAChR, AMPA-like) and slow
// (Nmdar, NMDA-like) components.
//
// Usage:
//   NMDAReceptor nmda;
//   nmda.Init(n_neurons);
//   // each timestep, after PropagateSpikes:
//   nmda.AccumulateFromSpikes(synapses, spiked, weight_scale);
//   nmda.Step(neurons, dt_ms);
struct NMDAReceptor {
  // Per-neuron state
  std::vector<float> g_nmda;    // NMDA conductance pool (decays slowly)
  std::vector<float> ca_nmda;   // NMDA-mediated intracellular calcium

  // Kinetic parameters
  float tau_nmda_ms = 80.0f;    // NMDA decay time constant (ms)
                                  // Mammalian NR2A: ~50ms, NR2B: ~300ms
                                  // (Cull-Candy et al. 2001, Curr Opin Neurobiol)
                                  // Drosophila dNR1/dNR2: ~60-100ms
                                  // Default 80ms: intermediate NR2A/2B mixture

  float mg_conc_mM = 1.0f;      // Extracellular [Mg2+] (mM)
                                  // Physiological: 1.0-1.2 mM (mammal/fly)

  float nmda_gain = 0.35f;      // NMDA conductance gain relative to AMPA weight
                                  // AMPA:NMDA charge ratio ~2:1 at cortical synapses
                                  // (Myme et al. 2003, J Neurophysiol)
                                  // 0.35 calibrated so that at moderate
                                  // depolarization (V=-40mV, B=0.23), NMDA
                                  // contributes ~8% additional current

  float ca_per_charge = 0.1f;   // Ca2+ entry per unit NMDA current (a.u.)
                                  // ~10-15% of NMDA current is carried by Ca2+
                                  // (Schneggenburger et al. 1993, J Physiol)

  float tau_ca_nmda_ms = 200.0f; // NMDA-calcium decay time constant (ms)
                                   // Spine Ca2+ transients: ~100-300ms
                                   // (Sabatini et al. 2002, Neuron)

  float max_g = 20.0f;           // Conductance clamp to prevent runaway

  bool initialized = false;

  void Init(size_t n) {
    g_nmda.assign(n, 0.0f);
    ca_nmda.assign(n, 0.0f);
    initialized = true;
  }

  // Mg2+ voltage-dependent block (Jahr & Stevens 1990).
  // Returns fraction of NMDA channels that are unblocked [0, 1].
  //
  // At V = -65mV: B = 0.06 (nearly fully blocked at rest)
  // At V = -40mV: B = 0.23 (partially open during EPSPs)
  // At V = -20mV: B = 0.50 (half open during dendritic spikes)
  // At V =   0mV: B = 0.78 (mostly open during somatic spikes)
  static float MgBlock(float v_mV, float mg_mM) {
    return 1.0f / (1.0f + (mg_mM / 3.57f) * std::exp(-0.062f * v_mV));
  }

  // Accumulate NMDA conductance from excitatory spikes.
  // Does a CSR walk over spiked neurons, processing only excitatory
  // synapses (kACh). Modulatory NTs (DA, 5HT, OA) and inhibitory NTs
  // (GABA, GluCl) do not activate NMDA receptors.
  //
  // Call after PropagateSpikes each timestep. The CSR data is already
  // in cache from the main propagation pass, so this second walk is
  // cheap (only processes spiked neurons, typically <5% of population).
  void AccumulateFromSpikes(const SynapseTable& synapses,
                            const uint8_t* spiked,
                            float weight_scale) {
    if (!initialized) return;
    const int n = static_cast<int>(synapses.n_neurons);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
    #endif
    for (int pre = 0; pre < n; ++pre) {
      if (!spiked[pre]) continue;
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        // Only excitatory (cholinergic) synapses activate NMDA receptors.
        // In Drosophila, ACh is the primary excitatory NT.
        // For mammalian models, this would be kGlut with positive sign.
        if (synapses.nt_type[s] != kACh) continue;

        float val = synapses.weight[s] * weight_scale * nmda_gain;
        uint32_t post_idx = synapses.post[s];

        #ifdef _OPENMP
        #pragma omp atomic
        #endif
        g_nmda[post_idx] += val;
      }
    }
  }

  // Step: decay NMDA conductance, apply voltage-dependent Mg2+ block,
  // inject NMDA current into neurons, update NMDA-calcium.
  //
  // The voltage dependency creates a positive feedback loop:
  // depolarization -> Mg block removed -> more NMDA current -> more
  // depolarization. This is bounded by the slow decay and max_g clamp.
  void Step(NeuronArray& neurons, float dt_ms) {
    if (!initialized) return;

    float decay = std::exp(-dt_ms / tau_nmda_ms);
    float ca_decay = std::exp(-dt_ms / tau_ca_nmda_ms);

    for (size_t i = 0; i < neurons.n; ++i) {
      // Skip neurons with no NMDA conductance (typically ~95% of population).
      // Avoids the expensive MgBlock exp() call for silent neurons.
      if (g_nmda[i] == 0.0f) {
        ca_nmda[i] *= ca_decay;
        continue;
      }

      // Voltage-dependent Mg2+ block
      float B = MgBlock(neurons.v[i], mg_conc_mM);

      // NMDA current = slow conductance pool * unblock fraction
      float i_nmda = g_nmda[i] * B;
      neurons.i_ext[i] += i_nmda;

      // NMDA-mediated calcium influx (only from positive current)
      float ca_entry = ca_per_charge * std::max(0.0f, i_nmda);
      ca_nmda[i] = ca_nmda[i] * ca_decay + ca_entry;

      // Decay NMDA conductance pool
      g_nmda[i] = std::min(g_nmda[i] * decay, max_g);
    }
  }

  // Mean NMDA conductance (diagnostic)
  float MeanG() const {
    if (g_nmda.empty()) return 0.0f;
    float sum = 0.0f;
    for (float g : g_nmda) sum += g;
    return sum / static_cast<float>(g_nmda.size());
  }

  // Mean NMDA calcium (diagnostic, for coupling to plasticity)
  float MeanCa() const {
    if (ca_nmda.empty()) return 0.0f;
    float sum = 0.0f;
    for (float c : ca_nmda) sum += c;
    return sum / static_cast<float>(ca_nmda.size());
  }

  void Clear() {
    std::fill(g_nmda.begin(), g_nmda.end(), 0.0f);
    std::fill(ca_nmda.begin(), ca_nmda.end(), 0.0f);
  }
};

}  // namespace mechabrain

#endif  // FWMC_NMDA_H_
