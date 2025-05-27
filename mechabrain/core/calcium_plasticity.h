#ifndef FWMC_CALCIUM_PLASTICITY_H_
#define FWMC_CALCIUM_PLASTICITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Calcium-dependent plasticity (CaDP): mechanistic learning rule that
// derives STDP from NMDA receptor calcium dynamics.
//
// Instead of hardcoding timing windows (classic STDP), this module reads
// the per-neuron NMDA-mediated calcium concentration (ca_nmda from
// NMDAReceptor) and applies weight changes based on the calcium level
// at each synapse's postsynaptic site when the presynaptic neuron fires.
//
// The Omega function determines the sign and magnitude of plasticity:
//
//   ca < theta_d:              no change (subthreshold calcium)
//   theta_d < ca < theta_p:   LTD (moderate calcium activates calcineurin)
//   ca > theta_p:             LTP (high calcium activates CaMKII)
//
// This naturally reproduces STDP timing curves because:
//   - Pre before post (~10ms): post depolarization removes Mg2+ block,
//     NMDA channels open, high Ca2+ entry -> LTP
//   - Post before pre: post depolarization is fading when pre arrives,
//     partial Mg2+ removal, moderate Ca2+ -> LTD
//   - Large dt: no temporal coincidence, Ca2+ stays below theta_d -> no change
//
// Additionally implements the BCM sliding threshold (Bienenstock, Cooper,
// Munro 1982): theta_p shifts as a function of postsynaptic activity
// history, preventing runaway potentiation. High activity raises the
// LTP threshold; low activity lowers it.
//
// References:
//   Shouval, Bear, Cooper (2002) PNAS 99:10831  (unified Ca model)
//   Graupner & Brunel (2012) PNAS 109:3991      (bistable Ca synapse)
//   Bienenstock, Cooper, Munro (1982) J Neurosci (BCM theory)
//   Rubin, Bhatt et al. (2005) J Neurophysiol    (simplified Ca rule)
//   Clopath et al. (2010) Nat Neurosci 13:344    (voltage-based rule)
//
// Drosophila: NMDA receptors (dNR1/dNR2) are required for long-term
// memory formation in the mushroom body (Xia et al. 2005; Wu et al. 2007).
// Ca2+-dependent plasticity via CaMKII (ala) is conserved and essential
// for courtship conditioning and olfactory LTM (Griffith et al. 1993).

struct CalciumPlasticityParams {
  // Omega function thresholds (dimensionless, in ca_nmda units).
  // Calibrated against Graupner-Brunel 2012 Table 1:
  //   theta_d = 1.0, theta_p = 1.3 (normalized calcium units)
  // Mapped to our ca_nmda scale where typical NMDA calcium peaks
  // are 0.01-0.5 a.u. depending on coincidence strength.
  float theta_d = 0.03f;    // depression threshold (calcineurin activation)
  float theta_p = 0.08f;    // potentiation threshold (CaMKII activation)

  // Transition steepness for smooth sigmoid Omega function.
  // Larger values = sharper transitions (more switch-like).
  float beta = 50.0f;

  // Learning rates.
  // Graupner-Brunel: gamma_d = 200, gamma_p = 321.808 (per second).
  // Scaled to per-ms and our weight units.
  float alpha_ltp = 0.008f;  // LTP amplitude (CaMKII pathway)
  float alpha_ltd = 0.005f;  // LTD amplitude (calcineurin pathway)

  // Weight bounds (shared with STDP for consistency).
  float w_min = 0.0f;
  float w_max = 10.0f;

  // BCM sliding threshold parameters.
  // theta_p_effective = theta_p * (mean_rate / target_rate)^2
  // When postsynaptic neuron is very active, theta_p increases,
  // making LTP harder to induce (prevents runaway excitation).
  bool bcm_sliding = true;
  float bcm_target_rate_hz = 5.0f;   // target firing rate for BCM
  float bcm_tau_ms = 5000.0f;        // time constant for rate averaging (5s)
  float bcm_min_factor = 0.5f;       // minimum theta_p scaling
  float bcm_max_factor = 3.0f;       // maximum theta_p scaling

  // Only apply to excitatory synapses (ACh in Drosophila, Glut in mammals).
  // NMDA receptors are not present at inhibitory synapses.
  bool excitatory_only = true;

  // Dopamine gating: when enabled, LTP magnitude is scaled by local
  // dopamine concentration, linking reward signals to associative learning.
  bool dopamine_modulated = false;
  float da_ltp_scale = 3.0f;   // DA scales LTP by (1 + da_ltp_scale * [DA])
};

// Smooth sigmoid for Omega function transitions.
// sigma(x, beta) = 1 / (1 + exp(-beta * x))
inline float CaDPSigmoid(float x, float beta) {
  return 1.0f / (1.0f + std::exp(-beta * x));
}

// Omega function: maps calcium concentration to weight change rate.
//
// Returns a value in [-alpha_ltd, +alpha_ltp]:
//   ca < theta_d:            ~0 (no change)
//   theta_d < ca < theta_p:  negative (LTD, calcineurin dominant)
//   ca > theta_p:            positive (LTP, CaMKII dominant)
//
// Uses smooth sigmoids for differentiability and numerical stability.
inline float Omega(float ca, float theta_d, float theta_p,
                   float alpha_ltp, float alpha_ltd, float beta) {
  // Depression gate: active above theta_d
  float d_gate = CaDPSigmoid(ca - theta_d, beta);
  // Potentiation gate: active above theta_p
  float p_gate = CaDPSigmoid(ca - theta_p, beta);

  // Below theta_d: both gates ~0, no change.
  // Between theta_d and theta_p: d_gate ~1, p_gate ~0 -> depression.
  // Above theta_p: both gates ~1 -> net potentiation.
  return alpha_ltp * p_gate - alpha_ltd * d_gate * (1.0f - p_gate);
}

// Per-neuron BCM state for sliding threshold computation.
struct BCMState {
  std::vector<float> rate_avg;   // exponential moving average of firing rate
  bool initialized = false;

  void Init(size_t n) {
    rate_avg.assign(n, 0.0f);
    initialized = true;
  }

  // Update rate averages from current spike state.
  void Update(const NeuronArray& neurons, float dt_ms, float tau_ms) {
    if (!initialized) return;
    float alpha = dt_ms / tau_ms;
    float decay = 1.0f - alpha;
    // Instantaneous rate estimate: spiked/dt converted to Hz
    float rate_scale = 1000.0f / std::max(0.1f, dt_ms);
    for (size_t i = 0; i < neurons.n; ++i) {
      float inst_rate = static_cast<float>(neurons.spiked[i]) * rate_scale;
      rate_avg[i] = rate_avg[i] * decay + inst_rate * alpha;
    }
  }

  // Compute BCM scaling factor for theta_p at a given neuron.
  // factor = clamp((rate_avg / target)^2, min, max)
  float ThresholdFactor(size_t neuron_idx, float target_rate,
                        float min_factor, float max_factor) const {
    if (!initialized || neuron_idx >= rate_avg.size()) return 1.0f;
    float ratio = rate_avg[neuron_idx] / std::max(0.1f, target_rate);
    float factor = ratio * ratio;
    return std::clamp(factor, min_factor, max_factor);
  }
};

// Main calcium-dependent plasticity update.
//
// Walk the CSR synapse graph for each neuron that spiked this timestep.
// For each of its postsynaptic targets (excitatory synapses only), read
// the target's ca_nmda and apply the Omega function to determine the
// weight change.
//
// This is efficient because:
//   1. Only processes spiked presynaptic neurons (typically <5% per step)
//   2. Uses per-neuron ca_nmda (already computed by NMDAReceptor), no
//      per-synapse calcium storage needed
//   3. CSR data is hot in cache from the preceding PropagateSpikes pass
//   4. OpenMP parallelization for large networks
//
// Call after NMDAReceptor::Step() and before IzhikevichStep() each timestep.
inline void CalciumPlasticityUpdate(
    SynapseTable& synapses,
    const NeuronArray& neurons,
    const float* ca_nmda,           // per-neuron NMDA calcium (from NMDAReceptor)
    const CalciumPlasticityParams& p,
    const BCMState* bcm = nullptr)  // optional BCM state for sliding threshold
{
  if (!ca_nmda) return;

  const int n = static_cast<int>(synapses.n_neurons);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
  #endif
  for (int pre = 0; pre < n; ++pre) {
    if (!neurons.spiked[pre]) continue;

    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];

    for (uint32_t s = start; s < end; ++s) {
      // Only excitatory synapses have NMDA receptors
      if (p.excitatory_only && synapses.nt_type[s] != kACh) continue;

      uint32_t post_idx = synapses.post[s];
      float ca = ca_nmda[post_idx];

      // Compute effective potentiation threshold (with BCM sliding)
      float theta_p_eff = p.theta_p;
      if (p.bcm_sliding && bcm && bcm->initialized) {
        float factor = bcm->ThresholdFactor(
            post_idx, p.bcm_target_rate_hz,
            p.bcm_min_factor, p.bcm_max_factor);
        theta_p_eff = p.theta_p * factor;
      }

      // Compute weight change from Omega function
      float dw = Omega(ca, p.theta_d, theta_p_eff,
                        p.alpha_ltp, p.alpha_ltd, p.beta);

      if (dw == 0.0f) continue;

      // Optional dopamine modulation of LTP
      if (p.dopamine_modulated && dw > 0.0f) {
        float da = neurons.dopamine[post_idx];
        dw *= (1.0f + p.da_ltp_scale * da);
      }

      synapses.weight[s] = std::clamp(
          synapses.weight[s] + dw, p.w_min, p.w_max);
    }
  }
}

// Convenience struct that bundles params + BCM state for easy wiring.
struct CalciumPlasticity {
  CalciumPlasticityParams params;
  BCMState bcm;
  bool initialized = false;

  void Init(size_t n_neurons) {
    bcm.Init(n_neurons);
    initialized = true;
  }

  // Call each timestep to update BCM rate averages.
  void UpdateBCM(const NeuronArray& neurons, float dt_ms) {
    if (!initialized) return;
    bcm.Update(neurons, dt_ms, params.bcm_tau_ms);
  }

  // Call each timestep after NMDAReceptor::Step() to apply plasticity.
  void Apply(SynapseTable& synapses, const NeuronArray& neurons,
             const float* ca_nmda) {
    if (!initialized) return;
    CalciumPlasticityUpdate(synapses, neurons, ca_nmda, params,
                            params.bcm_sliding ? &bcm : nullptr);
  }

  // Mean BCM rate (diagnostic)
  float MeanBCMRate() const {
    if (!bcm.initialized || bcm.rate_avg.empty()) return 0.0f;
    float sum = 0.0f;
    for (float r : bcm.rate_avg) sum += r;
    return sum / static_cast<float>(bcm.rate_avg.size());
  }
};

}  // namespace mechabrain

#endif  // FWMC_CALCIUM_PLASTICITY_H_
