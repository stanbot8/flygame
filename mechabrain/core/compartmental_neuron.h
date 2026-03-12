#ifndef FWMC_COMPARTMENTAL_NEURON_H_
#define FWMC_COMPARTMENTAL_NEURON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/platform.h"

namespace mechabrain {

// Reduced 3-compartment pyramidal neuron model for cortical simulation.
//
// Implements a biophysically grounded model with soma, apical dendrite,
// and basal dendrite compartments coupled by axial conductance. Each
// compartment carries its own membrane potential and active channels.
//
// The soma generates Na+/K+ action potentials. The apical dendrite
// supports Ca2+ spikes and NMDA-driven dendritic plateau potentials.
// Backpropagating action potentials (bAPs) are modeled as attenuated
// soma spikes propagating into dendrites.
//
// This model fills the LODLevel::kCompartmental (LOD 3) slot in the
// multiscale hierarchy. It runs where single-neuron biophysical detail
// matters (e.g., cortical pyramidal cells during closed-loop bridging).
//
// Key references:
//   Pinsky & Rinzel 1994, J Comput Neurosci 1:39    (2-compartment framework)
//   Gao et al. 2020, J Neurosci 40(44):8513         (minimal L5 pyramidal)
//   Jahr & Stevens 1990, J Neurosci 10(9):3178       (NMDA Mg2+ block)
//   Stuart & Sakmann 1994, Nature 367:69             (bAP attenuation)
//   Hay et al. 2011, PLoS Comput Biol 7:e1002107    (detailed L5b model)
//   Larkum et al. 2009, Science 325:756              (dendritic integration)

// ---- Channel gating helpers ----

// Boltzmann steady-state activation/inactivation.
inline float BoltzmannInf(float v, float v_half, float k) {
  return 1.0f / (1.0f + std::exp(-(v - v_half) / k));
}

// Time constant with bell-shaped voltage dependence.
inline float BellTau(float v, float v_peak, float sigma, float tau_min, float tau_max) {
  float x = (v - v_peak) / sigma;
  return tau_min + (tau_max - tau_min) * std::exp(-x * x);
}

// ---- Parameters ----

// Per-compartment passive properties.
struct CompartmentParams {
  float Cm = 1.0f;          // membrane capacitance (uF/cm2)
  float g_leak = 0.04f;     // leak conductance (mS/cm2)
  float E_leak = -70.0f;    // leak reversal (mV)
  float area_frac = 0.1f;   // fraction of total cell surface area
};

// Active channel conductances (mS/cm2).
struct ChannelParams {
  // Soma: fast Na+ and delayed-rectifier K+ (Pinsky & Rinzel 1994; Gao 2020)
  float g_Na = 25.0f;       // fast sodium
  float g_KDR = 10.0f;      // delayed rectifier potassium

  // Apical dendrite: Ca2+ spike zone (Hay 2011; Larkum 2009)
  float g_CaHVA = 3.0f;     // high-voltage-activated calcium
  float g_KCa = 5.0f;       // calcium-activated potassium (SK/BK)
  float g_Ih = 0.1f;        // hyperpolarization-activated cation (HCN)

  // Basal dendrite: moderate Na+ for local spikes
  float g_Na_basal = 5.0f;  // attenuated sodium in basal dendrites
};

// Reversal potentials (mV).
struct ReversalParams {
  float E_Na = 50.0f;       // sodium
  float E_K = -85.0f;       // potassium
  float E_Ca = 120.0f;      // calcium (Nernst at physiological [Ca]o/[Ca]i)
  float E_h = -45.0f;       // Ih (mixed Na+/K+ cation, Magee 1998)
  float E_NMDA = 0.0f;      // NMDA (non-selective cation)
};

// Inter-compartmental coupling.
// Derived from axial resistance and geometry.
// g_c in mS/cm2 following Pinsky & Rinzel convention.
struct CouplingParams {
  float g_soma_apical = 2.0f;   // soma <-> apical dendrite
  float g_soma_basal = 3.0f;    // soma <-> basal dendrite (shorter, tighter coupling)
};

// NMDA receptor parameters (Jahr & Stevens 1990).
struct NMDAParams {
  float Mg_mM = 1.0f;       // extracellular [Mg2+] in mM
  float tau_rise = 2.0f;    // rise time constant (ms)
  float tau_decay = 100.0f;  // decay time constant (ms)
  float g_max = 1.0f;       // peak NMDA conductance (mS/cm2)
};

// Calcium dynamics parameters.
// Thin submembrane shell model (Destexhe et al. 1993).
struct CalciumParams {
  float tau_Ca = 80.0f;     // calcium removal time constant (ms)
  float Ca_rest = 0.0001f;  // resting [Ca2+]i (mM, = 100 nM)
  // Conversion factor: current to [Ca2+] change.
  // depth = 1 um, F = 96485 C/mol.
  // d[Ca]/dt = -alpha * I_Ca / (2*F*depth) - (Ca - Ca_rest)/tau
  // alpha absorbs depth and F into a single gain.
  float alpha = 0.005f;     // mM per (uA/cm2 * ms)
};

// Backpropagating AP model (Stuart & Sakmann 1994).
struct BAPParams {
  float attenuation_apical = 0.5f;  // bAP amplitude at apical dendrite (fraction of soma)
  float attenuation_basal = 0.7f;   // bAP amplitude at basal dendrite
  float delay_ms = 0.5f;            // propagation delay from soma (ms)
  // bAP injects a brief depolarizing current pulse into dendrites when
  // the soma fires, scaled by the attenuation factor.
  float pulse_amplitude = 40.0f;    // mV above resting (soma spike ~ 100mV from rest)
  float pulse_duration_ms = 0.5f;   // duration of the bAP pulse
};

// Full parameter set for compartmental neurons.
struct CompartmentalParams {
  CompartmentParams soma   = {1.0f, 0.04f, -70.0f, 0.1f};
  CompartmentParams apical = {2.0f, 0.04f, -70.0f, 0.6f}; // 2x Cm for spine correction
  CompartmentParams basal  = {1.5f, 0.04f, -70.0f, 0.3f};

  ChannelParams channels;
  ReversalParams reversal;
  CouplingParams coupling;
  NMDAParams nmda;
  CalciumParams calcium;
  BAPParams bap;

  float v_thresh = -50.0f;  // somatic spike threshold (mV)
  float v_reset = -65.0f;   // somatic reset after spike
  float refractory_ms = 2.0f;
};

// ---- Structure-of-Arrays storage ----

// State for a population of compartmental neurons.
// Flat arrays for cache-friendly iteration and future SIMD/GPU porting.
struct CompartmentalArray {
  size_t n = 0;

  // Membrane potentials (mV)
  std::vector<float> v_soma;
  std::vector<float> v_apical;
  std::vector<float> v_basal;

  // Gating variables
  std::vector<float> m_Na;      // Na+ activation (soma)
  std::vector<float> h_Na;      // Na+ inactivation (soma)
  std::vector<float> n_KDR;     // K+ delayed rectifier (soma)
  std::vector<float> m_CaHVA;   // Ca2+ HVA activation (apical)
  std::vector<float> h_CaHVA;   // Ca2+ HVA inactivation (apical)
  std::vector<float> m_Ih;      // Ih activation (apical)
  std::vector<float> m_Na_b;    // Na+ activation (basal)
  std::vector<float> h_Na_b;    // Na+ inactivation (basal)

  // Calcium concentration in apical dendrite (mM)
  std::vector<float> Ca_i;

  // KCa gating (derived from Ca, not a differential variable)

  // NMDA synaptic gating
  std::vector<float> s_NMDA_apical;  // NMDA gating on apical dendrite
  std::vector<float> s_NMDA_basal;   // NMDA gating on basal dendrite

  // Synaptic input currents
  std::vector<float> i_syn_soma;     // fast synaptic input to soma
  std::vector<float> i_syn_apical;   // synaptic input to apical dendrite
  std::vector<float> i_syn_basal;    // synaptic input to basal dendrite

  // External stimulus currents
  std::vector<float> i_ext_soma;
  std::vector<float> i_ext_apical;
  std::vector<float> i_ext_basal;

  // Spike state
  std::vector<uint8_t> spiked;
  std::vector<float> last_spike_time;

  // bAP state: remaining pulse duration in each dendrite (ms)
  std::vector<float> bap_apical_remaining;
  std::vector<float> bap_basal_remaining;

  void Resize(size_t count) {
    n = count;
    v_soma.assign(count, -65.0f);
    v_apical.assign(count, -65.0f);
    v_basal.assign(count, -65.0f);

    // Na+ gating at rest (~-65mV): m ~ 0.05, h ~ 0.6
    m_Na.assign(count, 0.05f);
    h_Na.assign(count, 0.6f);
    n_KDR.assign(count, 0.3f);
    m_CaHVA.assign(count, 0.0f);
    h_CaHVA.assign(count, 1.0f);
    m_Ih.assign(count, 0.15f);  // partially open at rest
    m_Na_b.assign(count, 0.05f);
    h_Na_b.assign(count, 0.6f);

    Ca_i.assign(count, 0.0001f);  // resting [Ca2+]i = 100 nM

    s_NMDA_apical.assign(count, 0.0f);
    s_NMDA_basal.assign(count, 0.0f);

    i_syn_soma.assign(count, 0.0f);
    i_syn_apical.assign(count, 0.0f);
    i_syn_basal.assign(count, 0.0f);

    i_ext_soma.assign(count, 0.0f);
    i_ext_apical.assign(count, 0.0f);
    i_ext_basal.assign(count, 0.0f);

    spiked.assign(count, 0);
    last_spike_time.assign(count, -1e9f);

    bap_apical_remaining.assign(count, 0.0f);
    bap_basal_remaining.assign(count, 0.0f);
  }

  void ClearSynapticInput() {
    std::fill(i_syn_soma.begin(), i_syn_soma.end(), 0.0f);
    std::fill(i_syn_apical.begin(), i_syn_apical.end(), 0.0f);
    std::fill(i_syn_basal.begin(), i_syn_basal.end(), 0.0f);
  }

  void ClearExternalInput() {
    std::fill(i_ext_soma.begin(), i_ext_soma.end(), 0.0f);
    std::fill(i_ext_apical.begin(), i_ext_apical.end(), 0.0f);
    std::fill(i_ext_basal.begin(), i_ext_basal.end(), 0.0f);
  }

  int CountSpikes() const {
    int count = 0;
    for (size_t i = 0; i < n; ++i) count += spiked[i];
    return count;
  }
};

// ---- NMDA Mg2+ block ----

// Jahr & Stevens 1990: voltage-dependent magnesium block of NMDA receptors.
// Returns the unblocked fraction B(V) in [0,1].
// At -70mV: ~2% unblocked (strong block at rest).
// At -20mV: ~55% unblocked (significant current during depolarization).
// This is the key nonlinearity enabling dendritic coincidence detection.
inline float NMDAMgBlock(float v_mV, float Mg_mM) {
  return 1.0f / (1.0f + (Mg_mM / 3.57f) * std::exp(-0.062f * v_mV));
}

// ---- Channel kinetics ----

// Fast Na+ activation (Mainen & Sejnowski 1996 convention).
inline float mNaInf(float v) { return BoltzmannInf(v, -30.0f, 7.0f); }
inline float mNaTau(float /*v*/) { return 0.1f; }  // fast, ~0.1ms

// Fast Na+ inactivation.
inline float hNaInf(float v) { return BoltzmannInf(v, -53.0f, -7.0f); }
inline float hNaTau(float v) { return BellTau(v, -50.0f, 15.0f, 0.5f, 10.0f); }

// Delayed rectifier K+.
inline float nKDRInf(float v) { return BoltzmannInf(v, -30.0f, 10.0f); }
inline float nKDRTau(float v) { return BellTau(v, -30.0f, 20.0f, 1.0f, 10.0f); }

// High-voltage-activated Ca2+ (Hay et al. 2011).
inline float mCaHVAInf(float v) { return BoltzmannInf(v, -27.0f, 5.0f); }
inline float mCaHVATau(float /*v*/) { return 1.0f; }
inline float hCaHVAInf(float v) { return BoltzmannInf(v, -32.0f, -7.0f); }
inline float hCaHVATau(float /*v*/) { return 30.0f; }

// Ih / HCN channel (Magee 1998; Hay 2011).
// Opens on hyperpolarization (inverted Boltzmann).
inline float mIhInf(float v) { return BoltzmannInf(v, -82.0f, -9.0f); }
inline float mIhTau(float v) { return BellTau(v, -75.0f, 15.0f, 20.0f, 200.0f); }

// Ca-activated K+ (SK channel): Hill function of [Ca2+]i.
// Half-activation at ~0.5 uM = 0.0005 mM (Kohler et al. 1996).
inline float gKCa(float Ca_i_mM, float g_max) {
  float Ca_uM = Ca_i_mM * 1000.0f;
  float half = 0.5f;  // uM
  float hill = Ca_uM * Ca_uM / (Ca_uM * Ca_uM + half * half);  // Hill n=2
  return g_max * hill;
}

// Exponential Euler update for a gating variable toward its steady-state.
// gate += (inf - gate) * (1 - exp(-dt/tau))
inline float GateUpdate(float gate, float inf, float tau, float dt_ms) {
  return gate + (inf - gate) * (1.0f - std::exp(-dt_ms / tau));
}

// ---- Integration ----

// Step all compartmental neurons forward by dt_ms.
// Uses forward Euler with gating variable updates.
// Each neuron is independent (embarrassingly parallel).
inline void CompartmentalStep(CompartmentalArray& neurons, float dt_ms,
                               float sim_time_ms,
                               const CompartmentalParams& p) {
  const int nn = static_cast<int>(neurons.n);

  // Pre-compute constant-tau decay factors outside the loop.
  // Saves 7 exp() calls per neuron per timestep.
  // Gates with voltage-dependent tau (hNa, nKDR, mIh, hNa_b) still
  // compute exp() per neuron inside the loop.
  const float alpha_mNa   = 1.0f - std::exp(-dt_ms / 0.1f);   // mNaTau = 0.1 ms
  const float alpha_mCaHVA = 1.0f - std::exp(-dt_ms / 1.0f);  // mCaHVATau = 1.0 ms
  const float alpha_hCaHVA = 1.0f - std::exp(-dt_ms / 30.0f); // hCaHVATau = 30.0 ms
  const float alpha_mNa_b = alpha_mNa;                         // same as soma Na+
  const float nmda_decay   = std::exp(-dt_ms / p.nmda.tau_decay);

  // Pre-compute reciprocals for division in the inner loop.
  const float inv_Cm_soma   = dt_ms / p.soma.Cm;
  const float inv_Cm_apical = dt_ms / p.apical.Cm;
  const float inv_Cm_basal  = dt_ms / p.basal.Cm;

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(nn > 1000)
  #endif
  for (int idx = 0; idx < nn; ++idx) {
    const size_t i = static_cast<size_t>(idx);

    float vs = neurons.v_soma[i];
    float va = neurons.v_apical[i];
    float vb = neurons.v_basal[i];

    // ---- Somatic currents ----

    float m = neurons.m_Na[i];
    float h = neurons.h_Na[i];
    float nk = neurons.n_KDR[i];

    // Fast gate updates: constant-tau gates use precomputed alpha.
    m += (mNaInf(vs) - m) * alpha_mNa;
    h = GateUpdate(h, hNaInf(vs), hNaTau(vs), dt_ms);     // voltage-dependent tau
    nk = GateUpdate(nk, nKDRInf(vs), nKDRTau(vs), dt_ms);  // voltage-dependent tau

    float I_Na = p.channels.g_Na * m * m * m * h * (vs - p.reversal.E_Na);
    float I_KDR = p.channels.g_KDR * nk * nk * nk * nk * (vs - p.reversal.E_K);
    float I_leak_s = p.soma.g_leak * (vs - p.soma.E_leak);

    float I_coup_sa = p.coupling.g_soma_apical * (vs - va);
    float I_coup_sb = p.coupling.g_soma_basal * (vs - vb);

    float I_total_soma = -I_Na - I_KDR - I_leak_s - I_coup_sa - I_coup_sb
                         + neurons.i_syn_soma[i] + neurons.i_ext_soma[i];

    // ---- Apical dendritic currents ----

    float mc = neurons.m_CaHVA[i];
    float hc = neurons.h_CaHVA[i];
    float mih = neurons.m_Ih[i];

    mc += (mCaHVAInf(va) - mc) * alpha_mCaHVA;   // constant tau
    hc += (hCaHVAInf(va) - hc) * alpha_hCaHVA;   // constant tau
    mih = GateUpdate(mih, mIhInf(va), mIhTau(va), dt_ms);  // voltage-dependent tau

    float I_CaHVA = p.channels.g_CaHVA * mc * mc * hc * (va - p.reversal.E_Ca);

    float ca = neurons.Ca_i[i];
    float gkca = gKCa(ca, p.channels.g_KCa);
    float I_KCa = gkca * (va - p.reversal.E_K);

    float I_Ih = p.channels.g_Ih * mih * (va - p.reversal.E_h);

    float s_nmda_a = neurons.s_NMDA_apical[i];
    float B_a = NMDAMgBlock(va, p.nmda.Mg_mM);
    float I_NMDA_a = p.nmda.g_max * s_nmda_a * B_a * (va - p.reversal.E_NMDA);

    float I_leak_a = p.apical.g_leak * (va - p.apical.E_leak);
    float I_coup_as = p.coupling.g_soma_apical * (va - vs);

    float I_bap_a = 0.0f;
    float bap_a_rem = neurons.bap_apical_remaining[i];
    if (bap_a_rem > 0.0f) {
      I_bap_a = p.bap.attenuation_apical * p.bap.pulse_amplitude * 0.5f;
      bap_a_rem -= dt_ms;
      if (bap_a_rem < 0.0f) bap_a_rem = 0.0f;
    }

    float I_total_apical = -I_CaHVA - I_KCa - I_Ih - I_NMDA_a - I_leak_a
                           - I_coup_as + I_bap_a
                           + neurons.i_syn_apical[i] + neurons.i_ext_apical[i];

    // ---- Basal dendritic currents ----

    float mb = neurons.m_Na_b[i];
    float hb = neurons.h_Na_b[i];

    mb += (mNaInf(vb) - mb) * alpha_mNa_b;  // constant tau
    hb = GateUpdate(hb, hNaInf(vb), hNaTau(vb), dt_ms);  // voltage-dependent tau

    float I_Na_b = p.channels.g_Na_basal * mb * mb * mb * hb * (vb - p.reversal.E_Na);
    float I_leak_b = p.basal.g_leak * (vb - p.basal.E_leak);
    float I_coup_bs = p.coupling.g_soma_basal * (vb - vs);

    float s_nmda_b = neurons.s_NMDA_basal[i];
    float B_b = NMDAMgBlock(vb, p.nmda.Mg_mM);
    float I_NMDA_b = p.nmda.g_max * s_nmda_b * B_b * (vb - p.reversal.E_NMDA);

    float I_bap_b = 0.0f;
    float bap_b_rem = neurons.bap_basal_remaining[i];
    if (bap_b_rem > 0.0f) {
      I_bap_b = p.bap.attenuation_basal * p.bap.pulse_amplitude * 0.5f;
      bap_b_rem -= dt_ms;
      if (bap_b_rem < 0.0f) bap_b_rem = 0.0f;
    }

    float I_total_basal = -I_Na_b - I_NMDA_b - I_leak_b - I_coup_bs + I_bap_b
                          + neurons.i_syn_basal[i] + neurons.i_ext_basal[i];

    // ---- Membrane potential update (forward Euler) ----

    vs += I_total_soma * inv_Cm_soma;
    va += I_total_apical * inv_Cm_apical;
    vb += I_total_basal * inv_Cm_basal;

    // ---- Calcium dynamics (apical dendrite) ----
    float dCa = -p.calcium.alpha * I_CaHVA
                - (ca - p.calcium.Ca_rest) / p.calcium.tau_Ca;
    ca += dt_ms * dCa;
    ca = std::max(p.calcium.Ca_rest, ca);

    // ---- NMDA gating decay (precomputed factor) ----
    s_nmda_a *= nmda_decay;
    s_nmda_b *= nmda_decay;

    // ---- Spike detection (somatic) ----

    // Clamp divergent voltages
    if (!std::isfinite(vs) || vs > 60.0f) vs = p.v_reset;
    if (!std::isfinite(va) || std::abs(va) > 200.0f) va = p.apical.E_leak;
    if (!std::isfinite(vb) || std::abs(vb) > 200.0f) vb = p.basal.E_leak;
    if (!std::isfinite(ca)) ca = p.calcium.Ca_rest;

    bool in_refractory = (sim_time_ms - neurons.last_spike_time[i]) < p.refractory_ms;
    uint8_t fired = (!in_refractory && vs >= p.v_thresh) ? 1 : 0;

    if (fired) {
      vs = p.v_reset;
      neurons.last_spike_time[i] = sim_time_ms;

      // Initiate bAP: brief depolarizing pulse propagates into dendrites
      bap_a_rem = p.bap.pulse_duration_ms;
      bap_b_rem = p.bap.pulse_duration_ms;
    }

    // ---- Write back ----

    neurons.v_soma[i] = vs;
    neurons.v_apical[i] = va;
    neurons.v_basal[i] = vb;
    neurons.m_Na[i] = m;
    neurons.h_Na[i] = h;
    neurons.n_KDR[i] = nk;
    neurons.m_CaHVA[i] = mc;
    neurons.h_CaHVA[i] = hc;
    neurons.m_Ih[i] = mih;
    neurons.m_Na_b[i] = mb;
    neurons.h_Na_b[i] = hb;
    neurons.Ca_i[i] = ca;
    neurons.s_NMDA_apical[i] = s_nmda_a;
    neurons.s_NMDA_basal[i] = s_nmda_b;
    neurons.spiked[i] = fired;
    neurons.bap_apical_remaining[i] = bap_a_rem;
    neurons.bap_basal_remaining[i] = bap_b_rem;
  }
}

// Activate NMDA synapses on the apical dendrite of target neurons.
// Call this when a presynaptic spike arrives at an NMDA synapse.
// The gating variable steps toward 1 with an exponential rise.
inline void ActivateNMDA_Apical(CompartmentalArray& neurons,
                                 const uint32_t* targets, size_t n_targets,
                                 float dt_ms, float tau_rise) {
  float rise = 1.0f - std::exp(-dt_ms / tau_rise);
  for (size_t j = 0; j < n_targets; ++j) {
    uint32_t t = targets[j];
    neurons.s_NMDA_apical[t] += rise * (1.0f - neurons.s_NMDA_apical[t]);
    neurons.s_NMDA_apical[t] = std::min(neurons.s_NMDA_apical[t], 1.0f);
  }
}

// Activate NMDA synapses on the basal dendrite.
inline void ActivateNMDA_Basal(CompartmentalArray& neurons,
                                const uint32_t* targets, size_t n_targets,
                                float dt_ms, float tau_rise) {
  float rise = 1.0f - std::exp(-dt_ms / tau_rise);
  for (size_t j = 0; j < n_targets; ++j) {
    uint32_t t = targets[j];
    neurons.s_NMDA_basal[t] += rise * (1.0f - neurons.s_NMDA_basal[t]);
    neurons.s_NMDA_basal[t] = std::min(neurons.s_NMDA_basal[t], 1.0f);
  }
}

// Read the somatic voltage as a simplified "activity" measure [0,1].
// Useful for coupling compartmental neurons back to the LOD system.
inline float CompartmentalActivity(const CompartmentalArray& neurons,
                                    size_t i,
                                    const CompartmentalParams& p) {
  float v = neurons.v_soma[i];
  float range = p.v_thresh - p.v_reset;
  return std::clamp((v - p.v_reset) / range, 0.0f, 1.0f);
}

// Default parameters for mammalian L5 pyramidal neurons.
// These produce realistic behavior: somatic APs, dendritic Ca2+ spikes
// when apical dendrites are sufficiently depolarized, and bAP-triggered
// calcium transients in dendrites.
inline CompartmentalParams DefaultPyramidalParams() {
  CompartmentalParams p;
  // Soma: small, high Na+ density for reliable spike initiation
  p.soma = {1.0f, 0.04f, -70.0f, 0.1f};
  // Apical: large, with spine correction, Ca2+ hot zone
  p.apical = {2.0f, 0.04f, -70.0f, 0.6f};
  // Basal: moderate, tighter coupling to soma
  p.basal = {1.5f, 0.04f, -70.0f, 0.3f};

  p.channels.g_Na = 25.0f;
  p.channels.g_KDR = 10.0f;
  p.channels.g_CaHVA = 3.0f;
  p.channels.g_KCa = 5.0f;
  p.channels.g_Ih = 0.1f;
  p.channels.g_Na_basal = 5.0f;

  p.coupling.g_soma_apical = 2.0f;
  p.coupling.g_soma_basal = 3.0f;

  p.v_thresh = -50.0f;
  p.v_reset = -65.0f;
  p.refractory_ms = 2.0f;

  return p;
}

// Default parameters for L2/3 pyramidal neurons (thinner apical, less Ca).
inline CompartmentalParams DefaultL23PyramidalParams() {
  CompartmentalParams p = DefaultPyramidalParams();
  p.channels.g_CaHVA = 1.0f;    // weaker Ca2+ spike zone
  p.channels.g_KCa = 2.0f;
  p.channels.g_Ih = 0.05f;      // less Ih
  p.coupling.g_soma_apical = 2.5f;  // tighter coupling (shorter dendrite)
  return p;
}

}  // namespace mechabrain

#endif  // FWMC_COMPARTMENTAL_NEURON_H_
