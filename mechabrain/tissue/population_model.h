#ifndef FWMC_POPULATION_MODEL_H_
#define FWMC_POPULATION_MODEL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "core/platform.h"

namespace mechabrain {

// Montbrio-Pazo-Roxin (MPR) exact mean-field model for QIF neurons.
//
// Exact macroscopic reduction of quadratic integrate-and-fire (QIF)
// neuron populations (Montbrio, Pazo, Roxin, Phys Rev X 2015).
// Unlike Wilson-Cowan (LOD 0), this is not an approximation: it is
// the exact thermodynamic limit of the spiking model used at LOD 2.
//
// Two coupled E/I populations, each described by:
//   tau * dr/dt = Delta/(pi*tau) + 2*r*v
//   tau * dv/dt = v^2 + eta + I_syn - a - (pi*r*tau)^2
//   tau_a * da/dt = -a + alpha*tau_a*r
//
// r: mean firing rate (Hz), v: mean membrane potential, a: adaptation.
//
// This provides LOD 1: more informative than the Wilson-Cowan field
// (captures adaptation, refractory dynamics, spike statistics) at a
// fraction of the cost of individual neurons (6 ODEs per region vs.
// thousands of Izhikevich neurons).
//
// References:
//   Montbrio E, Pazo D, Roxin A (2015) Phys Rev X 5:021028
//   Byrne A, et al. (2020) J Math Neurosci 10:9 (with adaptation)
//   Benda J, Herz AVM (2003) Neural Comput 15:2523 (SFA)

struct PopulationParams {
  // Excitatory population
  float tau_e = 10.0f;      // membrane time constant (ms)
  float Delta_e = 2.0f;     // Lorentzian spread of excitability
  float eta_e = -5.0f;      // mean excitability (below threshold = excitable)

  // Inhibitory population
  float tau_i = 10.0f;
  float Delta_i = 2.0f;
  float eta_i = -5.0f;

  // Coupling strengths (dimensionless, scaled by tau in the ODE)
  float J_ee = 15.0f;       // E -> E recurrent excitation
  float J_ei = 15.0f;       // I -> E inhibition of excitatory
  float J_ie = 12.0f;       // E -> I excitation of inhibitory
  float J_ii = 3.0f;        // I -> I recurrent inhibition

  // Spike-frequency adaptation
  float alpha_e = 10.0f;    // adaptation strength, excitatory
  float alpha_i = 0.0f;     // PV interneurons: minimal adaptation
  float tau_a = 200.0f;     // adaptation time constant (ms)

  // External drive (baseline current)
  float h_e = 0.0f;
  float h_i = 0.0f;

  // Maximum firing rate for normalization (Hz)
  float max_rate_hz = 100.0f;

  // Drosophila defaults: faster dynamics, less adaptation
  static PopulationParams Drosophila() {
    PopulationParams p;
    p.tau_e = 10.0f;
    p.tau_i = 10.0f;
    p.Delta_e = 2.0f;
    p.Delta_i = 2.0f;
    p.eta_e = -5.0f;
    p.eta_i = -5.0f;
    p.J_ee = 12.0f;
    p.J_ei = 4.0f;
    p.J_ie = 13.0f;
    p.J_ii = 2.0f;
    p.alpha_e = 5.0f;
    p.alpha_i = 0.0f;
    p.tau_a = 150.0f;
    return p;
  }

  // Human cortical defaults: slower E cells, stronger adaptation
  static PopulationParams HumanCortical() {
    PopulationParams p;
    p.tau_e = 20.0f;     // pyramidal cells: longer time constant
    p.tau_i = 10.0f;     // PV interneurons: faster
    p.Delta_e = 1.5f;
    p.Delta_i = 2.0f;
    p.eta_e = -4.0f;
    p.eta_i = -4.0f;
    p.J_ee = 15.0f;
    p.J_ei = 15.0f;
    p.J_ie = 12.0f;
    p.J_ii = 3.0f;
    p.alpha_e = 15.0f;   // stronger adaptation in pyramidal cells
    p.alpha_i = 0.0f;
    p.tau_a = 300.0f;    // slower adaptation
    return p;
  }
};

struct PopulationState {
  // Excitatory population
  float r_e = 0.0f;    // mean firing rate (Hz)
  float v_e = -5.0f;   // mean membrane potential (dimensionless)
  float a_e = 0.0f;    // adaptation current

  // Inhibitory population
  float r_i = 0.0f;
  float v_i = -5.0f;
  float a_i = 0.0f;

  // External input (set each timestep, cleared after step)
  float I_ext_e = 0.0f;
  float I_ext_i = 0.0f;

  // Probabilistic spike output (for cross-region coupling)
  // Expected spike count this step for E and I populations,
  // given a region of n_e and n_i neurons.
  float expected_spikes_e = 0.0f;
  float expected_spikes_i = 0.0f;
};

// Compute the RHS of the MPR equations for one population.
// Returns {dr_dt, dv_dt, da_dt}.
struct MPRDerivs {
  float dr, dv, da;
};

inline MPRDerivs MPR_RHS(float r, float v, float a,
                          float tau, float Delta, float eta,
                          float I_total, float alpha, float tau_a) {
  MPRDerivs d;
  float pi_r_tau = kPi * r * tau;

  // dr/dt = [Delta/(pi*tau) + 2*r*v] / tau
  d.dr = (Delta / (kPi * tau) + 2.0f * r * v) / tau;

  // dv/dt = [v^2 + eta + I_total - a - (pi*r*tau)^2] / tau
  d.dv = (v * v + eta + I_total - a - pi_r_tau * pi_r_tau) / tau;

  // da/dt = [-a + alpha*tau_a*r] / tau_a
  d.da = (-a + alpha * tau_a * r) / tau_a;

  return d;
}

// Step the coupled E/I population model forward by dt_ms.
// Uses Heun's method (RK2) for stability with stiff dynamics.
inline void PopulationStep(PopulationState& s, float dt_ms,
                           const PopulationParams& p) {
  // Synaptic input to each population
  float I_e = p.J_ee * s.r_e * p.tau_e
            - p.J_ei * s.r_i * p.tau_e
            + p.h_e + s.I_ext_e;

  float I_i = p.J_ie * s.r_e * p.tau_i
            - p.J_ii * s.r_i * p.tau_i
            + p.h_i + s.I_ext_i;

  // Stage 1: evaluate derivatives at current state
  auto de1 = MPR_RHS(s.r_e, s.v_e, s.a_e,
                      p.tau_e, p.Delta_e, p.eta_e,
                      I_e, p.alpha_e, p.tau_a);
  auto di1 = MPR_RHS(s.r_i, s.v_i, s.a_i,
                      p.tau_i, p.Delta_i, p.eta_i,
                      I_i, p.alpha_i, p.tau_a);

  // Euler predict
  float r_e2 = s.r_e + dt_ms * de1.dr;
  float v_e2 = s.v_e + dt_ms * de1.dv;
  float a_e2 = s.a_e + dt_ms * de1.da;
  float r_i2 = s.r_i + dt_ms * di1.dr;
  float v_i2 = s.v_i + dt_ms * di1.dv;
  float a_i2 = s.a_i + dt_ms * di1.da;

  // Clamp firing rates to non-negative
  r_e2 = std::max(0.0f, r_e2);
  r_i2 = std::max(0.0f, r_i2);

  // Recompute synaptic input at predicted state
  float I_e2 = p.J_ee * r_e2 * p.tau_e
             - p.J_ei * r_i2 * p.tau_e
             + p.h_e + s.I_ext_e;

  float I_i2 = p.J_ie * r_e2 * p.tau_i
             - p.J_ii * r_i2 * p.tau_i
             + p.h_i + s.I_ext_i;

  // Stage 2: evaluate derivatives at predicted state
  auto de2 = MPR_RHS(r_e2, v_e2, a_e2,
                      p.tau_e, p.Delta_e, p.eta_e,
                      I_e2, p.alpha_e, p.tau_a);
  auto di2 = MPR_RHS(r_i2, v_i2, a_i2,
                      p.tau_i, p.Delta_i, p.eta_i,
                      I_i2, p.alpha_i, p.tau_a);

  // Heun correction: average of two slopes
  s.r_e += 0.5f * dt_ms * (de1.dr + de2.dr);
  s.v_e += 0.5f * dt_ms * (de1.dv + de2.dv);
  s.a_e += 0.5f * dt_ms * (de1.da + de2.da);
  s.r_i += 0.5f * dt_ms * (di1.dr + di2.dr);
  s.v_i += 0.5f * dt_ms * (di1.dv + di2.dv);
  s.a_i += 0.5f * dt_ms * (di1.da + di2.da);

  // Clamp firing rates (must be non-negative)
  s.r_e = std::max(0.0f, s.r_e);
  s.r_i = std::max(0.0f, s.r_i);

  // Clamp to maximum rate for numerical safety
  s.r_e = std::min(s.r_e, p.max_rate_hz);
  s.r_i = std::min(s.r_i, p.max_rate_hz);

  // Clamp adaptation (non-negative)
  s.a_e = std::max(0.0f, s.a_e);
  s.a_i = std::max(0.0f, s.a_i);

  // Guard against NaN/Inf from stiff dynamics
  if (!std::isfinite(s.r_e)) s.r_e = 0.0f;
  if (!std::isfinite(s.v_e)) s.v_e = p.eta_e;
  if (!std::isfinite(s.a_e)) s.a_e = 0.0f;
  if (!std::isfinite(s.r_i)) s.r_i = 0.0f;
  if (!std::isfinite(s.v_i)) s.v_i = p.eta_i;
  if (!std::isfinite(s.a_i)) s.a_i = 0.0f;

  // Compute expected spike counts for this timestep.
  // Used for probabilistic coupling with LOD 2 regions.
  // E(spikes) = rate * dt * n_neurons (set by caller based on region size).
  float dt_s = dt_ms / 1000.0f;
  s.expected_spikes_e = s.r_e * dt_s;  // per-neuron spike probability
  s.expected_spikes_i = s.r_i * dt_s;

  // Clear external input for next step
  s.I_ext_e = 0.0f;
  s.I_ext_i = 0.0f;
}

// Initialize population state from Wilson-Cowan field activity.
// E_field and I_field are in [0, 1].
inline PopulationState PopulationFromField(float E_field, float I_field,
                                           const PopulationParams& p) {
  PopulationState s;

  // Map field activity [0,1] to firing rate [0, max_rate_hz]
  s.r_e = E_field * p.max_rate_hz;
  s.r_i = I_field * p.max_rate_hz;

  // Derive mean voltage from steady-state MPR relation:
  // At steady state, dr/dt = 0 => v = -Delta / (2*pi*tau*r) for r > 0
  float min_rate = 0.1f;  // prevent division by zero
  if (s.r_e > min_rate) {
    s.v_e = -p.Delta_e / (2.0f * kPi * p.tau_e * s.r_e);
  } else {
    s.v_e = p.eta_e;  // subthreshold: v near excitability parameter
  }
  if (s.r_i > min_rate) {
    s.v_i = -p.Delta_i / (2.0f * kPi * p.tau_i * s.r_i);
  } else {
    s.v_i = p.eta_i;
  }

  // Steady-state adaptation: da/dt = 0 => a = alpha * tau_a * r
  s.a_e = p.alpha_e * p.tau_a * s.r_e;
  s.a_i = p.alpha_i * p.tau_a * s.r_i;

  return s;
}

// Collapse population state back to Wilson-Cowan field values.
// Returns {E_field, I_field} in [0, 1].
struct FieldValues {
  float E;
  float I;
};

inline FieldValues FieldFromPopulation(const PopulationState& s,
                                       const PopulationParams& p) {
  FieldValues f;
  f.E = std::clamp(s.r_e / p.max_rate_hz, 0.0f, 1.0f);
  f.I = std::clamp(s.r_i / p.max_rate_hz, 0.0f, 1.0f);
  return f;
}

}  // namespace mechabrain

#endif  // FWMC_POPULATION_MODEL_H_
