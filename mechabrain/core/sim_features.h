#ifndef FWMC_SIM_FEATURES_H_
#define FWMC_SIM_FEATURES_H_

namespace mechabrain {

// Central feature flags for brain simulation subsystems.
// Each flag toggles an independent subsystem on/off without affecting others.
// When off, the subsystem's per-step cost is zero (no computation, no memory
// traffic). When on, full biological fidelity is maintained.
//
// This decoupling is critical for scaling: a human brain sim (~86B neurons)
// needs to disable expensive features like STDP and structural plasticity
// during phases that don't require them, without changing any other code.
struct SimFeatures {
  // Core dynamics (these should almost always be on)
  bool izhikevich = true;        // spiking neuron dynamics
  bool synaptic_decay = true;    // exponential synaptic current decay
  bool spike_propagation = true; // CSR synapse walk

  // Plasticity (expensive, disable for pure inference)
  bool stdp = false;                // spike-timing dependent plasticity
  bool structural_plasticity = false; // synapse pruning/sprouting
  bool homeostasis = true;          // intrinsic excitability adjustment

  // Connectivity features
  bool gap_junctions = true;        // electrical synapses
  bool short_term_plasticity = false; // Tsodyks-Markram facilitation/depression
  bool stochastic_release = false;  // Monte Carlo vesicle release

  // Neuromodulation
  bool neuromodulation = true;      // DA/5HT/OA dynamics
  bool neuromodulator_effects = true; // DA/5HT/OA excitability modulation

  // Inhibitory plasticity
  bool inhibitory_plasticity = false; // Vogels iSTDP for E/I balance

  // Adaptation and temperature
  bool sfa = true;                  // spike-frequency adaptation (calcium sAHP)
  bool conduction_delays = true;    // distance-dependent axonal delays

  // Receptor dynamics
  bool nmda = true;                 // NMDA receptor voltage-dependent Mg2+ block
  bool calcium_plasticity = false;  // calcium-dependent plasticity (NMDA Ca -> LTP/LTD)

  // Sensorimotor (viewer only)
  bool cpg = true;                  // central pattern generator
  bool proprioception = true;       // body state feedback
  bool motor_output = true;         // descending neuron readout

  // Propagation mode
  bool per_step_propagation = true; // propagate spikes every substep (accurate timing)
                                    // When false, spikes accumulate across substeps and
                                    // propagate in a single CSR walk (faster, but adds
                                    // up to one frame of artificial latency).

  // Diagnostics
  bool recording = false;           // spike/voltage recording to disk
  bool rate_monitor = true;         // per-region firing rate validation vs literature

  // Count enabled features (for logging)
  int CountEnabled() const {
    int c = 0;
    if (izhikevich) ++c;
    if (synaptic_decay) ++c;
    if (spike_propagation) ++c;
    if (stdp) ++c;
    if (structural_plasticity) ++c;
    if (homeostasis) ++c;
    if (gap_junctions) ++c;
    if (short_term_plasticity) ++c;
    if (stochastic_release) ++c;
    if (neuromodulation) ++c;
    if (neuromodulator_effects) ++c;
    if (inhibitory_plasticity) ++c;
    if (sfa) ++c;
    if (conduction_delays) ++c;
    if (nmda) ++c;
    if (calcium_plasticity) ++c;
    if (cpg) ++c;
    if (proprioception) ++c;
    if (motor_output) ++c;
    if (per_step_propagation) ++c;
    if (recording) ++c;
    if (rate_monitor) ++c;
    return c;
  }

  // Preset: maximum biological fidelity (all features on)
  static SimFeatures Full() {
    SimFeatures f;
    f.stdp = true;
    f.structural_plasticity = true;
    f.short_term_plasticity = true;
    f.stochastic_release = true;
    f.inhibitory_plasticity = true;
    f.calcium_plasticity = true;
    f.recording = true;
    f.rate_monitor = true;
    return f;
  }

  // Preset: fast inference (minimal features for forward pass)
  static SimFeatures Inference() {
    SimFeatures f;
    f.homeostasis = false;
    f.gap_junctions = false;
    f.neuromodulation = false;
    f.sfa = false;
    f.conduction_delays = false;
    f.nmda = false;
    f.per_step_propagation = false;
    return f;
  }

  // Preset: connectivity-only (for benchmarking propagation)
  static SimFeatures Minimal() {
    SimFeatures f;
    f.homeostasis = false;
    f.gap_junctions = false;
    f.neuromodulation = false;
    f.sfa = false;
    f.conduction_delays = false;
    f.nmda = false;
    f.per_step_propagation = false;
    f.cpg = false;
    f.proprioception = false;
    f.motor_output = false;
    return f;
  }
};

}  // namespace mechabrain

#endif  // FWMC_SIM_FEATURES_H_
