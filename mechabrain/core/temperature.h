#ifndef FWMC_TEMPERATURE_H_
#define FWMC_TEMPERATURE_H_

#include <cmath>

namespace mechabrain {

// Temperature-dependent rate scaling for neural dynamics.
// Biological neural processes speed up with temperature following
// the Q10 rule: rate doubles (approx) for every 10C increase.
//
// Drosophila behavior is studied at 20-25C. Neural dynamics measured
// at room temp (~22C) vs in-vivo (~25C for a walking fly) differ
// by a factor of ~1.2-1.5x depending on the process.
//
// Q10 values for Drosophila (Soto-Padilla et al. 2018, Ueda & Wu 2015):
//   Ion channel kinetics: Q10 ~ 2-3
//   Synaptic release:     Q10 ~ 2-2.5
//   Membrane time constant: Q10 ~ 1.5-2
//   Spike threshold:       Q10 ~ 1 (relatively insensitive)
struct TemperatureModel {
  float reference_temp_c = 22.0f;  // temperature at which params were measured
  float current_temp_c = 25.0f;    // simulation temperature
  bool enabled = false;

  // Q10 coefficients for different processes
  float q10_channel = 2.5f;     // ion channel gating (affects Izhikevich a, b)
  float q10_synapse = 2.0f;     // synaptic release and decay
  float q10_membrane = 1.5f;    // membrane time constant

  // Compute Q10 scaling factor: Q10^((T_current - T_ref) / 10)
  // Returns 1.0 when disabled or at reference temperature.
  float ChannelScale() const {
    if (!enabled) return 1.0f;
    return std::pow(q10_channel, (current_temp_c - reference_temp_c) / 10.0f);
  }

  float SynapseScale() const {
    if (!enabled) return 1.0f;
    return std::pow(q10_synapse, (current_temp_c - reference_temp_c) / 10.0f);
  }

  float MembraneScale() const {
    if (!enabled) return 1.0f;
    return std::pow(q10_membrane, (current_temp_c - reference_temp_c) / 10.0f);
  }

  // Apply temperature scaling to Izhikevich parameters.
  // Scales 'a' (recovery rate) by channel Q10.
  // Does NOT modify the struct; returns scaled copies.
  float ScaledA(float a) const { return a * ChannelScale(); }

  // Synaptic tau scales inversely with temperature (faster at higher temp).
  float ScaledTauSyn(float tau_ms) const {
    float s = SynapseScale();
    return (s > 0.01f) ? tau_ms / s : tau_ms;
  }
};

}  // namespace mechabrain

#endif  // FWMC_TEMPERATURE_H_
