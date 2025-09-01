#ifndef FWMC_OPSIN_MODEL_H_
#define FWMC_OPSIN_MODEL_H_

#include <cmath>
#include <vector>

namespace mechabrain {

// Three-state opsin kinetic model for channelrhodopsins.
// Transitions: Closed -> Open -> Desensitized -> Closed
//
// Based on the Nikolic et al. (2009) three-state model
// (doi:10.1007/s11071-009-9484-6) with parameters from:
//   ChR2:     Nagel et al. 2003, doi:10.1073/pnas.1936192100
//   ChRmine:  Marshel et al. 2019, doi:10.1126/science.aaw5202
//   stGtACR2: Mahn et al. 2018, doi:10.1016/j.neuron.2018.01.040

enum class OpsinType : uint8_t {
  kChR2 = 0,       // excitatory, blue-light activated (470nm)
  kChRmine = 1,    // excitatory, red-shifted (590nm), large photocurrent
  kstGtACR2 = 2,   // inhibitory, soma-targeted anion channel (515nm)
};

// Kinetic parameters for a single opsin variant
struct OpsinParams {
  float tau_open_ms = 1.0f;       // time constant: closed to open (ms)
  float tau_close_ms = 10.0f;     // time constant: open to closed (ms)
  float tau_desens_ms = 50.0f;    // time constant: open to desensitized (ms)
  float tau_recover_ms = 500.0f;  // time constant: desensitized to closed (ms)
  float g_max = 1.0f;             // max conductance (nS), scales photocurrent
  float e_rev = 0.0f;             // reversal potential (mV), 0 for cation, -70 for anion
  float lambda_peak_nm = 470.0f;  // peak activation wavelength (nm)
  float lambda_width_nm = 40.0f;  // spectral half-width (nm)
  bool inhibitory = false;        // true for anion channels (GtACR family)
};

// Drosophila-specific adjustments:
// Fly neurons have higher input resistance (~1 GOhm vs ~100 MOhm in mammals)
// so the same conductance produces ~10x larger voltage deflection.
// Expression levels via UAS-GAL4 are typically lower than viral transduction.
// Tissue is thinner (brain ~500um deep) with less scattering at NIR.
//
// References:
//   Klapoetke et al. 2014, doi:10.1038/nmeth.2836 (CsChrimson in Drosophila)
//   Mohammad et al. 2017, doi:10.1038/s41592-017-0010-y (SPARC2 two-photon in fly)
//   Dana et al. 2019, doi:10.1038/s41592-019-0435-6 (two-photon holographic in fly)

// Scale factor for Drosophila vs mammalian neurons.
// Higher input resistance means less conductance needed.
inline constexpr float kDrosophilaGScale = 0.15f;

// Literature parameters for supported opsins
inline OpsinParams ParamsForOpsin(OpsinType type) {
  switch (type) {
    case OpsinType::kChR2:
      return {
        .tau_open_ms = 0.2f,       // fast activation (Nagel 2003)
        .tau_close_ms = 10.0f,     // ~10ms off kinetics
        .tau_desens_ms = 80.0f,    // slow desensitization
        .tau_recover_ms = 5000.0f, // very slow recovery from desensitized
        .g_max = 0.4f,             // ~0.4 nS single-channel (Lin 2011)
        .e_rev = 0.0f,             // nonselective cation
        .lambda_peak_nm = 470.0f,  // blue
        .lambda_width_nm = 40.0f,
        .inhibitory = false,
      };
    case OpsinType::kChRmine:
      return {
        .tau_open_ms = 0.5f,       // slightly slower activation
        .tau_close_ms = 30.0f,     // slower off kinetics than ChR2
        .tau_desens_ms = 200.0f,   // less prone to desensitization
        .tau_recover_ms = 3000.0f,
        .g_max = 2.5f,             // ~5x larger photocurrent than ChR2 (Marshel 2019)
        .e_rev = 0.0f,
        .lambda_peak_nm = 590.0f,  // red-shifted, good for deep tissue
        .lambda_width_nm = 50.0f,
        .inhibitory = false,
      };
    case OpsinType::kstGtACR2:
      return {
        .tau_open_ms = 0.3f,       // fast activation
        .tau_close_ms = 8.0f,      // fast off kinetics (Mahn 2018)
        .tau_desens_ms = 100.0f,
        .tau_recover_ms = 2000.0f,
        .g_max = 1.5f,             // large anion conductance
        .e_rev = -70.0f,           // chloride reversal (inhibitory)
        .lambda_peak_nm = 515.0f,  // green
        .lambda_width_nm = 35.0f,
        .inhibitory = true,
      };
  }
  return {};  // unreachable
}

// Drosophila-calibrated opsin parameters.
// Adjusts conductance for high input resistance and UAS-GAL4 expression levels.
inline OpsinParams DrosophilaParamsForOpsin(OpsinType type) {
  auto p = ParamsForOpsin(type);
  p.g_max *= kDrosophilaGScale;

  // Drosophila-specific kinetic adjustments (temperature: 25C vs 37C in mammals)
  // Slower kinetics at lower temperature (Q10 ~ 2.5)
  float q10_factor = 1.6f;  // approximate slowdown at 25C vs 37C
  p.tau_open_ms *= q10_factor;
  p.tau_close_ms *= q10_factor;
  p.tau_desens_ms *= q10_factor;
  p.tau_recover_ms *= q10_factor;

  return p;
}

// Per-neuron opsin state for the three-state model.
// State fractions: closed + open + desensitized = 1.0
struct OpsinState {
  float closed = 1.0f;
  float open = 0.0f;
  float desensitized = 0.0f;
};

// Opsin population model: tracks kinetic state for N neurons,
// computes photocurrent as a conductance term injected into i_ext.
struct OpsinPopulation {
  OpsinParams params;
  std::vector<OpsinState> states;  // per-neuron kinetic state
  std::vector<float> irradiance;   // per-neuron light power (mW/mm^2)

  void Init(size_t n_neurons, OpsinType type) {
    params = ParamsForOpsin(type);
    states.assign(n_neurons, {1.0f, 0.0f, 0.0f});
    irradiance.assign(n_neurons, 0.0f);
  }

  void Init(size_t n_neurons, const OpsinParams& p) {
    params = p;
    states.assign(n_neurons, {1.0f, 0.0f, 0.0f});
    irradiance.assign(n_neurons, 0.0f);
  }

  // Set light intensity for a specific neuron (mW/mm^2).
  // Called by the optogenetic writer when translating StimCommands.
  void SetIrradiance(uint32_t idx, float power_mw_mm2) {
    if (idx < irradiance.size()) {
      irradiance[idx] = power_mw_mm2;
    }
  }

  // Clear all irradiance (call at start of each step)
  void ClearIrradiance() {
    std::fill(irradiance.begin(), irradiance.end(), 0.0f);
  }

  // Step the kinetic model forward by dt_ms and compute photocurrents.
  // Returns per-neuron current (pA) to be added to i_ext.
  //
  // The light-dependent transition rate from closed to open is:
  //   k_open = irradiance * sigma / tau_open
  // where sigma is a normalized activation cross-section (set to 1.0
  // since irradiance is already normalized by the power curve).
  void Step(float dt_ms, const float* v_membrane, float* i_ext, size_t n) {
    size_t count = std::min(n, states.size());
    if (count == 0) return;

    // Hoist constant rate parameters out of per-neuron loop.
    // These depend only on opsin variant, not on individual neurons.
    const float inv_tau_open = 1.0f / params.tau_open_ms;
    const float k_close = 1.0f / params.tau_close_ms;
    const float k_desens = 1.0f / params.tau_desens_ms;
    const float k_recover = 1.0f / params.tau_recover_ms;
    const float k_close_desens = k_close + k_desens;
    const float neg_g_max = -params.g_max;  // pre-negate for current sign
    const float e_rev = params.e_rev;

    for (size_t i = 0; i < count; ++i) {
      auto& s = states[i];

      // Light-dependent opening rate
      float k_open = irradiance[i] * inv_tau_open;

      // Forward Euler integration of state fractions
      float d_closed = -k_open * s.closed + k_close * s.open + k_recover * s.desensitized;
      float d_open = k_open * s.closed - k_close_desens * s.open;
      float d_desens = k_desens * s.open - k_recover * s.desensitized;

      s.closed += dt_ms * d_closed;
      s.open += dt_ms * d_open;
      s.desensitized += dt_ms * d_desens;

      // Clamp to [0, 1] and renormalize (numerical safety)
      s.closed = std::max(0.0f, s.closed);
      s.open = std::max(0.0f, s.open);
      s.desensitized = std::max(0.0f, s.desensitized);
      float total = s.closed + s.open + s.desensitized;
      if (total > 0.0f) {
        float inv_total = 1.0f / total;
        s.closed *= inv_total;
        s.open *= inv_total;
        s.desensitized *= inv_total;
      } else {
        s.closed = 1.0f;
        s.open = 0.0f;
        s.desensitized = 0.0f;
      }

      // Photocurrent: I = -g_max * open_fraction * (V - E_rev)
      // Sign convention: positive i_ext = depolarization
      i_ext[i] += neg_g_max * s.open * (v_membrane[i] - e_rev);
    }
  }

  // Fraction of channels in the open state for a given neuron
  float OpenFraction(uint32_t idx) const {
    if (idx >= states.size()) return 0.0f;
    return states[idx].open;
  }

  // Fraction of channels desensitized (unavailable) for a given neuron
  float DesensitizedFraction(uint32_t idx) const {
    if (idx >= states.size()) return 0.0f;
    return states[idx].desensitized;
  }
};

}  // namespace mechabrain

#endif  // FWMC_OPSIN_MODEL_H_
