#ifndef FWMC_LIGHT_MODEL_H_
#define FWMC_LIGHT_MODEL_H_

#include <cmath>
#include <vector>
#include "core/platform.h"

namespace mechabrain {

// Tissue optics model for two-photon optogenetic stimulation.
//
// Models light attenuation through neural tissue using the
// Beer-Lambert law with wavelength-dependent scattering and
// absorption coefficients from Drosophila brain measurements.
//
// References:
//   Helmchen & Denk 2005, doi:10.1038/nmeth818 (two-photon scattering)
//   Prakash et al. 2012, doi:10.1038/nmeth.2063 (Drosophila tissue optics)
//   Podgorski & Bhatt 2024, doi:10.1038/s41593-024-01604-4 (depth limits)

// Tissue optical properties at a given wavelength
struct TissueParams {
  float mu_s = 10.0f;    // scattering coefficient (1/mm), Prakash 2012
  float mu_a = 0.1f;     // absorption coefficient (1/mm)
  float g_aniso = 0.9f;  // anisotropy factor (forward scattering bias)
  float n_tissue = 1.36f; // refractive index of neural tissue

  // Effective attenuation for two-photon excitation:
  // mu_eff = sqrt(3 * mu_a * (mu_a + mu_s * (1 - g)))
  // This accounts for the reduced scattering coefficient.
  float EffectiveAttenuation() const {
    float mu_s_prime = mu_s * (1.0f - g_aniso);  // reduced scattering
    return std::sqrt(3.0f * mu_a * (mu_a + mu_s_prime));
  }
};

// Default tissue parameters by wavelength band.
// Two-photon excitation uses longer wavelengths (900-1100nm)
// which penetrate deeper than single-photon equivalents.
inline TissueParams TissueParamsForWavelength(float lambda_nm) {
  // Scattering decreases with wavelength roughly as lambda^(-1.5)
  // Absorption increases above ~900nm due to water
  // Values calibrated for Drosophila brain (Prakash 2012)
  if (lambda_nm < 500.0f) {
    // Blue/green (single-photon): high scattering
    return {15.0f, 0.15f, 0.88f, 1.36f};
  } else if (lambda_nm < 700.0f) {
    // Red/orange (single-photon): moderate scattering
    return {10.0f, 0.1f, 0.9f, 1.36f};
  } else if (lambda_nm < 1000.0f) {
    // NIR (two-photon window): low scattering, low absorption
    return {6.0f, 0.05f, 0.92f, 1.36f};
  } else {
    // Long NIR (>1000nm): very low scattering, rising water absorption
    return {4.0f, 0.2f, 0.93f, 1.36f};
  }
}

// Point spread function model for focused laser spots.
// Computes the effective irradiance at a target neuron given
// the focal point position and laser power.
struct LightModel {
  TissueParams tissue;
  float objective_na = 1.0f;       // numerical aperture of objective
  float focal_x = 0.0f;           // focal point position (um)
  float focal_y = 0.0f;
  float focal_z = 0.0f;
  float laser_power_mw = 10.0f;   // total laser power at objective back aperture

  // Lateral resolution (1/e^2 radius) for two-photon excitation (um)
  // w_0 = 0.325 * lambda / (sqrt(2) * NA)  (Zipfel et al. 2003)
  float LateralResolution(float lambda_nm) const {
    return 0.325f * lambda_nm * 0.001f / (1.414f * objective_na);
  }

  // Axial resolution for two-photon (um)
  // z_R = 0.532 * lambda / (sqrt(2) * NA^2)
  float AxialResolution(float lambda_nm) const {
    return 0.532f * lambda_nm * 0.001f / (1.414f * objective_na * objective_na);
  }

  // Compute irradiance at a 3D position (mW/mm^2) given focal point.
  // Accounts for:
  //   1. Beer-Lambert attenuation with depth
  //   2. Gaussian beam lateral spread
  //   3. Axial defocus penalty
  //
  // Position units: micrometers (um). Neuron positions in FWMC are
  // stored in nanometers, so divide by 1000 before calling.
  float IrradianceAt(float target_x, float target_y, float target_z,
                     float lambda_nm) const {
    // Distance from focal point
    float dx = target_x - focal_x;
    float dy = target_y - focal_y;
    float dz = target_z - focal_z;
    float lateral_dist = std::sqrt(dx * dx + dy * dy);

    // Beam waist parameters
    float w0 = LateralResolution(lambda_nm);
    float zR = AxialResolution(lambda_nm);

    // Gaussian beam profile (lateral)
    float lateral_factor = std::exp(-2.0f * (lateral_dist * lateral_dist) / (w0 * w0));

    // Axial defocus: intensity falls as 1/(1 + (dz/zR)^2)^2 for two-photon
    float axial_ratio = dz / zR;
    float axial_factor = 1.0f / ((1.0f + axial_ratio * axial_ratio) *
                                  (1.0f + axial_ratio * axial_ratio));

    // Beer-Lambert depth attenuation (depth = focal_z, in mm)
    float depth_mm = focal_z * 0.001f;  // um to mm
    float mu_eff = tissue.EffectiveAttenuation();
    float depth_factor = std::exp(-mu_eff * depth_mm);

    // Peak irradiance at focus (mW/mm^2)
    // P / (pi * w0^2), converting w0 from um to mm
    float w0_mm = w0 * 0.001f;
    float peak_irradiance = laser_power_mw / (kPi * w0_mm * w0_mm);

    return peak_irradiance * lateral_factor * axial_factor * depth_factor;
  }

  // Compute irradiance for multiple neurons at once.
  // Positions in um (caller converts from nm if needed).
  void ComputeIrradiance(const float* x, const float* y, const float* z,
                         float* out_irradiance, size_t n,
                         float lambda_nm) const {
    for (size_t i = 0; i < n; ++i) {
      out_irradiance[i] = IrradianceAt(x[i], y[i], z[i], lambda_nm);
    }
  }

  // Compute effective irradiance for a set of target neurons,
  // accounting for multi-spot holographic addressing.
  // Total power is split equally among N simultaneous targets.
  void ComputeMultiSpotIrradiance(
      const float* /*x*/, const float* /*y*/, const float* z,
      const uint32_t* target_indices, size_t n_targets,
      float* out_irradiance, size_t n_neurons,
      float lambda_nm) const {

    // Zero all outputs
    for (size_t i = 0; i < n_neurons; ++i) {
      out_irradiance[i] = 0.0f;
    }

    if (n_targets == 0) return;

    // Power split: each spot gets P/N
    float power_per_spot = laser_power_mw / static_cast<float>(n_targets);
    float w0 = LateralResolution(lambda_nm);
    float w0_mm = w0 * 0.001f;
    float peak_per_spot = power_per_spot / (kPi * w0_mm * w0_mm);

    float mu_eff = tissue.EffectiveAttenuation();

    for (size_t t = 0; t < n_targets; ++t) {
      uint32_t idx = target_indices[t];
      if (idx >= n_neurons) continue;

      float tz = z[idx];
      float depth_mm = tz * 0.001f;
      float depth_factor = std::exp(-mu_eff * depth_mm);

      // Each target gets full lateral intensity (hologram focuses independently)
      // but depth attenuation still applies
      out_irradiance[idx] = peak_per_spot * depth_factor;
    }
  }

  // Maximum usable depth (um) where irradiance drops to 1/e of surface.
  // Useful for planning which neurons are addressable.
  float MaxDepth(float /*lambda_nm*/) const {
    float mu_eff = tissue.EffectiveAttenuation();
    if (mu_eff <= 0.0f) return 1e6f;
    return 1000.0f / mu_eff;  // mm to um
  }
};

}  // namespace mechabrain

#endif  // FWMC_LIGHT_MODEL_H_
