#ifndef FWMC_MULTISCALE_BRIDGE_H_
#define FWMC_MULTISCALE_BRIDGE_H_

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/neuron_array.h"
#include "core/parametric_gen.h"

namespace mechabrain {

// Couples mean-field (Wilson-Cowan neural field) regions with spiking
// (Izhikevich) regions in a single simulation.
//
// Motivation: for brain regions that have not been scanned or where
// single-neuron resolution is unnecessary, running Wilson-Cowan dynamics
// is orders of magnitude cheaper than spiking simulation. As scan data
// arrives for a region, it can be promoted from mean-field to spiking
// without changing the rest of the model.
//
// The bridge handles the interface between the two scales:
//
//   Mean-field -> Spiking:
//     The excitatory field activity (E) in a mean-field region is
//     converted to an equivalent external current injected into the
//     spiking neurons of downstream regions. This models afferent
//     drive from the unresolved population.
//
//   Spiking -> Mean-field:
//     The instantaneous firing rate of a spiking region is converted
//     to an excitatory stimulus injected into the mean-field grid at
//     the corresponding spatial location. This models efferent drive
//     from the resolved population back into the continuum.
//
// Both conversions use linear scaling with configurable gain. The
// gains can be calibrated so that a fully mean-field simulation and
// a fully spiking simulation produce comparable population-level
// activity patterns.

// Declares which regions are spiking and which are mean-field.
enum class RegionScale : uint8_t {
  kSpiking = 0,     // full Izhikevich spiking network
  kMeanField = 1,   // Wilson-Cowan continuum
};

// Coupling between a mean-field region and a spiking region.
struct ScaleCoupling {
  std::string from_region;  // source region name
  std::string to_region;    // target region name
  float gain = 1.0f;        // scaling factor for cross-scale current
};

// Per-region scale assignment.
struct RegionScaleSpec {
  std::string name;
  RegionScale scale = RegionScale::kSpiking;

  // Mean-field parameters (only used if scale == kMeanField)
  float mean_e = 0.1f;   // initial excitatory activity [0,1]
  float mean_i = 0.05f;  // initial inhibitory activity [0,1]
};

// Manages cross-scale coupling at each timestep.
struct MultiscaleBridge {
  // Region assignments
  std::vector<RegionScaleSpec> region_scales;

  // Cross-scale couplings
  std::vector<ScaleCoupling> couplings;

  // Mean-field state for each mean-field region.
  // Indexed by region index (only valid for mean-field regions).
  struct MeanFieldState {
    float e = 0.1f;   // excitatory activity [0,1]
    float i = 0.05f;  // inhibitory activity [0,1]
  };
  std::vector<MeanFieldState> mf_state;

  // Wilson-Cowan parameters (shared across all mean-field regions)
  float tau_e = 10.0f;
  float tau_i = 20.0f;
  float w_ee = 12.0f;
  float w_ei = 4.0f;
  float w_ie = 13.0f;
  float w_ii = 2.0f;
  float h_e = -2.0f;
  float h_i = -3.5f;
  float beta = 1.0f;
  float theta = 4.0f;

  // Current injection gain: mean-field E -> spiking i_ext
  float mf_to_spike_gain = 20.0f;

  // Firing rate to mean-field gain: spiking rate (Hz) -> E injection
  float spike_to_mf_gain = 0.01f;

  // Initialize from region ranges and scale specs.
  void Init(const std::vector<ParametricGenerator::RegionRange>& ranges) {
    mf_state.resize(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      auto spec = FindSpec(ranges[i].name);
      if (spec) {
        mf_state[i].e = spec->mean_e;
        mf_state[i].i = spec->mean_i;
      }
    }
  }

  // Check if a region is mean-field.
  bool IsMeanField(const std::string& name) const {
    for (const auto& rs : region_scales) {
      if (rs.name == name && rs.scale == RegionScale::kMeanField) return true;
    }
    return false;
  }

  // Check if a region is spiking.
  bool IsSpiking(const std::string& name) const {
    return !IsMeanField(name);
  }

  // Step all mean-field regions (Wilson-Cowan ODE, no spatial diffusion).
  // This is the 0-dimensional version for when we do not have a VoxelGrid.
  void StepMeanField(float dt_ms) {
    for (auto& mf : mf_state) {
      float se = Sigmoid(w_ee * mf.e - w_ei * mf.i + h_e);
      float si = Sigmoid(w_ie * mf.e - w_ii * mf.i + h_i);
      mf.e += dt_ms / tau_e * (-mf.e + se);
      mf.i += dt_ms / tau_i * (-mf.i + si);
      mf.e = std::clamp(mf.e, 0.0f, 1.0f);
      mf.i = std::clamp(mf.i, 0.0f, 1.0f);
    }
  }

  // Apply cross-scale couplings.
  // Call this after StepMeanField() and before the spiking Izhikevich step.
  void ApplyCouplings(
      const std::vector<ParametricGenerator::RegionRange>& ranges,
      NeuronArray& neurons,
      float dt_ms) {

    for (const auto& c : couplings) {
      int from_idx = FindRegionIndex(ranges, c.from_region);
      int to_idx = FindRegionIndex(ranges, c.to_region);
      if (from_idx < 0 || to_idx < 0) continue;

      bool from_mf = IsMeanField(c.from_region);
      bool to_mf = IsMeanField(c.to_region);

      if (from_mf && !to_mf) {
        // Mean-field -> Spiking: inject current proportional to E activity
        float drive = mf_state[static_cast<size_t>(from_idx)].e *
                      mf_to_spike_gain * c.gain;
        auto& r = ranges[static_cast<size_t>(to_idx)];
        for (uint32_t i = r.start; i < r.end; ++i) {
          neurons.i_ext[i] += drive;
        }
      } else if (!from_mf && to_mf) {
        // Spiking -> Mean-field: compute firing rate and inject into E
        auto& r = ranges[static_cast<size_t>(from_idx)];
        uint32_t region_n = r.end - r.start;
        if (region_n == 0) continue;
        int spikes = 0;
        for (uint32_t i = r.start; i < r.end; ++i) {
          spikes += neurons.spiked[i];
        }
        float rate_hz = static_cast<float>(spikes) /
                        static_cast<float>(region_n) *
                        (1000.0f / dt_ms);
        float injection = rate_hz * spike_to_mf_gain * c.gain;
        mf_state[static_cast<size_t>(to_idx)].e =
            std::clamp(mf_state[static_cast<size_t>(to_idx)].e + injection,
                       0.0f, 1.0f);
      }
      // Spiking<->Spiking and MF<->MF handled by their own systems
    }
  }

  // Get the mean-field excitatory activity for a region.
  float GetActivity(size_t region_idx) const {
    if (region_idx < mf_state.size()) return mf_state[region_idx].e;
    return 0.0f;
  }

 private:
  static float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }

  // Parameterized sigmoid (matches NeuralField)
  float Sigmoid(float x, float b, float th) const {
    return 1.0f / (1.0f + std::exp(-b * (x - th)));
  }

  const RegionScaleSpec* FindSpec(const std::string& name) const {
    for (const auto& rs : region_scales) {
      if (rs.name == name) return &rs;
    }
    return nullptr;
  }

  static int FindRegionIndex(
      const std::vector<ParametricGenerator::RegionRange>& ranges,
      const std::string& name) {
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (ranges[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }
};

}  // namespace mechabrain

#endif  // FWMC_MULTISCALE_BRIDGE_H_
