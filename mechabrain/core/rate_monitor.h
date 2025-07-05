#ifndef FWMC_RATE_MONITOR_H_
#define FWMC_RATE_MONITOR_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "core/log.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Literature-derived firing rate ranges for Drosophila brain regions (Hz).
// Sources: Murthy & Turner 2013 (KC), Wilson 2013 (PN), Aso 2014 (MBON).
struct FiringRateReference {
  const char* region_name;
  float min_hz;    // lower bound of plausible range
  float max_hz;    // upper bound of plausible range
  float typical_hz; // typical spontaneous rate
};

inline const FiringRateReference kDrosophilaRates[] = {
  {"ORN",            1.0f,   40.0f,  5.0f},  // Turner et al. 2008
  {"PN",             2.0f,   30.0f,  8.0f},  // Wilson 2013
  {"KC",             0.5f,   10.0f,  1.5f},  // Murthy & Turner 2013 (sparse)
  {"MBON",           1.0f,   25.0f,  5.0f},  // Aso et al. 2014
  {"DAN",            0.5f,   15.0f,  3.0f},  // Cohn et al. 2015
  {"central_brain",  1.0f,   30.0f,  5.0f},
  {"optic_lobe",     5.0f,   60.0f, 15.0f},  // Behnia et al. 2014
  {"antennal_lobe",  2.0f,   30.0f,  8.0f},
  {"central_complex",2.0f,   20.0f,  5.0f},  // Seelig & Jayaraman 2015
  {"lateral_horn",   1.0f,   20.0f,  5.0f},
  {"sez",            2.0f,   25.0f,  8.0f},
};
constexpr int kNumDrosophilaRates =
    sizeof(kDrosophilaRates) / sizeof(kDrosophilaRates[0]);

// Literature-derived firing rate ranges for mammalian cortical regions (Hz).
// Used for validation of human cortical column simulations.
inline const FiringRateReference kMammalianCorticalRates[] = {
  // Cortical layers
  // Barth & Bhatt (2012) Neuroscience 202:1; de Kock et al (2007) J Physiol 585:169
  {"L23",              1.0f,   15.0f,  5.0f},  // L2/3 pyramidal + interneurons
  {"L4",               3.0f,   20.0f,  8.0f},  // L4 spiny stellate
  {"L5",               2.0f,   25.0f,  8.0f},  // L5 pyramidal (de Kock & Sakmann 2009)
  {"L6",               1.0f,   10.0f,  3.0f},  // L6 corticothalamic (Bortone et al 2014)
  // Functional areas
  {"cortex",           1.0f,   30.0f,  5.0f},  // generic cortical
  {"thalamus",         5.0f,   30.0f, 15.0f},  // thalamocortical relay (Sherman 2001)
  {"TRN",              5.0f,   20.0f, 10.0f},  // thalamic reticular (Pinault 2004)
  {"NBM",              1.0f,    8.0f,  3.0f},  // nucleus basalis (Zaborszky et al 2015)
  // Combined column
  {"cortical_column",  1.0f,   20.0f,  5.0f},
};
constexpr int kNumMammalianRates =
    sizeof(kMammalianCorticalRates) / sizeof(kMammalianCorticalRates[0]);

// Per-region firing rate statistics for one measurement window.
struct RegionRate {
  std::string name;
  int region_id = -1;
  int n_neurons = 0;
  int n_spikes = 0;
  float rate_hz = 0.0f;       // measured firing rate
  float ref_min_hz = 0.0f;    // literature lower bound
  float ref_max_hz = 0.0f;    // literature upper bound
  float ref_typical_hz = 0.0f;

  bool in_range() const { return rate_hz >= ref_min_hz && rate_hz <= ref_max_hz; }
};

// Monitors per-region firing rates over a sliding window and compares
// against literature reference values.
//
// Usage:
//   RateMonitor mon;
//   mon.Init(neurons, region_names, dt_ms);
//   // each timestep:
//   mon.RecordStep(neurons);
//   // periodically:
//   auto rates = mon.ComputeRates();
//   mon.LogRates(rates);
struct RateMonitor {
  float dt_ms = 1.0f;
  float window_ms = 1000.0f;   // measurement window length

  // Per-region spike accumulators
  struct RegionAccum {
    std::string name;
    int region_id = 0;
    std::vector<uint32_t> neuron_indices;
    int spike_count = 0;
    int step_count = 0;

    // Reference rates (looked up from kDrosophilaRates)
    float ref_min = 0.0f;
    float ref_max = 100.0f;
    float ref_typical = 10.0f;
  };
  std::vector<RegionAccum> regions;
  int total_steps = 0;

  // Initialize from neuron array. Region names are used to look up
  // reference firing rates from the literature table.
  void Init(const NeuronArray& neurons,
            const std::vector<std::string>& region_names,
            float sim_dt_ms) {
    dt_ms = sim_dt_ms;
    regions.clear();
    total_steps = 0;

    // Group neurons by region index
    int max_region = 0;
    for (size_t i = 0; i < neurons.n; ++i) {
      max_region = std::max(max_region, static_cast<int>(neurons.region[i]));
    }

    regions.resize(max_region + 1);
    for (int r = 0; r <= max_region; ++r) {
      regions[r].region_id = r;
      regions[r].name = (r < static_cast<int>(region_names.size()))
                             ? region_names[r]
                             : "region_" + std::to_string(r);
      // Look up reference rates (check Drosophila first, then mammalian)
      bool found = false;
      for (int k = 0; k < kNumDrosophilaRates && !found; ++k) {
        if (regions[r].name == kDrosophilaRates[k].region_name) {
          regions[r].ref_min = kDrosophilaRates[k].min_hz;
          regions[r].ref_max = kDrosophilaRates[k].max_hz;
          regions[r].ref_typical = kDrosophilaRates[k].typical_hz;
          found = true;
        }
      }
      for (int k = 0; k < kNumMammalianRates && !found; ++k) {
        if (regions[r].name == kMammalianCorticalRates[k].region_name) {
          regions[r].ref_min = kMammalianCorticalRates[k].min_hz;
          regions[r].ref_max = kMammalianCorticalRates[k].max_hz;
          regions[r].ref_typical = kMammalianCorticalRates[k].typical_hz;
          found = true;
        }
      }
    }

    for (size_t i = 0; i < neurons.n; ++i) {
      int r = neurons.region[i];
      if (r >= 0 && r < static_cast<int>(regions.size())) {
        regions[r].neuron_indices.push_back(static_cast<uint32_t>(i));
      }
    }
  }

  // Simplified init using region index alone (no name lookup).
  void Init(const NeuronArray& neurons, float sim_dt_ms) {
    std::vector<std::string> names;
    Init(neurons, names, sim_dt_ms);
  }

  // Record spikes for one timestep.
  void RecordStep(const NeuronArray& neurons) {
    for (auto& reg : regions) {
      for (uint32_t idx : reg.neuron_indices) {
        reg.spike_count += neurons.spiked[idx];
      }
      reg.step_count++;
    }
    total_steps++;
  }

  // Compute firing rates and reset accumulators.
  std::vector<RegionRate> ComputeRates() {
    std::vector<RegionRate> rates;
    for (auto& reg : regions) {
      if (reg.neuron_indices.empty()) continue;
      RegionRate rr;
      rr.name = reg.name;
      rr.region_id = reg.region_id;
      rr.n_neurons = static_cast<int>(reg.neuron_indices.size());
      rr.n_spikes = reg.spike_count;

      float duration_s = reg.step_count * dt_ms / 1000.0f;
      rr.rate_hz = (duration_s > 0.0f)
                       ? static_cast<float>(reg.spike_count) /
                             (static_cast<float>(rr.n_neurons) * duration_s)
                       : 0.0f;

      rr.ref_min_hz = reg.ref_min;
      rr.ref_max_hz = reg.ref_max;
      rr.ref_typical_hz = reg.ref_typical;
      rates.push_back(rr);

      // Reset for next window
      reg.spike_count = 0;
      reg.step_count = 0;
    }
    return rates;
  }

  // Log a rates snapshot with in-range/out-of-range markers.
  static void LogRates(const std::vector<RegionRate>& rates) {
    for (const auto& r : rates) {
      const char* status = r.in_range() ? "OK" : "OUT";
      Log(LogLevel::kInfo,
          "  [%s] %s: %.1f Hz (%d neurons, %d spikes) ref=[%.1f, %.1f]",
          status, r.name.c_str(), r.rate_hz, r.n_neurons, r.n_spikes,
          r.ref_min_hz, r.ref_max_hz);
    }
  }

  // Count how many regions are within their reference range.
  static int CountInRange(const std::vector<RegionRate>& rates) {
    int count = 0;
    for (const auto& r : rates) {
      if (r.in_range()) count++;
    }
    return count;
  }
};

}  // namespace mechabrain

#endif  // FWMC_RATE_MONITOR_H_
