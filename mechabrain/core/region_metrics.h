#ifndef FWMC_REGION_METRICS_H_
#define FWMC_REGION_METRICS_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/log.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"

namespace mechabrain {

// Per-region activity snapshot.
struct RegionSnapshot {
  std::string name;
  uint32_t start = 0;
  uint32_t end = 0;
  int spike_count = 0;
  float mean_v = 0.0f;
  float firing_rate_hz = 0.0f;  // spikes / (n_neurons * dt_window_s)
  float fraction_active = 0.0f; // fraction with last_spike_time > 0
};

// Tracks per-region metrics during parametric brain simulation.
struct RegionMetrics {
  std::vector<ParametricGenerator::RegionRange> regions;
  std::vector<std::vector<RegionSnapshot>> history;  // per-snapshot, per-region

  void Init(const ParametricGenerator& gen) {
    regions = gen.region_ranges;
    history.clear();
  }

  // Record a snapshot of per-region activity at the current time.
  void Record(const NeuronArray& neurons, float /*sim_time_ms*/, float dt_ms,
              int window_steps) {
    std::vector<RegionSnapshot> snap;
    float window_s = (window_steps * dt_ms) / 1000.0f;

    for (const auto& reg : regions) {
      RegionSnapshot rs;
      rs.name = reg.name;
      rs.start = reg.start;
      rs.end = reg.end;

      float v_sum = 0.0f;
      int active = 0;

      for (uint32_t i = reg.start; i < reg.end; ++i) {
        if (neurons.spiked[i]) rs.spike_count++;
        v_sum += neurons.v[i];
        if (neurons.last_spike_time[i] > 0.0f) active++;
      }

      uint32_t n = reg.end - reg.start;
      rs.mean_v = (n > 0) ? v_sum / static_cast<float>(n) : 0.0f;
      rs.fraction_active = (n > 0) ?
          static_cast<float>(active) / static_cast<float>(n) : 0.0f;
      rs.firing_rate_hz = (n > 0 && window_s > 0) ?
          static_cast<float>(rs.spike_count) /
          (static_cast<float>(n) * window_s) : 0.0f;

      snap.push_back(rs);
    }

    history.push_back(std::move(snap));
  }

  // Log the latest snapshot.
  void LogLatest() const {
    if (history.empty()) return;
    const auto& snap = history.back();
    for (const auto& rs : snap) {
      Log(LogLevel::kInfo,
          "  %-20s  spikes=%-4d  rate=%.1fHz  active=%.0f%%  mean_v=%.1f",
          rs.name.c_str(), rs.spike_count, rs.firing_rate_hz,
          rs.fraction_active * 100.0f, rs.mean_v);
    }
  }

  // Get cumulative stats for each region across all snapshots.
  void LogSummary() const {
    if (history.empty()) return;

    Log(LogLevel::kInfo, "Region summary (%zu snapshots):", history.size());
    for (size_t r = 0; r < regions.size(); ++r) {
      float total_spikes = 0;
      float max_rate = 0;
      float mean_active = 0;

      for (const auto& snap : history) {
        if (r < snap.size()) {
          total_spikes += snap[r].spike_count;
          max_rate = std::max(max_rate, snap[r].firing_rate_hz);
          mean_active += snap[r].fraction_active;
        }
      }
      mean_active /= static_cast<float>(history.size());

      uint32_t n = regions[r].end - regions[r].start;
      Log(LogLevel::kInfo,
          "  %-20s  n=%-5u  total_spikes=%.0f  peak_rate=%.1fHz  mean_active=%.0f%%",
          regions[r].name.c_str(), n, total_spikes, max_rate,
          mean_active * 100.0f);
    }
  }
};

// Pre-resolved stimulus: region name lookup done once at setup.
struct ResolvedStimulus {
  float start_ms = 0.0f;
  float end_ms = 0.0f;
  float intensity = 0.0f;
  uint32_t region_start = 0;
  uint32_t n_target = 0;  // how many neurons to stimulate
};

// Resolve stimulus specs to region index ranges (call once before sim loop).
// Eliminates per-timestep string comparisons in ApplyStimuli.
inline std::vector<ResolvedStimulus> ResolveStimuli(
    const std::vector<StimulusSpec>& stimuli,
    const std::vector<ParametricGenerator::RegionRange>& regions) {
  std::vector<ResolvedStimulus> resolved;
  for (const auto& stim : stimuli) {
    for (const auto& reg : regions) {
      if (reg.name != stim.target_region) continue;
      ResolvedStimulus rs;
      rs.start_ms = stim.start_ms;
      rs.end_ms = stim.end_ms;
      rs.intensity = stim.intensity;
      rs.region_start = reg.start;
      uint32_t n = reg.end - reg.start;
      rs.n_target = (stim.fraction < 1.0f) ?
          std::min(static_cast<uint32_t>(std::round(stim.fraction * n)), n) : n;
      resolved.push_back(rs);
      break;
    }
  }
  return resolved;
}

// Apply pre-resolved stimuli (no string comparisons, O(n_stimuli) per step).
inline void ApplyResolvedStimuli(const std::vector<ResolvedStimulus>& resolved,
                                  NeuronArray& neurons, float sim_time_ms) {
  for (const auto& rs : resolved) {
    if (sim_time_ms < rs.start_ms || sim_time_ms >= rs.end_ms) continue;
    for (uint32_t j = 0; j < rs.n_target; ++j) {
      neurons.i_ext[rs.region_start + j] += rs.intensity;
    }
  }
}

// Apply stimulus specs to neurons based on region ranges (legacy API).
// Called each step. Only active stimuli inject current.
// Prefer ResolveStimuli + ApplyResolvedStimuli for hot-path usage.
inline void ApplyStimuli(const std::vector<StimulusSpec>& stimuli,
                          const std::vector<ParametricGenerator::RegionRange>& regions,
                          NeuronArray& neurons, float sim_time_ms,
                          uint32_t /*seed*/) {
  for (const auto& stim : stimuli) {
    if (sim_time_ms < stim.start_ms || sim_time_ms >= stim.end_ms) continue;

    // Find target region
    for (const auto& reg : regions) {
      if (reg.name != stim.target_region) continue;

      uint32_t n = reg.end - reg.start;
      uint32_t n_target = static_cast<uint32_t>(
          std::round(stim.fraction * n));
      n_target = std::min(n_target, n);

      if (stim.fraction < 1.0f) {
        for (uint32_t j = 0; j < n_target; ++j) {
          neurons.i_ext[reg.start + j] += stim.intensity;
        }
      } else {
        for (uint32_t j = reg.start; j < reg.end; ++j) {
          neurons.i_ext[j] += stim.intensity;
        }
      }
      break;
    }
  }
}

}  // namespace mechabrain

#endif  // FWMC_REGION_METRICS_H_
