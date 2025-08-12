#ifndef FWMC_STIMULUS_H_
#define FWMC_STIMULUS_H_

#include <algorithm>
#include <vector>
#include "core/neuron_array.h"
#include "core/stimulus_event.h"

namespace mechabrain {

// Applies stimulus protocol events to neurons at the correct times.
// Handles timed odor delivery, shock pulses, optogenetic activation, etc.
struct StimulusController {
  std::vector<StimulusEvent> protocol;
  size_t next_event = 0;  // index of next event to check
  float max_current = 15.0f;  // suprathreshold current at intensity=1.0

  void LoadProtocol(const std::vector<StimulusEvent>& events) {
    protocol = events;
    next_event = 0;
    // Sort by start time for efficient scanning
    std::sort(protocol.begin(), protocol.end(),
        [](const StimulusEvent& a, const StimulusEvent& b) {
          return a.start_ms < b.start_ms;
        });
  }

  // Apply all active stimuli at the given simulation time
  void Apply(float sim_time_ms, NeuronArray& neurons) {
    for (size_t i = 0; i < protocol.size(); ++i) {
      const auto& ev = protocol[i];
      if (sim_time_ms < ev.start_ms) continue;
      if (sim_time_ms > ev.end_ms) continue;

      // Active: inject external current proportional to intensity
      float current = ev.intensity * max_current;  // scale to suprathreshold
      for (uint32_t idx : ev.target_neurons) {
        if (idx < neurons.n) {
          neurons.i_ext[idx] += current;
        }
      }
    }
  }

  // Query which events are active at a given time (for logging)
  std::vector<const StimulusEvent*> ActiveAt(float sim_time_ms) const {
    std::vector<const StimulusEvent*> active;
    for (const auto& ev : protocol) {
      if (sim_time_ms >= ev.start_ms && sim_time_ms <= ev.end_ms) {
        active.push_back(&ev);
      }
    }
    return active;
  }
};

}  // namespace mechabrain

#endif  // FWMC_STIMULUS_H_
