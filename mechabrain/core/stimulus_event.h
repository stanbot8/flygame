#ifndef FWMC_STIMULUS_EVENT_H_
#define FWMC_STIMULUS_EVENT_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mechabrain {

// A timed stimulus event (extracted to lightweight header to avoid
// pulling in all of experiment_config.h for consumers that only
// need the event definition).
struct StimulusEvent {
  float start_ms;
  float end_ms;
  float intensity;                   // normalized [0, 1]
  std::string label;                 // e.g., "odor_A", "shock", "light"
  std::vector<uint32_t> target_neurons;  // which neurons receive this stimulus
};

}  // namespace mechabrain

#endif  // FWMC_STIMULUS_EVENT_H_
