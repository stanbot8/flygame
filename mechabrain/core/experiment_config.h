#ifndef FWMC_EXPERIMENT_CONFIG_H_
#define FWMC_EXPERIMENT_CONFIG_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/cell_type_defs.h"
#include "core/stimulus_event.h"

namespace mechabrain {

// Brain region identifiers matching FlyWire neuropil annotations
enum class Region : uint8_t {
  kUnknown = 0,
  kAL = 1,       // antennal lobe
  kMB = 2,       // mushroom body (Kenyon cells)
  kMBON = 3,     // mushroom body output neurons
  kDAN = 4,      // dopaminergic neurons
  kLH = 5,       // lateral horn
  kCX = 6,       // central complex
  kOL = 7,       // optic lobe
  kSEZ = 8,      // subesophageal zone
  kPN = 9,       // projection neurons
};

// Experiment configuration loaded from JSON or built programmatically
struct ExperimentConfig {
  // Metadata
  std::string name;
  std::string fly_strain;
  std::string date;
  std::string notes;

  // Simulation parameters
  float dt_ms = 0.1f;
  float duration_ms = 10000.0f;
  float weight_scale = 1.0f;
  int metrics_interval = 1000;
  bool enable_stdp = false;

  // Bridge mode (0=open-loop, 1=shadow, 2=closed-loop)
  int bridge_mode = 0;

  // Replacement thresholds
  float monitor_threshold = 0.6f;
  float bridge_threshold = 0.8f;
  float resync_threshold = 0.4f;
  float min_observation_ms = 10000.0f;

  // Neurons to monitor/replace (by index)
  std::vector<uint32_t> monitor_neurons;

  // Per-neuron cell type assignments (index → CellType)
  std::unordered_map<uint32_t, CellType> neuron_types;

  // Calibration
  int calibration_interval = 10000;  // apply gradient updates every N steps (0=disabled)
  float calibration_lr = 0.001f;     // learning rate for supervised calibration

  // Stimulus protocol: ordered list of timed events
  std::vector<StimulusEvent> stimulus_protocol;

  // Data paths
  std::string connectome_dir = "data";
  std::string recording_input;   // path to pre-recorded neural data (empty = none)
  std::string output_dir = "results";

  // Recording options
  bool record_spikes = true;
  bool record_voltages = false;        // expensive: records all v[i] each step
  bool record_shadow_metrics = true;
  bool record_per_neuron_error = true;
  int recording_interval = 1;          // record every N steps (1 = every step)
};

}  // namespace mechabrain

#endif  // FWMC_EXPERIMENT_CONFIG_H_
