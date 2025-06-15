#ifndef FWMC_MOTOR_OUTPUT_H_
#define FWMC_MOTOR_OUTPUT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"

namespace mechabrain {

// Fictive locomotion command derived from descending neuron activity.
// Maps SEZ/descending neuron firing to behavioral primitives.
//
// Based on Drosophila motor control (Namiki et al. 2018, Bidaye et al. 2014):
//   - Forward velocity from symmetric descending neuron drive
//   - Turning from asymmetric L/R descending activity
//   - Approach/avoidance from MBON balance (Aso & Rubin 2016)
struct MotorCommand {
  float forward_velocity = 0.0f;  // mm/s, positive = forward
  float angular_velocity = 0.0f;  // rad/s, positive = left turn
  float approach_drive = 0.0f;    // positive = approach, negative = avoid
  float freeze = 0.0f;            // [0,1] freeze probability
};

// Maps neural activity to fictive locomotion.
//
// Reads spike rates from motor-relevant neurons (descending neurons in SEZ,
// MBONs for valence) and produces a MotorCommand at each timestep.
//
// Usage:
//   MotorOutput motor;
//   motor.Init(neurons, region_indices);
//   // each timestep:
//   motor.Update(neurons, dt_ms);
//   MotorCommand cmd = motor.Command();
struct MotorOutput {
  // Which neuron indices drive each motor channel
  std::vector<uint32_t> descending_left;   // left-biased descending neurons
  std::vector<uint32_t> descending_right;  // right-biased descending neurons
  std::vector<uint32_t> approach_neurons;  // MBONs driving approach
  std::vector<uint32_t> avoid_neurons;     // MBONs driving avoidance

  // Exponential moving average of spike rates (Hz)
  float rate_left = 0.0f;
  float rate_right = 0.0f;
  float rate_approach = 0.0f;
  float rate_avoid = 0.0f;

  // Parameters
  float tau_ms = 10.0f;            // rate smoothing time constant (snappy)
  float velocity_gain = 5.0f;      // spikes/s -> mm/s (flies accelerate fast)
  float turning_gain = 1.5f;       // asymmetry -> rad/s (sharp turns)
  float valence_gain = 1.0f;       // MBON rate -> approach/avoid drive
  float freeze_threshold = 0.5f;   // low total activity -> freeze
  float max_forward = 30.0f;       // max forward speed (mm/s, Drosophila ~30)
  float max_angular = 6.28f;       // max turn rate (rad/s)

  MotorCommand current;

  // Initialize with explicit neuron index lists.
  void Init(const std::vector<uint32_t>& desc_l,
            const std::vector<uint32_t>& desc_r,
            const std::vector<uint32_t>& approach,
            const std::vector<uint32_t>& avoid) {
    descending_left = desc_l;
    descending_right = desc_r;
    approach_neurons = approach;
    avoid_neurons = avoid;
    rate_left = rate_right = rate_approach = rate_avoid = 0.0f;
    current = {};
  }

  // Auto-initialize from neuron regions:
  //   sez_region = SEZ region index (split L/R by x-coordinate midpoint)
  //   mbon_region = MBON region index (approach = cholinergic, avoid = GABAergic)
  void InitFromRegions(const NeuronArray& neurons,
                       int sez_region, int mbon_region,
                       float midline_x = 250.0f) {
    descending_left.clear();
    descending_right.clear();
    approach_neurons.clear();
    avoid_neurons.clear();

    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == sez_region) {
        if (neurons.x[i] < midline_x)
          descending_left.push_back(static_cast<uint32_t>(i));
        else
          descending_right.push_back(static_cast<uint32_t>(i));
      }
      if (neurons.region[i] == mbon_region) {
        // Use cell type to distinguish approach/avoid MBONs:
        // cholinergic MBONs (type 2) drive approach
        // GABAergic MBONs (type 3) drive avoidance
        if (neurons.type[i] == 2 || neurons.type[i] == 0)
          approach_neurons.push_back(static_cast<uint32_t>(i));
        else if (neurons.type[i] == 3 || neurons.type[i] == 4)
          avoid_neurons.push_back(static_cast<uint32_t>(i));
      }
    }

    rate_left = rate_right = rate_approach = rate_avoid = 0.0f;
    current = {};
  }

  // Count spikes in a neuron subset.
  static float SpikeRate(const NeuronArray& neurons,
                         const std::vector<uint32_t>& indices) {
    if (indices.empty()) return 0.0f;
    int count = 0;
    for (uint32_t idx : indices) {
      count += neurons.spiked[idx];
    }
    return static_cast<float>(count) / static_cast<float>(indices.size());
  }

  // Update motor output from current neural state.
  void Update(const NeuronArray& neurons, float dt_ms) {
    float alpha = 1.0f - std::exp(-dt_ms / tau_ms);

    // Exponential moving average of spike rates
    float inst_left = SpikeRate(neurons, descending_left) * (1000.0f / dt_ms);
    float inst_right = SpikeRate(neurons, descending_right) * (1000.0f / dt_ms);
    float inst_approach = SpikeRate(neurons, approach_neurons) * (1000.0f / dt_ms);
    float inst_avoid = SpikeRate(neurons, avoid_neurons) * (1000.0f / dt_ms);

    rate_left += alpha * (inst_left - rate_left);
    rate_right += alpha * (inst_right - rate_right);
    rate_approach += alpha * (inst_approach - rate_approach);
    rate_avoid += alpha * (inst_avoid - rate_avoid);

    // Forward velocity: average bilateral drive.
    float mean_drive = (rate_left + rate_right) * 0.5f;
    current.forward_velocity = std::clamp(
        mean_drive * velocity_gain, -max_forward, max_forward);

    // Angular velocity: L/R asymmetry
    float asymmetry = rate_left - rate_right;
    current.angular_velocity = std::clamp(
        asymmetry * turning_gain, -max_angular, max_angular);

    // Approach/avoid valence
    current.approach_drive = (rate_approach - rate_avoid) * valence_gain;

    // Freeze when total descending drive is very low
    float total_drive = rate_left + rate_right;
    current.freeze = (total_drive < freeze_threshold) ? 1.0f : 0.0f;
  }

  const MotorCommand& Command() const { return current; }

  // Check if any motor neurons are assigned.
  bool HasMotorNeurons() const {
    return !descending_left.empty() || !descending_right.empty() ||
           !approach_neurons.empty() || !avoid_neurons.empty();
  }

  // Total number of motor-relevant neurons.
  size_t TotalNeurons() const {
    return descending_left.size() + descending_right.size() +
           approach_neurons.size() + avoid_neurons.size();
  }
};

}  // namespace mechabrain

#endif  // FWMC_MOTOR_OUTPUT_H_
