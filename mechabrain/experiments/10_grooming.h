#ifndef FWMC_GROOMING_EXPERIMENT_H_
#define FWMC_GROOMING_EXPERIMENT_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/rate_monitor.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Results from a grooming hierarchy experiment.
//
// Drosophila grooming follows a stereotyped priority sequence when
// multiple body parts are simultaneously stimulated with dust:
//   1. Eyes (highest priority)
//   2. Antennae
//   3. Abdomen
//   4. Wings (lowest priority)
//
// The mechanism is NOT a chain: all motor programs activate in
// parallel, but higher-priority programs suppress lower ones via
// GABAergic inhibition. When the highest-priority stimulus ends,
// suppression lifts and the next program takes over.
//
// Seeds et al. 2014 eLife: "A suppression hierarchy among competing
// motor programs drives sequential grooming in Drosophila"
struct GroomingResult {
  // Dominant motor program per time window.
  // Window 0: 0-100ms (eye stimulus active, all stimuli on)
  // Window 1: 100-200ms (eye removed, antenna highest)
  // Window 2: 200-300ms (antenna removed, abdomen highest)
  // Window 3: 300-500ms (abdomen removed, wing only)
  // Values: 0=eye, 1=ant, 2=abd, 3=wing, -1=none
  int dominant[4] = {-1, -1, -1, -1};

  // Spike counts per channel per window
  int spikes_per_window[4][4] = {};  // [window][channel]

  // Suppression: in each window, ratio of dominant's spikes to total
  float dominance_ratio = 0.0f;

  // Timing
  double elapsed_seconds = 0.0;

  // Did the circuit produce output?
  bool responded() const {
    return dominant[0] >= 0 && dominant[1] >= 0;
  }

  // Is the grooming sequence correct?
  // Each window should be dominated by the highest remaining priority.
  bool sequence_correct() const {
    return dominant[0] == 0 &&  // eye dominates window 0
           dominant[1] == 1 &&  // antenna dominates window 1
           dominant[2] == 2 &&  // abdomen dominates window 2
           dominant[3] == 3;    // wing dominates window 3
  }

  bool passed() const {
    return sequence_correct() && dominance_ratio > 0.3f;
  }
};

// Self-contained Drosophila grooming hierarchy experiment.
//
// Circuit (Seeds et al. 2014, Hampel et al. 2015):
//
//   Sensory_eye (10) -----> DNg_eye (15) -----> MN_eye (10)
//   Sensory_ant (10) -----> DNg_ant (15) -----> MN_ant (10)
//   Sensory_abd (10) -----> DNg_abd (15) -----> MN_abd (10)
//   Sensory_wing (10) ----> DNg_wing (15) ----> MN_wing (10)
//
//   Suppression hierarchy (GABAergic):
//   DNg_eye  ----|  DNg_ant, DNg_abd, DNg_wing  (eye suppresses all)
//   DNg_ant  ----|  DNg_abd, DNg_wing            (ant suppresses lower)
//   DNg_abd  ----|  DNg_wing                     (abd suppresses wing)
//
//   Inhibitory interneurons (INH, 20) mediate the suppression.
//   Total: 180 neurons
//
// Protocol:
//   1. Simultaneous dust stimulus to all 4 body regions (0-500ms)
//   2. Remove eye stimulus at 100ms (simulate grooming completion)
//   3. Remove antenna stimulus at 200ms
//   4. Remove abdomen stimulus at 300ms
//   5. Measure motor neuron onset times and suppression
//
// Expected:
//   - Eye motor fires first (~5-15ms onset)
//   - Antenna motor fires after eye stimulus removed (~100-120ms)
//   - Abdomen motor fires after antenna removed (~200-220ms)
//   - Wing motor fires after abdomen removed (~300-320ms)
//   - Strong suppression of lower-priority programs during higher activity
struct GroomingExperiment {
  // Circuit parameters
  uint32_t n_sensory_per_region = 10;
  uint32_t n_dng_per_region = 15;
  uint32_t n_motor_per_region = 10;

  // Connectivity
  float sensory_to_dng_density = 0.4f;    // sensory -> own DNg (excitatory)
  float dng_to_motor_density = 0.4f;      // DNg -> own motor (excitatory)
  float suppress_density = 0.5f;          // sensory -> lower DNg (inhibitory)

  // Stimulus timing (ms)
  float stimulus_start = 0.0f;
  float eye_stim_end = 100.0f;      // eye grooming "completes" first
  float antenna_stim_end = 200.0f;
  float abdomen_stim_end = 300.0f;
  float wing_stim_end = 500.0f;
  float total_duration = 500.0f;

  // Stimulus current
  float stim_current = 12.0f;
  float dt_ms = 0.1f;

  GroomingResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937 rng(seed);

    // --- Region layout ---
    // 4 sensory regions + 4 DNg regions + 4 motor regions = 12 regions
    const uint32_t n_per_channel = n_sensory_per_region + n_dng_per_region +
                                    n_motor_per_region;
    const uint32_t total = n_per_channel * 4;

    NeuronArray neurons;
    neurons.Resize(total);
    SynapseTable synapses;
    CellTypeManager types;

    // Assign regions and cell types
    // Region layout per channel: sensory, dng, motor
    // Channels: 0=eye, 1=antenna, 2=abdomen, 3=wing
    struct ChannelRange {
      uint32_t sens_start, sens_end;
      uint32_t dng_start, dng_end;
      uint32_t motor_start, motor_end;
    };
    ChannelRange channels[4];

    uint32_t idx = 0;
    for (int ch = 0; ch < 4; ++ch) {
      channels[ch].sens_start = idx;
      channels[ch].sens_end = idx + n_sensory_per_region;
      for (uint32_t i = idx; i < idx + n_sensory_per_region; ++i) {
        neurons.region[i] = static_cast<uint8_t>(ch * 3);
        neurons.type[i] = static_cast<uint8_t>(CellType::kGroomingSensory);
      }
      idx += n_sensory_per_region;

      channels[ch].dng_start = idx;
      channels[ch].dng_end = idx + n_dng_per_region;
      for (uint32_t i = idx; i < idx + n_dng_per_region; ++i) {
        neurons.region[i] = static_cast<uint8_t>(ch * 3 + 1);
        neurons.type[i] = static_cast<uint8_t>(CellType::kDNg_command);
      }
      idx += n_dng_per_region;

      channels[ch].motor_start = idx;
      channels[ch].motor_end = idx + n_motor_per_region;
      for (uint32_t i = idx; i < idx + n_motor_per_region; ++i) {
        neurons.region[i] = static_cast<uint8_t>(ch * 3 + 2);
        neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
      }
      idx += n_motor_per_region;
    }

    // Assign biophysical params
    types.AssignFromTypes(neurons);

    // --- Build connectivity ---
    std::vector<uint32_t> pre_vec, post_vec;
    std::vector<float> weight_vec;
    std::vector<uint8_t> nt_vec;

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::normal_distribution<float> w_dist(1.0f, 0.2f);

    auto connect = [&](uint32_t from_start, uint32_t from_end,
                       uint32_t to_start, uint32_t to_end,
                       float density, float w_mean, NTType nt) {
      std::normal_distribution<float> wd(w_mean, w_mean * 0.2f);
      for (uint32_t i = from_start; i < from_end; ++i) {
        for (uint32_t j = to_start; j < to_end; ++j) {
          if (coin(rng) < density) {
            pre_vec.push_back(i);
            post_vec.push_back(j);
            weight_vec.push_back(std::max(0.01f, wd(rng)));
            nt_vec.push_back(static_cast<uint8_t>(nt));
          }
        }
      }
    };

    // Within each channel: sensory -> DNg -> motor (excitatory)
    // Weights scaled to overcome tau_syn=3ms exponential decay
    // (comparable to navigation experiment at 2-35 pA range).
    for (int ch = 0; ch < 4; ++ch) {
      connect(channels[ch].sens_start, channels[ch].sens_end,
              channels[ch].dng_start, channels[ch].dng_end,
              sensory_to_dng_density, 10.0f, kACh);
      connect(channels[ch].dng_start, channels[ch].dng_end,
              channels[ch].motor_start, channels[ch].motor_end,
              dng_to_motor_density, 8.0f, kACh);
    }

    // Suppression hierarchy (Seeds et al. 2014):
    // Each channel's DNg inhibits ALL lower-priority DNg populations.
    //   eye_DNg -| {ant_DNg, abd_DNg, wing_DNg}
    //   ant_DNg -| {abd_DNg, wing_DNg}
    //   abd_DNg -| {wing_DNg}
    //
    // DNg->DNg (not sensory->DNg): inhibition flows only when the
    // higher-priority DNg is actively firing. When stimulus is removed,
    // DNg stops, suppression lifts, and the next channel takes over.
    for (int ch = 0; ch < 3; ++ch) {
      for (int target = ch + 1; target < 4; ++target) {
        connect(channels[ch].dng_start, channels[ch].dng_end,
                channels[target].dng_start, channels[target].dng_end,
                suppress_density, 15.0f, kGABA);
      }
    }

    synapses.BuildFromCOO(total, pre_vec, post_vec, weight_vec, nt_vec);

    Log(LogLevel::kInfo, "[grooming] %u neurons, %zu synapses",
        total, synapses.Size());

    // --- Run simulation ---
    GroomingResult result;
    uint32_t steps = static_cast<uint32_t>(total_duration / dt_ms);

    for (uint32_t step = 0; step < steps; ++step) {
      float t_ms = step * dt_ms;

      // Clear external current
      std::fill(neurons.i_ext.begin(), neurons.i_ext.end(), 0.0f);

      // Apply dust stimuli (timed removal simulates grooming completion)
      auto inject_stim = [&](int ch, float end_time) {
        if (t_ms >= stimulus_start && t_ms < end_time) {
          for (uint32_t i = channels[ch].sens_start;
               i < channels[ch].sens_end; ++i) {
            neurons.i_ext[i] += stim_current;
          }
        }
      };
      inject_stim(0, eye_stim_end);
      inject_stim(1, antenna_stim_end);
      inject_stim(2, abdomen_stim_end);
      inject_stim(3, wing_stim_end);

      // Decay synaptic current, propagate spikes, step neurons
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(),
                               neurons.i_syn.data(), 1.0f);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, t_ms, types);

      // Count motor spikes per channel per time window
      // Windows: [0,100), [100,200), [200,300), [300,500)
      int win = -1;
      if (t_ms < eye_stim_end) win = 0;
      else if (t_ms < antenna_stim_end) win = 1;
      else if (t_ms < abdomen_stim_end) win = 2;
      else win = 3;

      for (int ch = 0; ch < 4; ++ch) {
        for (uint32_t i = channels[ch].motor_start;
             i < channels[ch].motor_end; ++i) {
          if (neurons.spiked[i]) {
            result.spikes_per_window[win][ch]++;
          }
        }
      }
    }

    // Determine dominant motor program per window
    float total_dominance = 0.0f;
    int n_windows = 0;
    for (int w = 0; w < 4; ++w) {
      int best_ch = -1, best_count = 0, total_spikes = 0;
      for (int ch = 0; ch < 4; ++ch) {
        total_spikes += result.spikes_per_window[w][ch];
        if (result.spikes_per_window[w][ch] > best_count) {
          best_count = result.spikes_per_window[w][ch];
          best_ch = ch;
        }
      }
      result.dominant[w] = best_ch;
      if (total_spikes > 0) {
        total_dominance += static_cast<float>(best_count) / total_spikes;
        ++n_windows;
      }
    }
    result.dominance_ratio = (n_windows > 0)
        ? total_dominance / n_windows : 0.0f;

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[grooming] done in %.2fs: dom=[%d,%d,%d,%d], dominance=%.2f, seq=%s",
        result.elapsed_seconds,
        result.dominant[0], result.dominant[1],
        result.dominant[2], result.dominant[3],
        result.dominance_ratio,
        result.sequence_correct() ? "CORRECT" : "WRONG");

    // Log per-window spike counts
    const char* names[] = {"eye", "ant", "abd", "wing"};
    for (int w = 0; w < 4; ++w) {
      Log(LogLevel::kInfo, "[grooming]   win%d: %s=%d %s=%d %s=%d %s=%d",
          w,
          names[0], result.spikes_per_window[w][0],
          names[1], result.spikes_per_window[w][1],
          names[2], result.spikes_per_window[w][2],
          names[3], result.spikes_per_window[w][3]);
    }

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_GROOMING_EXPERIMENT_H_
