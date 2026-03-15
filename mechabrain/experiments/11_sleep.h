#ifndef FWMC_SLEEP_EXPERIMENT_H_
#define FWMC_SLEEP_EXPERIMENT_H_

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
#include "core/synapse_table.h"

namespace mechabrain {

// Results from a sleep-wake cycling experiment.
//
// Drosophila sleep is regulated by a recurrent circuit in the central
// complex involving dorsal fan-shaped body (dFB) sleep-promoting neurons,
// ellipsoid body R5 ring neurons that encode sleep pressure via
// activity-dependent plasticity, and helicon cells that gate locomotion.
//
// Key circuit (Donlea et al. 2018, Neuron 97:378-389):
//   R5 (EB ring neurons): sleep pressure accumulator
//   dFB (dorsal FB): sleep-promoting output, releases GABA + AstA
//   Helicon cells: locomotion gate, inhibited by dFB during sleep
//   OAA (octopaminergic arousal): wake-promoting, inhibited by dFB
//
// Sleep pressure builds via synaptic potentiation of R5->dFB during
// waking. When pressure exceeds threshold, dFB activates, silencing
// OAA and helicon cells. During sleep, R5 synapses undergo homeostatic
// downscaling, reducing pressure until dFB deactivates and wake resumes.
//
// Donlea et al. 2014 Science 343:97: R5 sleep drive
// Donlea et al. 2018 Neuron 97:378: dFB-helicon-R5 recurrent circuit
// Liu et al. 2016 Cell 165:1347: sleep resets synaptic potentiation
// Pimentel et al. 2016 Nature 536:333: dFB firing mode switch
struct SleepResult {
  // Number of sleep bouts detected
  int sleep_bouts = 0;

  // Number of wake bouts detected
  int wake_bouts = 0;

  // Mean sleep bout duration (ms)
  float mean_sleep_duration_ms = 0.0f;

  // Mean wake bout duration (ms)
  float mean_wake_duration_ms = 0.0f;

  // Sleep fraction (proportion of total time spent sleeping)
  float sleep_fraction = 0.0f;

  // Sleep deprivation rebound: ratio of post-deprivation sleep to baseline
  float rebound_ratio = 1.0f;

  // Whether sleep-wake cycling occurred
  bool cycling = false;

  // Whether sleep deprivation produced rebound
  bool rebound = false;

  // Timing
  double elapsed_seconds = 0.0;

  bool responded() const {
    return sleep_bouts >= 1 && wake_bouts >= 1;
  }

  // Sleep-wake cycling with rebound after deprivation
  bool passed() const {
    return cycling && sleep_bouts >= 2 && rebound;
  }
};

// Self-contained Drosophila sleep-wake cycling experiment.
//
// Circuit (Donlea et al. 2018):
//
//   R5 (20) --------> dFB (23) --------> OAA (15)  [GABA, inhibitory]
//     ^                  |                   |
//     |                  | [AstA, inhib]     | [OA, excitatory]
//     |                  v                   v
//     +------------- Helicon (20) -------> Motor (15)
//
//   Homeostatic plasticity on R5->dFB synapses:
//     Wake: potentiation (sleep pressure builds)
//     Sleep: depression (sleep pressure dissipates)
//
//   DN1p clock neurons (10) provide circadian excitation to dFB.
//
// Total: 103 neurons
//
// Protocol:
//   Phase 1 (0-2000ms): Free-running baseline, observe cycling
//   Phase 2 (2000-3000ms): Sleep deprivation (forced OAA activation)
//   Phase 3 (3000-5000ms): Recovery, observe rebound sleep
//
// Expected:
//   - Spontaneous sleep-wake cycling (period ~300-600ms scaled)
//   - Sleep deprivation prevents sleep in phase 2
//   - Rebound sleep in phase 3 (longer/deeper bouts)
struct SleepExperiment {
  // Circuit sizes (loosely matching Drosophila neuron counts)
  uint32_t n_R5 = 20;        // EB ring neurons (R2/R4m, renamed R5)
  uint32_t n_dFB = 23;       // dorsal fan-shaped body sleep neurons
  uint32_t n_helicon = 20;   // helicon cells (locomotion gate)
  uint32_t n_OAA = 15;       // octopaminergic arousal neurons
  uint32_t n_motor = 15;     // downstream motor/locomotion
  uint32_t n_DN1p = 10;      // clock neurons (circadian input)

  // Connectivity densities
  float R5_to_dFB_density = 0.4f;       // excitatory (glutamate/ACh)
  float dFB_to_OAA_density = 0.5f;      // inhibitory (GABA)
  float dFB_to_helicon_density = 0.5f;  // inhibitory (AstA/GABA)
  float OAA_to_helicon_density = 0.4f;  // excitatory (OA modulates helicon)
  float R5_recurrent_density = 0.1f;    // R5 recurrent excitation
  float helicon_to_motor_density = 0.5f;// excitatory (primary motor drive)
  float helicon_to_R5_density = 0.15f;  // weak excitatory feedback
  float DN1p_to_dFB_density = 0.3f;     // excitatory (circadian drive)

  // Homeostatic plasticity parameters
  // R5->dFB weights potentiate during wake, depress during sleep.
  // Proportional to distance from bound: creates natural oscillation.
  // Implements sleep pressure (Liu et al. 2016, Donlea et al. 2014).
  float potentiation_rate = 0.010f;   // max weight increase per ms during wake
  float depression_rate = 0.025f;     // max weight decrease per ms during sleep
  float w_min = 3.0f;                 // minimum R5->dFB weight (fully rested)
  float w_max = 22.0f;               // maximum R5->dFB weight (max sleep debt)
  float w_init = 10.0f;              // initial weight (moderate pressure)

  // Sleep detection: motor rate below this threshold = sleeping
  float sleep_threshold_hz = 3.0f;
  float wake_threshold_hz = 8.0f;

  // Timing
  float baseline_end_ms = 2000.0f;
  float deprivation_end_ms = 3000.0f;
  float total_duration_ms = 5000.0f;
  float dt_ms = 0.1f;

  // Deprivation current injected into OAA during phase 2
  float deprivation_current = 15.0f;

  // Background tonic drive
  float R5_tonic = 7.0f;      // tonic drive to R5 (waking sensory input)
  float OAA_tonic = 6.0f;     // tonic arousal drive to OAA
  float helicon_tonic = 5.0f; // tonic visual/sensory drive to helicon
  float DN1p_tonic = 2.0f;    // circadian oscillation amplitude

  SleepResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937 rng(seed);

    // --- Region layout ---
    const uint32_t total = n_R5 + n_dFB + n_helicon + n_OAA + n_motor + n_DN1p;

    NeuronArray neurons;
    neurons.Resize(total);
    SynapseTable synapses;
    CellTypeManager types;

    // Assign regions and types
    uint32_t idx = 0;
    uint32_t r5_start = idx;
    for (uint32_t i = idx; i < idx + n_R5; ++i) {
      neurons.region[i] = 0;  // EB
      neurons.type[i] = static_cast<uint8_t>(CellType::kRingNeuron);
    }
    idx += n_R5;

    uint32_t dfb_start = idx;
    for (uint32_t i = idx; i < idx + n_dFB; ++i) {
      neurons.region[i] = 1;  // dFB
      neurons.type[i] = static_cast<uint8_t>(CellType::kFC);
    }
    idx += n_dFB;

    uint32_t hel_start = idx;
    for (uint32_t i = idx; i < idx + n_helicon; ++i) {
      neurons.region[i] = 2;  // helicon
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    }
    idx += n_helicon;

    uint32_t oaa_start = idx;
    for (uint32_t i = idx; i < idx + n_OAA; ++i) {
      neurons.region[i] = 3;  // OAA
      neurons.type[i] = static_cast<uint8_t>(CellType::kOctopaminergic);
    }
    idx += n_OAA;

    uint32_t motor_start = idx;
    for (uint32_t i = idx; i < idx + n_motor; ++i) {
      neurons.region[i] = 4;  // motor
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    }
    idx += n_motor;

    uint32_t dn1p_start = idx;
    for (uint32_t i = idx; i < idx + n_DN1p; ++i) {
      neurons.region[i] = 5;  // DN1p clock
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    }

    types.AssignFromTypes(neurons);

    // --- Build connectivity ---
    std::vector<uint32_t> pre_vec, post_vec;
    std::vector<float> weight_vec;
    std::vector<uint8_t> nt_vec;

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    auto connect = [&](uint32_t from_start, uint32_t from_n,
                       uint32_t to_start, uint32_t to_n,
                       float density, float w_mean, NTType nt) {
      std::normal_distribution<float> wd(w_mean, w_mean * 0.2f);
      for (uint32_t i = from_start; i < from_start + from_n; ++i) {
        for (uint32_t j = to_start; j < to_start + to_n; ++j) {
          if (coin(rng) < density) {
            pre_vec.push_back(i);
            post_vec.push_back(j);
            weight_vec.push_back(std::max(0.01f, wd(rng)));
            nt_vec.push_back(static_cast<uint8_t>(nt));
          }
        }
      }
    };

    // R5 -> dFB (excitatory, sleep pressure pathway)
    // These weights undergo homeostatic plasticity
    uint32_t r5_dfb_syn_start = static_cast<uint32_t>(pre_vec.size());
    connect(r5_start, n_R5, dfb_start, n_dFB,
            R5_to_dFB_density, w_init, kACh);
    uint32_t r5_dfb_syn_end = static_cast<uint32_t>(pre_vec.size());

    // dFB -> OAA (inhibitory GABA, sleep suppresses arousal)
    connect(dfb_start, n_dFB, oaa_start, n_OAA,
            dFB_to_OAA_density, 18.0f, kGABA);

    // dFB -> helicon (inhibitory AstA/GABA, sleep suppresses locomotion)
    // This is the primary sleep effector: dFB silences helicon -> motor stops
    connect(dfb_start, n_dFB, hel_start, n_helicon,
            dFB_to_helicon_density, 20.0f, kGABA);

    // OAA -> helicon (excitatory, arousal drives helicon gate)
    connect(oaa_start, n_OAA, hel_start, n_helicon,
            OAA_to_helicon_density, 8.0f, kACh);

    // R5 recurrent excitation (maintains tonic firing for pressure accumulation)
    connect(r5_start, n_R5, r5_start, n_R5,
            R5_recurrent_density, 2.0f, kACh);

    // Helicon -> motor (excitatory, primary motor drive path)
    connect(hel_start, n_helicon, motor_start, n_motor,
            helicon_to_motor_density, 10.0f, kACh);

    // Helicon -> R5 (weak excitatory feedback, recurrent loop)
    connect(hel_start, n_helicon, r5_start, n_R5,
            helicon_to_R5_density, 3.0f, kACh);

    // DN1p -> dFB (excitatory circadian drive)
    connect(dn1p_start, n_DN1p, dfb_start, n_dFB,
            DN1p_to_dFB_density, 5.0f, kACh);

    synapses.BuildFromCOO(total, pre_vec, post_vec, weight_vec, nt_vec);

    Log(LogLevel::kInfo, "[sleep] %u neurons, %zu synapses, "
        "R5->dFB synapses: %u-%u",
        total, synapses.Size(), r5_dfb_syn_start, r5_dfb_syn_end);

    // Find R5->dFB synapse indices in CSR for homeostatic plasticity.
    // After BuildFromCOO, the COO indices no longer map directly to CSR.
    // Instead, identify them by pre in [r5_start, r5_start+n_R5) and
    // post in [dfb_start, dfb_start+n_dFB).
    std::vector<uint32_t> r5_dfb_indices;
    for (uint32_t pre = r5_start; pre < r5_start + n_R5; ++pre) {
      uint32_t row_start = synapses.row_ptr[pre];
      uint32_t row_end = synapses.row_ptr[pre + 1];
      for (uint32_t s = row_start; s < row_end; ++s) {
        if (synapses.post[s] >= dfb_start &&
            synapses.post[s] < dfb_start + n_dFB) {
          r5_dfb_indices.push_back(s);
        }
      }
    }

    Log(LogLevel::kInfo, "[sleep] %zu R5->dFB plastic synapses identified",
        r5_dfb_indices.size());

    // --- Run simulation ---
    SleepResult result;
    uint32_t steps = static_cast<uint32_t>(total_duration_ms / dt_ms);

    // State tracking
    bool currently_sleeping = false;
    float bout_start_ms = 0.0f;
    std::vector<float> sleep_durations;
    std::vector<float> wake_durations;
    float total_sleep_ms = 0.0f;

    // Rate estimation window (sliding 50ms)
    const int rate_window = static_cast<int>(50.0f / dt_ms);
    std::vector<int> motor_spike_history(rate_window, 0);
    int history_idx = 0;

    // Baseline sleep stats for rebound comparison
    float baseline_sleep_ms = 0.0f;
    float recovery_sleep_ms = 0.0f;

    for (uint32_t step = 0; step < steps; ++step) {
      float t_ms = step * dt_ms;

      // Clear external current
      std::fill(neurons.i_ext.begin(), neurons.i_ext.end(), 0.0f);

      // Tonic drives
      // R5: tonic input representing sensory/waking drive
      for (uint32_t i = r5_start; i < r5_start + n_R5; ++i) {
        neurons.i_ext[i] += R5_tonic;
      }

      // OAA: tonic arousal
      for (uint32_t i = oaa_start; i < oaa_start + n_OAA; ++i) {
        neurons.i_ext[i] += OAA_tonic;
      }

      // DN1p: slow circadian oscillation (sinusoidal, period ~1000ms scaled)
      float circadian = DN1p_tonic *
          (0.5f + 0.5f * std::sin(2.0f * 3.14159f * t_ms / 1000.0f));
      for (uint32_t i = dn1p_start; i < dn1p_start + n_DN1p; ++i) {
        neurons.i_ext[i] += circadian;
      }

      // Helicon: tonic visual/sensory drive (primary locomotion gate)
      for (uint32_t i = hel_start; i < hel_start + n_helicon; ++i) {
        neurons.i_ext[i] += helicon_tonic;
      }

      // Phase 2: sleep deprivation (mechanical stimulation keeps fly awake)
      // Inject current into helicon AND motor directly, bypassing dFB
      // inhibition. This models forced locomotion (e.g. rotating platform)
      // that prevents sleep regardless of sleep pressure.
      if (t_ms >= baseline_end_ms && t_ms < deprivation_end_ms) {
        for (uint32_t i = oaa_start; i < oaa_start + n_OAA; ++i) {
          neurons.i_ext[i] += deprivation_current;
        }
        for (uint32_t i = hel_start; i < hel_start + n_helicon; ++i) {
          neurons.i_ext[i] += deprivation_current * 0.5f;
        }
        for (uint32_t i = motor_start; i < motor_start + n_motor; ++i) {
          neurons.i_ext[i] += deprivation_current * 0.3f;
        }
      }

      // Decay synaptic current, propagate spikes, step neurons
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(),
                               neurons.i_syn.data(), 1.0f);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, t_ms, types);

      // Count motor spikes for rate estimation
      int motor_spikes = 0;
      for (uint32_t i = motor_start; i < motor_start + n_motor; ++i) {
        if (neurons.spiked[i]) motor_spikes++;
      }
      motor_spike_history[history_idx] = motor_spikes;
      history_idx = (history_idx + 1) % rate_window;

      // Compute motor firing rate (Hz)
      int total_motor_spikes = 0;
      for (int s : motor_spike_history) total_motor_spikes += s;
      float motor_rate_hz = static_cast<float>(total_motor_spikes) /
          (50.0f * 0.001f * n_motor);  // spikes / (window_sec * n_neurons)

      // Sleep-wake state detection with hysteresis
      bool was_sleeping = currently_sleeping;
      if (currently_sleeping && motor_rate_hz > wake_threshold_hz) {
        currently_sleeping = false;
      } else if (!currently_sleeping && motor_rate_hz < sleep_threshold_hz) {
        currently_sleeping = true;
      }

      // Bout tracking
      if (currently_sleeping != was_sleeping) {
        float bout_duration = t_ms - bout_start_ms;
        if (bout_duration > 10.0f) {  // ignore transients < 10ms
          if (was_sleeping) {
            sleep_durations.push_back(bout_duration);
            total_sleep_ms += bout_duration;
            if (t_ms < baseline_end_ms) baseline_sleep_ms += bout_duration;
            if (t_ms >= deprivation_end_ms) recovery_sleep_ms += bout_duration;
          } else {
            wake_durations.push_back(bout_duration);
          }
        }
        bout_start_ms = t_ms;
      }

      // Track sleep in current bout at end
      if (step == steps - 1 && currently_sleeping) {
        float bout_duration = t_ms - bout_start_ms;
        if (bout_duration > 10.0f) {
          sleep_durations.push_back(bout_duration);
          total_sleep_ms += bout_duration;
          if (t_ms >= deprivation_end_ms) recovery_sleep_ms += bout_duration;
        }
      }

      // Homeostatic plasticity on R5->dFB synapses (proportional).
      // Wake: dw = +rate * (w_max - w) * dt   (slows as ceiling approached)
      // Sleep: dw = -rate * (w - w_min) * dt   (slows as floor approached)
      // Creates natural oscillation around an equilibrium weight.
      if (!currently_sleeping) {
        for (uint32_t s : r5_dfb_indices) {
          float w = synapses.weight[s];
          synapses.weight[s] = std::min(w_max,
              w + potentiation_rate * (w_max - w) * dt_ms);
        }
      } else {
        // Depression during sleep (Liu et al. 2016: sleep resets potentiation)
        for (uint32_t s : r5_dfb_indices) {
          float w = synapses.weight[s];
          synapses.weight[s] = std::max(w_min,
              w - depression_rate * (w - w_min) * dt_ms);
        }
      }
    }

    // Compute results
    result.sleep_bouts = static_cast<int>(sleep_durations.size());
    result.wake_bouts = static_cast<int>(wake_durations.size());

    if (!sleep_durations.empty()) {
      float sum = 0.0f;
      for (float d : sleep_durations) sum += d;
      result.mean_sleep_duration_ms = sum / sleep_durations.size();
    }

    if (!wake_durations.empty()) {
      float sum = 0.0f;
      for (float d : wake_durations) sum += d;
      result.mean_wake_duration_ms = sum / wake_durations.size();
    }

    result.sleep_fraction = total_sleep_ms / total_duration_ms;
    result.cycling = (result.sleep_bouts >= 2 && result.wake_bouts >= 2);

    // Rebound: compare recovery sleep density to baseline sleep density
    float baseline_duration = baseline_end_ms;
    float recovery_duration = total_duration_ms - deprivation_end_ms;
    float baseline_density = baseline_sleep_ms / baseline_duration;
    float recovery_density = recovery_sleep_ms / recovery_duration;
    result.rebound_ratio = (baseline_density > 0.001f)
        ? recovery_density / baseline_density : 1.0f;
    result.rebound = (result.rebound_ratio > 1.1f);  // >10% more sleep after deprivation

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[sleep] done in %.2fs: %d sleep bouts, %d wake bouts, "
        "sleep_frac=%.2f, cycling=%s",
        result.elapsed_seconds,
        result.sleep_bouts, result.wake_bouts,
        result.sleep_fraction,
        result.cycling ? "YES" : "NO");

    Log(LogLevel::kInfo,
        "[sleep] mean_sleep=%.1fms, mean_wake=%.1fms, "
        "rebound=%.2fx (%s)",
        result.mean_sleep_duration_ms, result.mean_wake_duration_ms,
        result.rebound_ratio,
        result.rebound ? "YES" : "NO");

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_SLEEP_EXPERIMENT_H_
