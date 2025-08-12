#ifndef FWMC_TWIN_BRIDGE_H_
#define FWMC_TWIN_BRIDGE_H_

#include <chrono>
#include <memory>
#include "bridge/bridge_channel.h"
#include "bridge/shadow_tracker.h"
#include "bridge/neuron_replacer.h"
#include "bridge/optogenetic_writer.h"
#include "bridge/validation.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"

namespace mechabrain {

// Operating modes for the bridge
enum class BridgeMode {
  kOpenLoop,    // Phase 1: predict only, no feedback
  kShadow,     // Phase 2: run in parallel, measure drift
  kClosedLoop, // Phase 3+: bidirectional, replacing neurons
};

// Main bridge controller.
// Orchestrates the read channel, digital twin simulation,
// shadow tracking, neuron replacement, and write channel
// in a single real-time loop.
struct TwinBridge {
  // Components
  NeuronArray digital;          // the digital twin neuron state
  SynapseTable synapses;        // connectome graph
  IzhikevichParams izh_params;  // neuron dynamics parameters
  ShadowTracker shadow;
  NeuronReplacer replacer;
  OptogeneticWriter writer;

  // I/O channels (owned, polymorphic)
  std::unique_ptr<ReadChannel> read_channel;
  std::unique_ptr<WriteChannel> write_channel;

  // Configuration
  BridgeMode mode = BridgeMode::kOpenLoop;
  float dt_ms = 0.1f;
  float weight_scale = 1.0f;
  float resync_threshold = 0.4f;  // auto-resync if correlation drops below
  float resync_cooldown_ms = 100.0f;  // minimum time between resyncs

  // Hierarchical loop rates:
  //   Fast (every step):     simulation dynamics + spike propagation
  //   Medium (every N steps): shadow measurement + read channel decode
  //   Slow (every M steps):   actuation commands + SLM hologram update
  int shadow_interval = 10;     // medium tick: 1ms at dt=0.1ms
  int actuation_interval = 50;  // slow tick: 5ms at dt=0.1ms

  // Bio input injection parameters
  float bio_spike_threshold = 0.7f;    // spike_prob above this → inject current
  float bio_injection_current = 15.0f; // current injected for detected bio spikes

  // Adaptive boundary: auto-promote neighbors of drifting neurons
  bool adaptive_boundaries = false;
  float boundary_drift_threshold = 0.5f;
  std::vector<std::vector<uint32_t>> adjacency;      // post-synaptic (forward)
  std::vector<std::vector<uint32_t>> reverse_adjacency; // pre-synaptic (backward)

  // Metrics
  float sim_time_ms = 0.0f;
  int total_steps = 0;
  int total_resyncs = 0;

  // Real-time latency tracking
  float last_step_us = 0.0f;       // last step duration in microseconds
  float max_step_us = 0.0f;        // worst-case step duration
  float mean_step_us = 0.0f;       // running average
  int deadline_misses = 0;         // steps that exceeded dt_ms

  // Per-component latency profiler (microseconds)
  // Tracks time spent in each phase of the bridge loop.
  struct LatencyProfile {
    float propagate_us = 0.0f;   // spike propagation (CSR traversal)
    float bio_inject_us = 0.0f;  // bio input injection
    float neuron_step_us = 0.0f; // Izhikevich/LIF dynamics
    float shadow_us = 0.0f;     // shadow tracking + correlation
    float actuation_us = 0.0f;  // command generation + write channel
    float opsin_us = 0.0f;      // opsin kinetics step
    float validation_us = 0.0f; // validation recording
    float total_us = 0.0f;      // sum of all components

    void Print() const {
      printf("  Latency breakdown (us):\n");
      printf("    propagate:  %8.1f  (%.1f%%)\n", propagate_us, 100.0f * propagate_us / std::max(total_us, 0.01f));
      printf("    bio_inject: %8.1f  (%.1f%%)\n", bio_inject_us, 100.0f * bio_inject_us / std::max(total_us, 0.01f));
      printf("    neuron_step:%8.1f  (%.1f%%)\n", neuron_step_us, 100.0f * neuron_step_us / std::max(total_us, 0.01f));
      printf("    shadow:     %8.1f  (%.1f%%)\n", shadow_us, 100.0f * shadow_us / std::max(total_us, 0.01f));
      printf("    actuation:  %8.1f  (%.1f%%)\n", actuation_us, 100.0f * actuation_us / std::max(total_us, 0.01f));
      printf("    opsin:      %8.1f  (%.1f%%)\n", opsin_us, 100.0f * opsin_us / std::max(total_us, 0.01f));
      printf("    validation: %8.1f  (%.1f%%)\n", validation_us, 100.0f * validation_us / std::max(total_us, 0.01f));
      printf("    TOTAL:      %8.1f\n", total_us);
    }
  };
  bool enable_profiling = false;
  LatencyProfile last_profile;
  LatencyProfile mean_profile;

  // Validation: compare digital twin against biological recordings
  bool enable_validation = false;
  ValidationEngine validator;
  std::vector<SpikeTrain> sim_spike_trains;
  std::vector<SpikeTrain> bio_spike_trains;

  void Init(size_t n_neurons) {
    digital.Resize(n_neurons);
    replacer.Init(n_neurons);
    writer.InitSafety(n_neurons);
    // Prevent division by zero in modulo checks
    if (shadow_interval == 0) shadow_interval = 1;
    if (actuation_interval == 0) actuation_interval = 1;
  }

  // Build adjacency lists from CSR synapse table for adaptive boundary detection.
  // Forward: adjacency[pre] = post-synaptic targets (output neurons).
  // Reverse: reverse_adjacency[post] = pre-synaptic sources (input neurons).
  // Both directions are needed: divergence propagates forward through outputs
  // AND backward through inputs (a drifting neuron receives wrong input from
  // its pre-synaptic partners, so monitoring them catches upstream causes).
  void BuildAdjacency() {
    adjacency.resize(synapses.n_neurons);
    reverse_adjacency.resize(synapses.n_neurons);
    for (size_t i = 0; i < synapses.n_neurons; ++i) {
      adjacency[i].clear();
      reverse_adjacency[i].clear();
    }
    for (size_t pre = 0; pre < synapses.n_neurons; ++pre) {
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        uint32_t post = synapses.post[s];
        adjacency[pre].push_back(post);
        if (post < synapses.n_neurons) {
          reverse_adjacency[post].push_back(static_cast<uint32_t>(pre));
        }
      }
    }
  }

  // Run one simulation step of the bridge loop.
  // Hierarchical tick structure:
  //   FAST  (every step):           simulation dynamics + spike propagation
  //   MEDIUM (every shadow_interval): read channel + shadow measurement
  //   SLOW  (every actuation_interval): write-back + SLM hologram update
  //
  // This reduces I/O and heavy computation to their natural timescales
  // while keeping the simulation loop tight for numerical accuracy.
  void Step() {
    auto step_start = std::chrono::high_resolution_clock::now();

    bool is_medium_tick = (total_steps % shadow_interval == 0);
    bool is_slow_tick = (total_steps % actuation_interval == 0);

    // --- MEDIUM TICK: Read biological state ---
    std::vector<BioReading> bio_readings;
    if (is_medium_tick && read_channel) {
      bio_readings = read_channel->ReadFrame(sim_time_ms);
    } else if (!is_medium_tick && read_channel && !last_bio_readings_.empty()) {
      // Reuse last bio readings on fast ticks (stale but cheap)
      bio_readings = last_bio_readings_;
    }

    // --- FAST TICK: always runs ---
    auto tp0 = std::chrono::high_resolution_clock::now();

    // Synaptic input: either clear (delta) or exponentially decay (realistic)
    if (izh_params.tau_syn_ms > 0.0f) {
      digital.DecaySynapticInput(dt_ms, izh_params.tau_syn_ms);
    } else {
      digital.ClearSynapticInput();
    }
    synapses.PropagateSpikes(digital.spiked.data(), digital.i_syn.data(),
                             weight_scale);

    auto tp1 = std::chrono::high_resolution_clock::now();

    // Inject biological inputs into digital twin
    if (mode != BridgeMode::kOpenLoop) {
      for (const auto& r : bio_readings) {
        if (r.neuron_idx >= digital.n) continue;
        auto nstate = replacer.state[r.neuron_idx];
        if (nstate == NeuronReplacer::State::kMonitored ||
            nstate == NeuronReplacer::State::kBiological) {
          if (r.spike_prob > bio_spike_threshold) {
            digital.i_ext[r.neuron_idx] += bio_injection_current;
          }
        }
      }
    }

    auto tp2 = std::chrono::high_resolution_clock::now();

    // Step digital twin dynamics
    IzhikevichStep(digital, dt_ms, sim_time_ms, izh_params);

    auto tp3 = std::chrono::high_resolution_clock::now();

    // Record spikes for validation (if enabled)
    if (enable_validation) {
      ValidationEngine::RecordSpikes(digital, sim_time_ms, sim_spike_trains);
      if (!bio_readings.empty()) {
        ValidationEngine::RecordBioSpikes(bio_readings, sim_time_ms,
                                          0.5f, bio_spike_trains);
      }
    }

    auto tp4 = std::chrono::high_resolution_clock::now();

    // --- MEDIUM TICK: Shadow tracking ---
    auto tp_shadow_start = std::chrono::high_resolution_clock::now();
    if (is_medium_tick &&
        (mode == BridgeMode::kShadow || mode == BridgeMode::kClosedLoop)) {
      auto drift = shadow.Measure(digital, bio_readings, sim_time_ms);

      for (const auto& r : bio_readings) {
        if (r.neuron_idx >= digital.n) continue;
        float match = (digital.spiked[r.neuron_idx] && r.spike_prob > 0.5f) ||
                      (!digital.spiked[r.neuron_idx] && r.spike_prob < 0.5f)
                      ? 1.0f : 0.0f;
        replacer.UpdateCorrelation(r.neuron_idx, match, dt_ms * shadow_interval);
      }

      // Auto-resync with cooldown: trigger on hard threshold, accelerating
      // drift, or pathological firing rate divergence
      float time_since_resync = sim_time_ms - shadow.last_resync_time;
      bool needs_resync = shadow.DriftExceedsThreshold(resync_threshold)
                       || shadow.RateDiverged(5.0f)
                       || shadow.DriftAccelerating();
      if (needs_resync && time_since_resync >= resync_cooldown_ms) {
        shadow.Resync(digital, bio_readings, sim_time_ms);
        total_resyncs++;
        replacer.RollbackDiverged(resync_threshold);
      }

      // Adaptive boundary refinement: expand monitoring to neighbors of
      // drifting neurons in both directions (post-synaptic targets catch
      // downstream propagation, pre-synaptic sources catch upstream causes)
      if (adaptive_boundaries && !adjacency.empty()) {
        auto expanded = replacer.AutoPromoteNeighbors(
            adjacency, boundary_drift_threshold, reverse_adjacency);
        if (!expanded.empty()) {
          Log(LogLevel::kInfo,
              "t=%.1fms: adaptive boundary expanded %zu neurons to MONITORED",
              sim_time_ms, expanded.size());
        }
      }
    }

    auto tp_shadow_end = std::chrono::high_resolution_clock::now();

    // --- SLOW TICK: Actuation + replacement advancement ---
    auto tp_act_start = std::chrono::high_resolution_clock::now();
    if (is_slow_tick && mode == BridgeMode::kClosedLoop) {
      // Write back to biology
      if (write_channel) {
        // Check for pre-staged pattern first (predictive pre-staging)
        const auto* staged = writer.GetStagedPattern(sim_time_ms);
        std::vector<StimCommand> commands;
        if (staged) {
          commands = staged->commands;
        } else {
          commands = writer.GenerateCommands(digital, bio_readings, sim_time_ms);
        }

        // Filter: only BRIDGED or REPLACED neurons
        std::vector<StimCommand> filtered;
        for (const auto& cmd : commands) {
          for (const auto& m : writer.target_map) {
            if (m.bio_target_idx == cmd.neuron_idx) {
              auto s = replacer.state[m.digital_idx];
              if (s == NeuronReplacer::State::kBridged ||
                  s == NeuronReplacer::State::kReplaced) {
                filtered.push_back(cmd);
              }
              break;
            }
          }
        }

        // Galvo-SLM hybrid split
        auto split = writer.SplitGalvoSLM(filtered);
        // Send galvo commands immediately (fast retarget)
        // Send SLM commands as batch hologram update
        // Both go through write channel; hardware sorts the addressing
        std::vector<StimCommand> combined;
        combined.insert(combined.end(), split.galvo.begin(), split.galvo.end());
        combined.insert(combined.end(), split.slm.begin(), split.slm.end());
        // Apply opsin kinetics if enabled (photocurrent injection)
        // Use elapsed time since last slow tick, not the fast dt
        writer.ApplyOpsinStep(combined, digital, dt_ms * actuation_interval);

        write_channel->WriteFrame(combined);
      }

      // Pre-stage next patterns for future actuation ticks
      writer.PreStagePatterns(digital, bio_readings, sim_time_ms, dt_ms);

      // Try to advance neuron replacement states
      auto promoted = replacer.TryAdvance();
      if (!promoted.empty()) {
        Log(LogLevel::kInfo,
            "t=%.1fms: %zu neurons promoted (%.1f%% replaced)",
            sim_time_ms, promoted.size(),
            replacer.ReplacementFraction() * 100.0f);
      }
    }

    auto tp_act_end = std::chrono::high_resolution_clock::now();

    // Cache bio readings for fast ticks
    if (is_medium_tick) {
      last_bio_readings_ = std::move(bio_readings);
    }

    sim_time_ms += dt_ms;
    total_steps++;

    // Latency measurement
    auto step_end = std::chrono::high_resolution_clock::now();
    last_step_us = std::chrono::duration<float, std::micro>(step_end - step_start).count();
    if (last_step_us > max_step_us) max_step_us = last_step_us;
    mean_step_us += 0.001f * (last_step_us - mean_step_us);

    // Per-component profiling
    if (enable_profiling) {
      auto us = [](auto a, auto b) {
        return std::chrono::duration<float, std::micro>(b - a).count();
      };
      last_profile.propagate_us = us(tp0, tp1);
      last_profile.bio_inject_us = us(tp1, tp2);
      last_profile.neuron_step_us = us(tp2, tp3);
      last_profile.validation_us = us(tp3, tp4);
      last_profile.shadow_us = us(tp_shadow_start, tp_shadow_end);
      last_profile.actuation_us = us(tp_act_start, tp_act_end);
      last_profile.opsin_us = 0.0f;  // tracked inside actuation
      last_profile.total_us = last_step_us;

      // Exponential moving average
      float alpha = 0.01f;
      mean_profile.propagate_us += alpha * (last_profile.propagate_us - mean_profile.propagate_us);
      mean_profile.bio_inject_us += alpha * (last_profile.bio_inject_us - mean_profile.bio_inject_us);
      mean_profile.neuron_step_us += alpha * (last_profile.neuron_step_us - mean_profile.neuron_step_us);
      mean_profile.validation_us += alpha * (last_profile.validation_us - mean_profile.validation_us);
      mean_profile.shadow_us += alpha * (last_profile.shadow_us - mean_profile.shadow_us);
      mean_profile.actuation_us += alpha * (last_profile.actuation_us - mean_profile.actuation_us);
      mean_profile.total_us += alpha * (last_profile.total_us - mean_profile.total_us);
    }

    float deadline_us = dt_ms * 1000.0f;
    if (last_step_us > deadline_us) {
      deadline_misses++;
      if (deadline_misses <= 10 || deadline_misses % 1000 == 0) {
        Log(LogLevel::kWarn, "Step %d exceeded deadline: %.0fus > %.0fus (%d total misses)",
            total_steps, last_step_us, deadline_us, deadline_misses);
      }
    }
  }

  // Cached bio readings for fast ticks between medium ticks
  std::vector<BioReading> last_bio_readings_;

  // Run the bridge for a duration
  void Run(float duration_ms, int metrics_interval = 1000) {
    int n_steps = static_cast<int>(duration_ms / dt_ms);
    for (int s = 0; s < n_steps; ++s) {
      Step();
      if (s % metrics_interval == 0) {
        int spikes = digital.CountSpikes();
        if (mode == BridgeMode::kOpenLoop) {
          Log(LogLevel::kInfo, "t=%.1fms  spikes=%d  mode=%s",
              sim_time_ms, spikes, ModeStr());
        } else if (!shadow.history.empty()) {
          Log(LogLevel::kInfo,
              "t=%.1fms  spikes=%d  mode=%s  corr=%.3f  rmse=%.3f  resyncs=%d",
              sim_time_ms, spikes, ModeStr(),
              shadow.history.back().spike_correlation,
              shadow.history.back().population_rmse,
              total_resyncs);
        }
        if (mode == BridgeMode::kClosedLoop) {
          Log(LogLevel::kInfo, "  replaced=%.1f%%  step_us=%.0f/%.0f",
              replacer.ReplacementFraction() * 100.0f,
              mean_step_us, max_step_us);
        }
      }
    }
  }

  // Get validation results (call after Run completes)
  PopulationValidation GetValidationResults() const {
    if (!enable_validation || sim_spike_trains.empty()) {
      return {};
    }
    return validator.ValidatePopulation(sim_spike_trains, bio_spike_trains,
                                        sim_time_ms);
  }

  const char* ModeStr() const {
    switch (mode) {
      case BridgeMode::kOpenLoop: return "open-loop";
      case BridgeMode::kShadow: return "shadow";
      case BridgeMode::kClosedLoop: return "closed-loop";
    }
    return "unknown";
  }
};

}  // namespace mechabrain

#endif  // FWMC_TWIN_BRIDGE_H_
