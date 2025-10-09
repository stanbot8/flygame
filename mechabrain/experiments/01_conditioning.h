#ifndef FWMC_CONDITIONING_EXPERIMENT_H_
#define FWMC_CONDITIONING_EXPERIMENT_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/intrinsic_homeostasis.h"
#include "core/log.h"
#include "core/motor_output.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/spike_frequency_adaptation.h"
#include "core/stdp.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Results from a conditioning experiment: captures MBON responses and
// weight changes to quantify associative learning.
struct ConditioningResult {
  // MBON spike counts during test trials (summed across all MBONs)
  int pre_test_cs_plus_spikes = 0;   // CS+ odor before training
  int pre_test_cs_minus_spikes = 0;  // CS- odor before training
  int post_test_cs_plus_spikes = 0;  // CS+ odor after training
  int post_test_cs_minus_spikes = 0; // CS- odor after training

  // Mean KC->MBON weight change
  float mean_weight_before = 0.0f;
  float mean_weight_after = 0.0f;
  float weight_change_ratio = 0.0f;  // after / before

  // Learning index: (post CS+ - pre CS+) / (pre CS+ + 1)
  // Positive = potentiation, negative = depression
  float learning_index = 0.0f;

  // Discrimination index: difference in CS+ vs CS- response after training
  // Higher = better odor discrimination
  float discrimination_index = 0.0f;

  // Behavioral readout (motor output)
  float pre_approach_cs_plus = 0.0f;   // approach drive before training
  float post_approach_cs_plus = 0.0f;  // approach drive after training
  float post_approach_cs_minus = 0.0f; // approach drive to CS- after training
  float behavioral_learning = 0.0f;    // post CS+ approach - pre CS+ approach

  // Firing rate validation
  int regions_in_range = 0;   // how many regions have plausible rates
  int regions_total = 0;      // total regions measured

  // Timing
  double elapsed_seconds = 0.0;
  int total_training_trials = 0;

  bool learned() const { return std::abs(learning_index) > 0.05f; }
};

// Self-contained olfactory conditioning experiment on a programmatically
// generated mushroom body circuit.
//
// Architecture (Aso & Rubin 2016):
//   ORN (100) -> PN (50) -> KC (500) -> MBON (20)
//                                  ^
//                           DAN (20) dopamine
//
// Protocol:
//   1. Pre-test: present CS+ and CS- odors, measure MBON baseline
//   2. Training: N trials of CS+/US pairing (odor A + DAN reward)
//                N trials of CS- alone (odor B, no reward)
//   3. Post-test: re-present CS+ and CS-, measure MBON change
//
// STDP is dopamine-gated: only KC->MBON synapses active during DAN
// firing undergo lasting weight changes (three-factor learning rule).
struct ConditioningExperiment {
  // Circuit parameters
  uint32_t n_orn = 100;
  uint32_t n_pn = 50;
  uint32_t n_kc = 500;
  uint32_t n_mbon = 20;
  uint32_t n_dan = 20;

  // Connectivity densities
  float orn_pn_density = 0.30f;    // convergent
  float pn_kc_density = 0.05f;     // divergent (sparse coding)
  float kc_mbon_density = 0.10f;   // readout
  float dan_kc_density = 0.15f;    // dopamine modulation
  float kc_internal_density = 0.0f; // KCs are largely non-recurrent

  // Timing (ms)
  float dt_ms = 0.1f;
  float test_duration_ms = 500.0f;     // each test trial
  float trial_duration_ms = 1000.0f;   // each training trial
  float iti_ms = 500.0f;               // inter-trial interval
  int n_training_trials = 5;           // CS+/US pairings

  // Stimulus
  float odor_intensity = 15.0f;  // pA injected to ORNs
  float reward_intensity = 20.0f; // pA injected to DANs
  float background_current = 5.0f; // tonic drive to keep network active

  // STDP parameters
  STDPParams stdp_params = {
    .a_plus = 0.005f,
    .a_minus = 0.006f,
    .tau_plus = 20.0f,
    .tau_minus = 20.0f,
    .w_min = 0.0f,
    .w_max = 10.0f,
    .dopamine_gated = true,
    .da_scale = 5.0f,
    .use_eligibility_traces = true,
    .tau_eligibility_ms = 1000.0f,
  };

  // Internal state (initialized in Run())
  SpikeFrequencyAdaptation sfa_;  // calcium-based sAHP

  uint32_t seed = 42;
  float weight_scale = 1.0f;

  // Run the full conditioning experiment.
  ConditioningResult Run() {
    auto t0 = std::chrono::high_resolution_clock::now();

    Log(LogLevel::kInfo, "=== Olfactory Conditioning Experiment ===");

    // Build the mushroom body circuit
    NeuronArray neurons;
    SynapseTable synapses;
    CellTypeManager types;
    BuildCircuit(neurons, synapses, types);

    // Region index ranges
    uint32_t orn_start = 0, orn_end = n_orn;
    uint32_t pn_start = orn_end, pn_end = pn_start + n_pn;
    uint32_t kc_start = pn_end, kc_end = kc_start + n_kc;
    uint32_t mbon_start = kc_end, mbon_end = mbon_start + n_mbon;
    uint32_t dan_start = mbon_end, dan_end = dan_start + n_dan;
    (void)pn_start; (void)pn_end; (void)dan_end;

    // CS+ targets first half of ORNs, CS- targets second half
    uint32_t cs_plus_start = orn_start;
    uint32_t cs_plus_end = orn_start + n_orn / 2;
    uint32_t cs_minus_start = orn_start + n_orn / 2;
    uint32_t cs_minus_end = orn_end;

    // Initialize eligibility traces for three-factor learning
    synapses.InitEligibilityTraces();

    // Spike-frequency adaptation (calcium-based sAHP)
    sfa_.Init(neurons.n);

    // Synaptic scaling for homeostasis
    SynapticScaling scaling;
    scaling.Init(neurons.n);

    // Intrinsic excitability homeostasis
    IntrinsicHomeostasis homeo;
    homeo.Init(neurons.n, 5.0f, dt_ms);
    homeo.update_interval_ms = 500.0f;  // adjust every 500ms

    // Motor output: MBONs drive approach/avoid behavior
    MotorOutput motor;
    {
      std::vector<uint32_t> approach_idx, avoid_idx;
      for (uint32_t i = mbon_start; i < mbon_end; ++i) {
        // Cholinergic MBONs (type 2) drive approach,
        // GABAergic (type 3) and glutamatergic (type 4) drive avoidance
        if (neurons.type[i] == 2 || neurons.type[i] == 0)
          approach_idx.push_back(i);
        else
          avoid_idx.push_back(i);
      }
      // No descending neurons in this simple circuit; use empty lists
      motor.Init({}, {}, approach_idx, avoid_idx);
    }

    // Rate monitor with region names matching literature references
    RateMonitor rate_mon;
    {
      std::vector<std::string> rnames = {"ORN", "PN", "KC", "MBON", "DAN"};
      rate_mon.Init(neurons, rnames, dt_ms);
    }

    ConditioningResult result;
    result.total_training_trials = n_training_trials;

    // Measure mean KC->MBON weight before training
    result.mean_weight_before = MeanKCtoMBONWeight(synapses, kc_start, kc_end,
                                                    mbon_start, mbon_end);

    Log(LogLevel::kInfo, "Circuit: %zu neurons, %zu synapses",
        neurons.n, synapses.Size());
    Log(LogLevel::kInfo, "Mean KC->MBON weight before training: %.4f",
        result.mean_weight_before);

    // ---- Phase 1: Pre-test ----
    Log(LogLevel::kInfo, "--- Phase 1: Pre-test ---");
    ResetState(neurons);
    result.pre_test_cs_plus_spikes = RunTestTrial(
        neurons, synapses, types, cs_plus_start, cs_plus_end,
        mbon_start, mbon_end, motor, rate_mon, "CS+ pre-test");
    result.pre_approach_cs_plus = motor.Command().approach_drive;

    ResetState(neurons);
    result.pre_test_cs_minus_spikes = RunTestTrial(
        neurons, synapses, types, cs_minus_start, cs_minus_end,
        mbon_start, mbon_end, motor, rate_mon, "CS- pre-test");

    // ---- Phase 2: Training ----
    Log(LogLevel::kInfo, "--- Phase 2: Training (%d trials) ---",
        n_training_trials);
    for (int trial = 0; trial < n_training_trials; ++trial) {
      // CS+ trial: odor A + DAN reward
      ResetState(neurons);
      RunTrainingTrial(neurons, synapses, types, scaling, homeo,
                       cs_plus_start, cs_plus_end,
                       dan_start, dan_end,
                       mbon_start, mbon_end,
                       true, trial);

      // CS- trial: odor B, no reward
      ResetState(neurons);
      RunTrainingTrial(neurons, synapses, types, scaling, homeo,
                       cs_minus_start, cs_minus_end,
                       dan_start, dan_end,
                       mbon_start, mbon_end,
                       false, trial);
    }

    // ---- Phase 3: Post-test ----
    Log(LogLevel::kInfo, "--- Phase 3: Post-test ---");
    ResetState(neurons);
    result.post_test_cs_plus_spikes = RunTestTrial(
        neurons, synapses, types, cs_plus_start, cs_plus_end,
        mbon_start, mbon_end, motor, rate_mon, "CS+ post-test");
    result.post_approach_cs_plus = motor.Command().approach_drive;

    ResetState(neurons);
    result.post_test_cs_minus_spikes = RunTestTrial(
        neurons, synapses, types, cs_minus_start, cs_minus_end,
        mbon_start, mbon_end, motor, rate_mon, "CS- post-test");
    result.post_approach_cs_minus = motor.Command().approach_drive;

    // ---- Compute results ----
    result.mean_weight_after = MeanKCtoMBONWeight(synapses, kc_start, kc_end,
                                                   mbon_start, mbon_end);
    result.weight_change_ratio = (result.mean_weight_before > 0.0f)
        ? result.mean_weight_after / result.mean_weight_before
        : 0.0f;

    // Learning index: change in CS+ MBON response
    result.learning_index = static_cast<float>(
        result.post_test_cs_plus_spikes - result.pre_test_cs_plus_spikes)
        / static_cast<float>(result.pre_test_cs_plus_spikes + 1);

    // Discrimination: CS+ vs CS- difference after training
    result.discrimination_index = static_cast<float>(
        result.post_test_cs_plus_spikes - result.post_test_cs_minus_spikes)
        / static_cast<float>(
            result.post_test_cs_plus_spikes + result.post_test_cs_minus_spikes + 1);

    // Behavioral learning: change in approach drive to CS+
    result.behavioral_learning =
        result.post_approach_cs_plus - result.pre_approach_cs_plus;

    // Final firing rate check
    auto final_rates = rate_mon.ComputeRates();
    result.regions_in_range = RateMonitor::CountInRange(final_rates);
    result.regions_total = static_cast<int>(final_rates.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

    // ---- Report ----
    Log(LogLevel::kInfo, "=== Conditioning Results ===");
    Log(LogLevel::kInfo, "Pre-test:  CS+ MBON spikes=%d  CS-=%d",
        result.pre_test_cs_plus_spikes, result.pre_test_cs_minus_spikes);
    Log(LogLevel::kInfo, "Post-test: CS+ MBON spikes=%d  CS-=%d",
        result.post_test_cs_plus_spikes, result.post_test_cs_minus_spikes);
    Log(LogLevel::kInfo, "KC->MBON weight: %.4f -> %.4f (ratio=%.3f)",
        result.mean_weight_before, result.mean_weight_after,
        result.weight_change_ratio);
    Log(LogLevel::kInfo, "Learning index: %.3f  Discrimination: %.3f",
        result.learning_index, result.discrimination_index);
    Log(LogLevel::kInfo, "Approach drive: CS+ %.3f -> %.3f  CS- %.3f",
        result.pre_approach_cs_plus, result.post_approach_cs_plus,
        result.post_approach_cs_minus);
    Log(LogLevel::kInfo, "Behavioral learning: %.3f", result.behavioral_learning);
    Log(LogLevel::kInfo, "Firing rates: %d/%d regions in biological range",
        result.regions_in_range, result.regions_total);
    RateMonitor::LogRates(final_rates);
    Log(LogLevel::kInfo, "Learned: %s (%.3fs elapsed)",
        result.learned() ? "YES" : "NO", result.elapsed_seconds);

    return result;
  }

 private:
  void BuildCircuit(NeuronArray& neurons, SynapseTable& synapses,
                    CellTypeManager& types) {
    BrainSpec spec;
    spec.name = "mushroom_body_conditioning";
    spec.seed = seed;
    spec.global_weight_mean = 1.0f;
    spec.global_weight_std = 0.2f;

    // Region 0: ORN (olfactory receptor neurons)
    RegionSpec orn_region;
    orn_region.name = "ORN";
    orn_region.n_neurons = n_orn;
    orn_region.internal_density = 0.0f;  // no recurrence
    orn_region.default_nt = kACh;
    orn_region.cell_types = {{CellType::kORN, 1.0f}};
    spec.regions.push_back(orn_region);

    // Region 1: PN (projection neurons in antennal lobe)
    RegionSpec pn_region;
    pn_region.name = "PN";
    pn_region.n_neurons = n_pn;
    pn_region.internal_density = 0.05f;  // lateral inhibition
    pn_region.default_nt = kACh;
    pn_region.cell_types = {{CellType::kPN_excitatory, 0.8f},
                            {CellType::kLN_local, 0.2f}};
    spec.regions.push_back(pn_region);

    // Region 2: KC (Kenyon cells, mushroom body intrinsic)
    RegionSpec kc_region;
    kc_region.name = "KC";
    kc_region.n_neurons = n_kc;
    kc_region.internal_density = kc_internal_density;
    kc_region.default_nt = kACh;
    kc_region.cell_types = {{CellType::kKenyonCell, 1.0f}};
    spec.regions.push_back(kc_region);

    // Region 3: MBON (mushroom body output neurons)
    RegionSpec mbon_region;
    mbon_region.name = "MBON";
    mbon_region.n_neurons = n_mbon;
    mbon_region.internal_density = 0.0f;
    mbon_region.default_nt = kACh;
    mbon_region.cell_types = {{CellType::kMBON_cholinergic, 0.5f},
                              {CellType::kMBON_gabaergic, 0.3f},
                              {CellType::kMBON_glutamatergic, 0.2f}};
    spec.regions.push_back(mbon_region);

    // Region 4: DAN (dopaminergic neurons)
    RegionSpec dan_region;
    dan_region.name = "DAN";
    dan_region.n_neurons = n_dan;
    dan_region.internal_density = 0.0f;
    dan_region.default_nt = kDA;
    dan_region.cell_types = {{CellType::kDAN_PAM, 0.6f},
                             {CellType::kDAN_PPL1, 0.4f}};
    spec.regions.push_back(dan_region);

    // Projections (feedforward pathway)
    spec.projections.push_back({"ORN", "PN", orn_pn_density, kACh, 2.0f, 0.3f});
    spec.projections.push_back({"PN", "KC", pn_kc_density, kACh, 1.5f, 0.3f});
    spec.projections.push_back({"KC", "MBON", kc_mbon_density, kACh, 1.0f, 0.2f});

    // DAN -> KC/MBON: dopaminergic modulation (these synapses deliver DA)
    spec.projections.push_back({"DAN", "KC", dan_kc_density, kDA, 0.5f, 0.1f});
    spec.projections.push_back({"DAN", "MBON", 0.2f, kDA, 0.5f, 0.1f});

    ParametricGenerator gen;
    gen.Generate(spec, neurons, synapses, types);
  }

  void ResetState(NeuronArray& neurons) {
    for (size_t i = 0; i < neurons.n; ++i) {
      neurons.v[i] = -65.0f;
      neurons.u[i] = -13.0f;
      neurons.i_syn[i] = 0.0f;
      neurons.i_ext[i] = 0.0f;
      neurons.spiked[i] = 0;
      neurons.dopamine[i] = 0.0f;
      neurons.serotonin[i] = 0.0f;
      neurons.octopamine[i] = 0.0f;
      neurons.last_spike_time[i] = -1e9f;
    }
  }

  // Run a test trial: inject odor, count MBON spikes, update motor. No STDP.
  int RunTestTrial(NeuronArray& neurons, SynapseTable& synapses,
                   const CellTypeManager& types,
                   uint32_t odor_start, uint32_t odor_end,
                   uint32_t mbon_start, uint32_t mbon_end,
                   MotorOutput& motor, RateMonitor& rate_mon,
                   const char* label) {
    int n_steps = static_cast<int>(test_duration_ms / dt_ms);
    int mbon_spikes = 0;
    float sim_time = 0.0f;

    for (int step = 0; step < n_steps; ++step) {
      neurons.ClearExternalInput();

      // Tonic background drive
      for (size_t i = 0; i < neurons.n; ++i) {
        neurons.i_ext[i] = background_current;
      }

      // Odor stimulus for first 80% of trial
      if (sim_time < test_duration_ms * 0.8f) {
        for (uint32_t i = odor_start; i < odor_end; ++i) {
          neurons.i_ext[i] += odor_intensity;
        }
      }

      // Use synaptic decay instead of clear for realistic PSC waveforms
      neurons.DecaySynapticInput(dt_ms, 3.0f);  // tau=3ms (fast ACh)
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(),
                               weight_scale);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
      sfa_.Update(neurons, dt_ms);

      // Motor output and rate tracking
      motor.Update(neurons, dt_ms);
      rate_mon.RecordStep(neurons);

      sim_time += dt_ms;

      // Count MBON spikes
      for (uint32_t i = mbon_start; i < mbon_end; ++i) {
        mbon_spikes += neurons.spiked[i];
      }
    }

    Log(LogLevel::kInfo, "  %s: MBON spikes=%d  approach=%.3f",
        label, mbon_spikes, motor.Command().approach_drive);
    return mbon_spikes;
  }

  // Run a training trial with STDP, neuromodulation, and eligibility traces.
  void RunTrainingTrial(NeuronArray& neurons, SynapseTable& synapses,
                        const CellTypeManager& types,
                        SynapticScaling& scaling,
                        IntrinsicHomeostasis& homeo,
                        uint32_t odor_start, uint32_t odor_end,
                        uint32_t dan_start, uint32_t dan_end,
                        uint32_t mbon_start, uint32_t mbon_end,
                        bool with_reward, int trial_idx) {
    int n_steps = static_cast<int>(trial_duration_ms / dt_ms);
    float sim_time = 0.0f;
    int mbon_spikes = 0;

    for (int step = 0; step < n_steps; ++step) {
      neurons.ClearExternalInput();

      // Background drive
      for (size_t i = 0; i < neurons.n; ++i) {
        neurons.i_ext[i] = background_current;
      }

      // Odor stimulus (first 80% of trial)
      if (sim_time < trial_duration_ms * 0.8f) {
        for (uint32_t i = odor_start; i < odor_end; ++i) {
          neurons.i_ext[i] += odor_intensity;
        }
      }

      // Reward: DAN activation, delayed 200ms after trial start,
      // lasting until 60% through the trial
      if (with_reward && sim_time >= 200.0f &&
          sim_time < trial_duration_ms * 0.6f) {
        for (uint32_t i = dan_start; i < dan_end; ++i) {
          neurons.i_ext[i] += reward_intensity;
        }
      }

      // Use synaptic decay instead of clear for realistic PSC waveforms
      neurons.DecaySynapticInput(dt_ms, 3.0f);  // tau=3ms (fast ACh)
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(),
                               weight_scale);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
      sfa_.Update(neurons, dt_ms);

      // Neuromodulator dynamics (DAN spikes release dopamine)
      NeuromodulatorUpdate(neurons, synapses, dt_ms);

      // STDP: spike pairs set eligibility traces
      STDPUpdate(synapses, neurons, sim_time, stdp_params);

      // Three-factor: dopamine converts eligibility traces to weight changes
      if (stdp_params.use_eligibility_traces) {
        EligibilityTraceUpdate(synapses, neurons, dt_ms, stdp_params);
      }

      // Synaptic scaling (every 500 steps)
      scaling.AccumulateSpikes(neurons, dt_ms);
      if (step > 0 && step % 500 == 0) {
        scaling.Apply(synapses, stdp_params);
      }

      // Intrinsic homeostasis (slow excitability adjustment)
      homeo.RecordSpikes(neurons);
      homeo.MaybeApply(neurons);

      sim_time += dt_ms;

      for (uint32_t i = mbon_start; i < mbon_end; ++i) {
        mbon_spikes += neurons.spiked[i];
      }
    }

    Log(LogLevel::kInfo, "  %s trial %d: MBON spikes=%d",
        with_reward ? "CS+" : "CS-", trial_idx + 1, mbon_spikes);
  }

  // Compute mean weight of KC->MBON synapses in the CSR table.
  float MeanKCtoMBONWeight(const SynapseTable& synapses,
                           uint32_t kc_start, uint32_t kc_end,
                           uint32_t mbon_start, uint32_t mbon_end) {
    float sum = 0.0f;
    int count = 0;
    for (uint32_t pre = kc_start; pre < kc_end; ++pre) {
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        uint32_t post = synapses.post[s];
        if (post >= mbon_start && post < mbon_end) {
          sum += synapses.weight[s];
          count++;
        }
      }
    }
    return (count > 0) ? sum / static_cast<float>(count) : 0.0f;
  }
};

}  // namespace mechabrain

#endif  // FWMC_CONDITIONING_EXPERIMENT_H_
