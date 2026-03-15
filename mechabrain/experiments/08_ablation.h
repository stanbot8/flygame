#ifndef FWMC_ABLATION_EXPERIMENT_H_
#define FWMC_ABLATION_EXPERIMENT_H_

// Systematic ablation study: how does circuit behavior degrade
// as neurons are progressively silenced?
//
// This answers: "at what percentage of neuron loss does the circuit
// fail to produce the trained behavior?" This calibrates the
// replacement rate needed for neural prosthesis.
//
// Protocol:
//   1. Build and train mushroom body (same as conditioning)
//   2. Measure baseline CS+ response
//   3. Progressively ablate KC neurons (0%, 5%, 10%, ..., 90%)
//   4. After each ablation step, re-measure CS+ response
//   5. Compute degradation curve: continuity vs. ablation fraction
//
// Key readout: graceful degradation curve. Sparse KC coding (Hige et al.
// 2015) predicts gradual degradation with high robustness to partial loss.
//
// References:
//   Aso & Rubin 2016 (MB architecture)
//   Hige et al. 2015 (KC sparse coding robustness)
//   Caron et al. 2013 (random KC connectivity)

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
#include "core/parametric_gen.h"
#include "core/spike_frequency_adaptation.h"
#include "core/stdp.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Results from ablation at a single fraction.
struct AblationPoint {
  float ablation_fraction = 0.0f;
  int neurons_silenced = 0;
  int cs_plus_spikes = 0;
  float continuity = 0.0f;  // cs_plus_spikes / baseline
};

// Results from the full ablation study.
struct AblationResult {
  int baseline_pre_spikes = 0;      // before training
  int baseline_post_spikes = 0;     // after training, no ablation
  float learning_index = 0.0f;

  std::vector<AblationPoint> curve;

  float half_life_fraction = 0.0f;  // ablation % where continuity < 50%
  float cliff_fraction = 0.0f;      // ablation % where continuity < 10%
  float graceful_score = 0.0f;      // area under degradation curve (0-1)

  double elapsed_seconds = 0.0;

  bool passed() const {
    return learning_index > 1.5f && graceful_score > 0.3f;
  }
};

struct AblationExperiment {
  std::vector<float> ablation_fractions = {
    0.0f, 0.05f, 0.10f, 0.20f, 0.30f, 0.40f, 0.50f,
    0.60f, 0.70f, 0.80f, 0.90f
  };

  // Circuit sizes
  uint32_t n_orn = 100;
  uint32_t n_pn = 50;
  uint32_t n_kc = 500;
  uint32_t n_mbon = 20;
  uint32_t n_dan = 20;

  // Training
  int n_training_trials = 5;
  float trial_ms = 500.0f;
  float test_ms = 500.0f;
  float dt_ms = 0.1f;

  // Stimulus
  float odor_intensity = 15.0f;
  float reward_intensity = 20.0f;
  float background_current = 5.0f;

  // STDP parameters (dopamine-gated, matching conditioning)
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

  AblationResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    AblationResult result;
    std::mt19937 rng(seed);

    // Build circuit using BrainSpec + ParametricGenerator
    BrainSpec spec;
    spec.name = "mushroom_body_ablation";
    spec.seed = seed;

    RegionSpec orn_r;
    orn_r.name = "ORN"; orn_r.n_neurons = n_orn;
    orn_r.default_nt = kACh;
    orn_r.cell_types = {{CellType::kORN, 1.0f}};
    spec.regions.push_back(orn_r);

    RegionSpec pn_r;
    pn_r.name = "PN"; pn_r.n_neurons = n_pn;
    pn_r.default_nt = kACh;
    pn_r.cell_types = {{CellType::kPN_excitatory, 1.0f}};
    spec.regions.push_back(pn_r);

    RegionSpec kc_r;
    kc_r.name = "KC"; kc_r.n_neurons = n_kc;
    kc_r.default_nt = kACh;
    kc_r.cell_types = {{CellType::kKenyonCell, 1.0f}};
    spec.regions.push_back(kc_r);

    RegionSpec mbon_r;
    mbon_r.name = "MBON"; mbon_r.n_neurons = n_mbon;
    mbon_r.default_nt = kACh;
    mbon_r.cell_types = {{CellType::kMBON_cholinergic, 1.0f}};
    spec.regions.push_back(mbon_r);

    RegionSpec dan_r;
    dan_r.name = "DAN"; dan_r.n_neurons = n_dan;
    dan_r.default_nt = kDA;
    dan_r.cell_types = {{CellType::kDAN_PPL1, 1.0f}};
    spec.regions.push_back(dan_r);

    spec.projections.push_back({"ORN", "PN", 0.3f, kACh, 2.5f, 0.3f});
    spec.projections.push_back({"PN", "KC", 0.05f, kACh, 2.5f, 0.3f});
    spec.projections.push_back({"KC", "MBON", 0.15f, kACh, 1.0f, 0.2f});
    spec.projections.push_back({"DAN", "MBON", 0.3f, kDA, -2.0f, 0.1f});
    spec.projections.push_back({"MBON", "DAN", 0.2f, kACh, 1.5f, 0.1f});

    NeuronArray neurons;
    SynapseTable synapses;
    CellTypeManager types;
    ParametricGenerator gen;
    gen.Generate(spec, neurons, synapses, types);

    uint32_t total = static_cast<uint32_t>(neurons.n);

    // Extract region ranges
    uint32_t r_kc_s  = gen.region_ranges[2].start;
    uint32_t r_kc_e  = gen.region_ranges[2].end;
    uint32_t r_mbon_s = gen.region_ranges[3].start;
    uint32_t r_mbon_e = gen.region_ranges[3].end;
    uint32_t r_dan_s = gen.region_ranges[4].start;
    uint32_t r_dan_e = gen.region_ranges[4].end;

    // Initialize eligibility traces for three-factor learning
    synapses.InitEligibilityTraces();

    SpikeFrequencyAdaptation sfa;
    sfa.Init(total);

    // CS+ odor: random 30% of ORNs
    std::vector<uint32_t> cs_plus_orns;
    for (uint32_t i = gen.region_ranges[0].start; i < gen.region_ranges[0].end; ++i) {
      if (std::uniform_real_distribution<float>(0, 1)(rng) < 0.3f)
        cs_plus_orns.push_back(i);
    }

    // Reset neuron state to resting potential
    auto reset_state = [&]() {
      for (uint32_t i = 0; i < total; ++i) {
        neurons.v[i] = -65.0f;
        neurons.u[i] = -13.0f;
        neurons.i_syn[i] = 0.0f;
        neurons.i_ext[i] = 0.0f;
        neurons.spiked[i] = 0;
        neurons.dopamine[i] = 0.0f;
        neurons.last_spike_time[i] = -1e9f;
      }
    };

    // Helper: run one trial
    auto run_trial = [&](bool training, const std::vector<bool>& silenced,
                         float duration_ms) -> int {
      reset_state();

      int mbon_spikes = 0;
      int steps = static_cast<int>(duration_ms / dt_ms);
      float sim_time = 0.0f;

      for (int step = 0; step < steps; ++step) {
        neurons.ClearExternalInput();

        // Background drive
        for (uint32_t i = 0; i < total; ++i) {
          neurons.i_ext[i] = background_current;
        }

        // Odor stimulus
        for (uint32_t orn : cs_plus_orns) {
          neurons.i_ext[orn] += odor_intensity;
        }

        // US (dopamine) during training
        if (training) {
          for (uint32_t i = r_dan_s; i < r_dan_e; ++i) {
            neurons.i_ext[i] += reward_intensity;
          }
        }

        // Silence ablated neurons (clamp to rest)
        for (uint32_t i = 0; i < total; ++i) {
          if (silenced[i]) {
            neurons.i_ext[i] = 0.0f;
            neurons.v[i] = -65.0f;
            neurons.u[i] = -13.0f;
            neurons.spiked[i] = 0;
          }
        }

        neurons.DecaySynapticInput(dt_ms, 3.0f);
        synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);

        // Zero synaptic input to silenced neurons
        for (uint32_t i = 0; i < total; ++i) {
          if (silenced[i]) neurons.i_syn[i] = 0.0f;
        }

        if (training) {
          // Neuromodulator dynamics (DAN spikes release dopamine)
          NeuromodulatorUpdate(neurons, synapses, dt_ms);
          // STDP with eligibility traces
          STDPUpdate(synapses, neurons, sim_time, stdp_params);
          if (stdp_params.use_eligibility_traces) {
            EligibilityTraceUpdate(synapses, neurons, dt_ms, stdp_params);
          }
        }

        sfa.Update(neurons, dt_ms);
        IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
        sim_time += dt_ms;

        for (uint32_t i = r_mbon_s; i < r_mbon_e; ++i) {
          if (neurons.spiked[i]) mbon_spikes++;
        }
      }
      return mbon_spikes;
    };

    std::vector<bool> no_silence(total, false);

    // Phase 1: Pre-training test
    result.baseline_pre_spikes = run_trial(false, no_silence, test_ms);

    // Phase 2: Training (CS+ with dopamine)
    for (int t = 0; t < n_training_trials; ++t) {
      (void)run_trial(true, no_silence, trial_ms);
    }

    // Phase 3: Post-training baseline
    result.baseline_post_spikes = run_trial(false, no_silence, test_ms);
    result.learning_index = (result.baseline_pre_spikes > 0) ?
        static_cast<float>(result.baseline_post_spikes) / result.baseline_pre_spikes : 0.0f;

    Log(LogLevel::kInfo, "[ablation] trained: pre=%d post=%d LI=%.2f",
        result.baseline_pre_spikes, result.baseline_post_spikes, result.learning_index);

    // Phase 4: Progressive KC ablation
    uint32_t n_ablatable = r_kc_e - r_kc_s;
    std::vector<uint32_t> ablation_order;
    for (uint32_t i = r_kc_s; i < r_kc_e; ++i)
      ablation_order.push_back(i);
    std::shuffle(ablation_order.begin(), ablation_order.end(), rng);

    float area_sum = 0.0f;
    float prev_frac = 0.0f;
    float prev_cont = 1.0f;

    for (float frac : ablation_fractions) {
      int n_kill = static_cast<int>(frac * n_ablatable);
      std::vector<bool> silenced(total, false);
      for (int i = 0; i < n_kill; ++i) {
        silenced[ablation_order[static_cast<size_t>(i)]] = true;
      }

      int spikes = run_trial(false, silenced, test_ms);
      float cont = (result.baseline_post_spikes > 0) ?
          static_cast<float>(spikes) / result.baseline_post_spikes : 0.0f;
      cont = std::clamp(cont, 0.0f, 2.0f);

      AblationPoint pt;
      pt.ablation_fraction = frac;
      pt.neurons_silenced = n_kill;
      pt.cs_plus_spikes = spikes;
      pt.continuity = cont;
      result.curve.push_back(pt);

      // Trapezoidal area under curve
      if (frac > 0.0f) {
        area_sum += (prev_cont + cont) * 0.5f * (frac - prev_frac);
      }
      prev_frac = frac;
      prev_cont = cont;

      if (result.half_life_fraction == 0.0f && cont < 0.5f)
        result.half_life_fraction = frac;
      if (result.cliff_fraction == 0.0f && cont < 0.1f)
        result.cliff_fraction = frac;

      Log(LogLevel::kInfo, "[ablation] %3.0f%%: silenced=%d spikes=%d continuity=%.2f",
          frac * 100, n_kill, spikes, cont);
    }

    float max_frac = ablation_fractions.back();
    result.graceful_score = (max_frac > 0) ? area_sum / max_frac : 0.0f;

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[ablation] done in %.2fs: graceful=%.2f half_life=%.0f%% cliff=%.0f%%",
        result.elapsed_seconds, result.graceful_score,
        result.half_life_fraction * 100, result.cliff_fraction * 100);

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_ABLATION_EXPERIMENT_H_
