#ifndef FWMC_COMPENSATED_ABLATION_H_
#define FWMC_COMPENSATED_ABLATION_H_

// Compensated ablation: the core neural prosthesis demonstration.
//
// Compares two degradation curves:
//   A) Pure ablation: progressively kill KC neurons, measure MBON response
//   B) Compensated: kill same neurons, but digital twin fills in for lost ones
//
// The key result: curve B degrades slower than curve A.
// This proves the digital twin extends functional circuit lifetime.
//
// Uses the general-purpose DigitalCompensator from mechabrain/core.
//
// References:
//   Hige et al. 2015 (KC sparse coding)
//   Aso & Rubin 2016 (MB architecture)
//   Gradmann 2023 (neural prosthetics continuity criterion)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/digital_compensator.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"
#include "core/spike_frequency_adaptation.h"
#include "core/stdp.h"
#include "core/synapse_table.h"

namespace mechabrain {

struct CompensatedPoint {
  float ablation_fraction = 0.0f;
  int neurons_silenced = 0;
  int pure_spikes = 0;
  float pure_continuity = 0.0f;
  int compensated_spikes = 0;
  float compensated_continuity = 0.0f;
  float benefit = 0.0f;  // compensated_continuity - pure_continuity
};

struct CompensatedAblationResult {
  int baseline_pre_spikes = 0;
  int baseline_post_spikes = 0;
  float learning_index = 0.0f;

  std::vector<CompensatedPoint> curve;

  float pure_graceful = 0.0f;
  float compensated_graceful = 0.0f;
  float compensation_benefit = 0.0f;
  float pure_half_life = 0.0f;
  float compensated_half_life = 0.0f;
  float lifetime_extension = 0.0f;

  double elapsed_seconds = 0.0;

  bool passed() const {
    return learning_index > 1.5f &&
           compensation_benefit > 0.05f &&
           compensated_graceful > pure_graceful;
  }
};

struct CompensatedAblationExperiment {
  std::vector<float> ablation_fractions = {
    0.0f, 0.10f, 0.20f, 0.30f, 0.40f, 0.50f,
    0.60f, 0.70f, 0.80f, 0.90f
  };

  uint32_t n_orn = 100;
  uint32_t n_pn = 50;
  uint32_t n_kc = 500;
  uint32_t n_mbon = 20;
  uint32_t n_dan = 20;

  int n_training_trials = 5;
  float trial_ms = 500.0f;
  float test_ms = 500.0f;
  float dt_ms = 0.1f;

  float odor_intensity = 15.0f;
  float reward_intensity = 20.0f;
  float background_current = 5.0f;

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

  CompensatedAblationResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    CompensatedAblationResult result;
    std::mt19937 rng(seed);

    // Build circuit
    BrainSpec spec;
    spec.name = "compensated_ablation_mb";
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
    uint32_t r_kc_s = gen.region_ranges[2].start;
    uint32_t r_kc_e = gen.region_ranges[2].end;
    uint32_t r_mbon_s = gen.region_ranges[3].start;
    uint32_t r_mbon_e = gen.region_ranges[3].end;
    uint32_t r_dan_s = gen.region_ranges[4].start;
    uint32_t r_dan_e = gen.region_ranges[4].end;

    synapses.InitEligibilityTraces();

    SpikeFrequencyAdaptation sfa;
    sfa.Init(total);

    // CS+ odor: random 30% of ORNs
    std::vector<uint32_t> cs_plus_orns;
    for (uint32_t i = gen.region_ranges[0].start; i < gen.region_ranges[0].end; ++i) {
      if (std::uniform_real_distribution<float>(0, 1)(rng) < 0.3f)
        cs_plus_orns.push_back(i);
    }

    // Initialize the digital compensator
    DigitalCompensator compensator;
    compensator.Init(neurons, types);

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

    // Apply stimulus to neurons (background + odor)
    auto apply_stimulus = [&]() {
      neurons.ClearExternalInput();
      for (uint32_t i = 0; i < total; ++i)
        neurons.i_ext[i] = background_current;
      for (uint32_t orn : cs_plus_orns)
        neurons.i_ext[orn] += odor_intensity;
    };

    // Run a test trial with optional compensation
    auto run_trial = [&](const std::vector<bool>& silenced,
                         float duration_ms, bool use_compensator) -> int {
      reset_state();
      if (use_compensator) {
        compensator.SetSilenced(silenced);
        compensator.ResetTwin();
      }

      int mbon_spikes = 0;
      int steps = static_cast<int>(duration_ms / dt_ms);
      float sim_time = 0.0f;

      for (int step = 0; step < steps; ++step) {
        apply_stimulus();

        if (use_compensator) {
          // Compensator handles: twin input, combined spike propagation,
          // bio silencing, and synaptic input injection
          compensator.PreStep(neurons, synapses, types, dt_ms);
        } else {
          // Pure ablation: just silence neurons and propagate normally
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
          for (uint32_t i = 0; i < total; ++i) {
            if (silenced[i]) neurons.i_syn[i] = 0.0f;
          }
        }

        sfa.Update(neurons, dt_ms);
        IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);

        if (use_compensator) {
          compensator.PostStep(types, dt_ms, sim_time);
        }

        sim_time += dt_ms;

        for (uint32_t i = r_mbon_s; i < r_mbon_e; ++i) {
          if (neurons.spiked[i]) mbon_spikes++;
        }
      }
      return mbon_spikes;
    };

    // Run a training trial (no ablation, no twin)
    auto run_training = [&](float duration_ms) {
      reset_state();
      int steps = static_cast<int>(duration_ms / dt_ms);
      float sim_time = 0.0f;

      for (int step = 0; step < steps; ++step) {
        apply_stimulus();
        for (uint32_t i = r_dan_s; i < r_dan_e; ++i)
          neurons.i_ext[i] += reward_intensity;

        neurons.DecaySynapticInput(dt_ms, 3.0f);
        synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);

        NeuromodulatorUpdate(neurons, synapses, dt_ms);
        STDPUpdate(synapses, neurons, sim_time, stdp_params);
        if (stdp_params.use_eligibility_traces)
          EligibilityTraceUpdate(synapses, neurons, dt_ms, stdp_params);

        sfa.Update(neurons, dt_ms);
        IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
        sim_time += dt_ms;
      }
    };

    std::vector<bool> no_silence(total, false);

    // Phase 1: Pre-training test
    result.baseline_pre_spikes = run_trial(no_silence, test_ms, false);

    // Phase 2: Training
    for (int t = 0; t < n_training_trials; ++t)
      run_training(trial_ms);

    // Phase 3: Post-training baseline
    result.baseline_post_spikes = run_trial(no_silence, test_ms, false);
    result.learning_index = (result.baseline_pre_spikes > 0) ?
        static_cast<float>(result.baseline_post_spikes) / result.baseline_pre_spikes : 0.0f;

    Log(LogLevel::kInfo, "[compensated] trained: pre=%d post=%d LI=%.2f",
        result.baseline_pre_spikes, result.baseline_post_spikes, result.learning_index);

    // Phase 4: Progressive ablation -- both pure and compensated
    uint32_t n_ablatable = r_kc_e - r_kc_s;
    std::vector<uint32_t> ablation_order;
    for (uint32_t i = r_kc_s; i < r_kc_e; ++i)
      ablation_order.push_back(i);
    std::shuffle(ablation_order.begin(), ablation_order.end(), rng);

    float pure_area = 0.0f, comp_area = 0.0f;
    float prev_frac = 0.0f;
    float prev_pure = 1.0f, prev_comp = 1.0f;

    Log(LogLevel::kInfo, "[compensated] %5s  %8s  %8s  %8s  %8s  %8s",
        "frac", "pure_spk", "pure_cnt", "comp_spk", "comp_cnt", "benefit");

    for (float frac : ablation_fractions) {
      int n_kill = static_cast<int>(frac * n_ablatable);
      std::vector<bool> silenced(total, false);
      for (int i = 0; i < n_kill; ++i)
        silenced[ablation_order[static_cast<size_t>(i)]] = true;

      // A) Pure ablation
      int pure_spikes = run_trial(silenced, test_ms, false);
      float pure_cont = (result.baseline_post_spikes > 0) ?
          static_cast<float>(pure_spikes) / result.baseline_post_spikes : 0.0f;
      pure_cont = std::clamp(pure_cont, 0.0f, 2.0f);

      // B) Compensated ablation (digital twin fills in)
      int comp_spikes = run_trial(silenced, test_ms, true);
      float comp_cont = (result.baseline_post_spikes > 0) ?
          static_cast<float>(comp_spikes) / result.baseline_post_spikes : 0.0f;
      comp_cont = std::clamp(comp_cont, 0.0f, 2.0f);

      CompensatedPoint pt;
      pt.ablation_fraction = frac;
      pt.neurons_silenced = n_kill;
      pt.pure_spikes = pure_spikes;
      pt.pure_continuity = pure_cont;
      pt.compensated_spikes = comp_spikes;
      pt.compensated_continuity = comp_cont;
      pt.benefit = comp_cont - pure_cont;
      result.curve.push_back(pt);

      if (frac > 0.0f) {
        float df = frac - prev_frac;
        pure_area += (prev_pure + pure_cont) * 0.5f * df;
        comp_area += (prev_comp + comp_cont) * 0.5f * df;
      }
      prev_frac = frac;
      prev_pure = pure_cont;
      prev_comp = comp_cont;

      if (result.pure_half_life == 0.0f && pure_cont < 0.5f)
        result.pure_half_life = frac;
      if (result.compensated_half_life == 0.0f && comp_cont < 0.5f)
        result.compensated_half_life = frac;

      Log(LogLevel::kInfo, "[compensated] %4.0f%%  %8d  %8.2f  %8d  %8.2f  %+7.2f",
          frac * 100, pure_spikes, pure_cont, comp_spikes, comp_cont, pt.benefit);
    }

    float max_frac = ablation_fractions.back();
    result.pure_graceful = (max_frac > 0) ? pure_area / max_frac : 0.0f;
    result.compensated_graceful = (max_frac > 0) ? comp_area / max_frac : 0.0f;
    result.compensation_benefit = result.compensated_graceful - result.pure_graceful;

    if (result.pure_half_life > 0.0f && result.compensated_half_life > 0.0f)
      result.lifetime_extension = result.compensated_half_life / result.pure_half_life;
    else if (result.compensated_half_life == 0.0f && result.pure_half_life > 0.0f)
      result.lifetime_extension = 999.0f;

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo, "[compensated] done in %.2fs", result.elapsed_seconds);
    Log(LogLevel::kInfo, "[compensated] pure: graceful=%.2f half_life=%.0f%%",
        result.pure_graceful, result.pure_half_life * 100);
    Log(LogLevel::kInfo, "[compensated] comp: graceful=%.2f half_life=%.0f%%",
        result.compensated_graceful, result.compensated_half_life * 100);
    Log(LogLevel::kInfo, "[compensated] benefit=%.2f lifetime_ext=%.1fx",
        result.compensation_benefit, result.lifetime_extension);

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_COMPENSATED_ABLATION_H_
