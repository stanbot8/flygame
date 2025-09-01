#ifndef FWMC_OPTOGENETICS_H_
#define FWMC_OPTOGENETICS_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "bridge/opsin_model.h"
#include "bridge/optogenetic_writer.h"
#include "bridge/light_model.h"
#include "core/brain_spec_loader.h"
#include "core/cell_types.h"
#include "core/izhikevich.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"
#include "core/recorder.h"
#include "core/species.h"
#include "core/stdp.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Configuration for an optogenetics experiment.
struct OptogeneticsConfig {
  // Brain to stimulate (loaded from .brain spec file, or built inline)
  std::string brain_spec_path;

  // Opsin types expressed in the target population
  OpsinType excitatory_opsin = OpsinType::kChRmine;
  OpsinType inhibitory_opsin = OpsinType::kstGtACR2;

  // Laser/optical parameters
  float laser_power_mw = 10.0f;    // per-spot power
  float objective_na = 1.0f;       // numerical aperture
  float wavelength_nm = 590.0f;    // excitation wavelength (ChRmine default)

  // Target selection
  std::string target_region;        // region name to photostimulate
  float target_fraction = 0.2f;     // fraction of region neurons expressing opsin
  uint32_t seed = 42;

  // Stimulus protocol
  float baseline_ms = 500.0f;       // pre-stim baseline recording
  float stim_on_ms = 200.0f;        // light-on duration
  float stim_off_ms = 300.0f;       // inter-pulse interval
  int n_pulses = 5;                  // number of light pulses
  float post_stim_ms = 500.0f;      // post-stim recording

  // Simulation
  float dt_ms = 0.5f;
  float weight_scale = 1.0f;
  bool enable_stdp = false;

  // Output
  std::string output_dir = "opto_results";
  bool record_spikes = true;
  bool record_voltages = false;

  // Computed total duration
  float TotalDuration() const {
    return baseline_ms + n_pulses * (stim_on_ms + stim_off_ms) + post_stim_ms;
  }
};

// Results from an optogenetics experiment.
struct OptogeneticsResult {
  // Per-pulse response (spike count in target region during stim window)
  std::vector<int> pulse_spike_counts;

  // Baseline firing rate (Hz) in target region
  float baseline_rate_hz = 0.0f;

  // Evoked firing rate (Hz) in target region during stimulation
  float evoked_rate_hz = 0.0f;

  // Modulation index: (evoked - baseline) / (evoked + baseline)
  // +1 = pure excitation, -1 = pure silencing, 0 = no effect
  float modulation_index = 0.0f;

  // Off-target effects: firing rate change in non-targeted regions
  float off_target_rate_change_hz = 0.0f;

  // Network-level effect: total spike count change
  float network_modulation = 0.0f;

  // Opsin desensitization: open fraction at end of last pulse
  float final_open_fraction = 0.0f;

  // Total stimulation commands generated
  int total_commands = 0;

  // Safety: max cumulative energy per neuron
  float max_energy = 0.0f;

  // Timing
  double elapsed_seconds = 0.0;

  bool has_effect() const { return std::abs(modulation_index) > 0.05f; }
};

// Self-contained optogenetics experiment on a parametrically generated brain.
//
// Simulates two-photon holographic optogenetics (Packer et al. 2015):
//   1. Build brain from spec (any species)
//   2. Express opsins in target neurons
//   3. Simulate light delivery with tissue optics
//   4. Run pulsed photostimulation protocol
//   5. Record neural responses and quantify effects
//
// This is the core building block for training AI to design optogenetics
// experiments: the digital twin runs the experiment in silico, the AI
// observes the outcome, and iterates on stimulation parameters.
struct OptogeneticsExperiment {
  OptogeneticsConfig config;
  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  OptogeneticWriter writer;

  // Target neuron indices (expressing opsin)
  std::vector<uint32_t> target_neurons;

  // Run the experiment. Returns results.
  OptogeneticsResult Run() {
    return Run(config);
  }

  OptogeneticsResult Run(const OptogeneticsConfig& cfg) {
    config = cfg;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Load or build brain
    BrainSpec spec;
    if (!cfg.brain_spec_path.empty()) {
      auto result = BrainSpecLoader::Load(cfg.brain_spec_path);
      if (!result) {
        Log(LogLevel::kError, "OptogeneticsExperiment: %s",
            result.error().message.c_str());
        return {};
      }
      spec = std::move(*result);
    } else {
      // Default: mouse cortical column (most common optogenetics target)
      spec.name = "opto_default";
      spec.species = Species::kMouse;
      spec.seed = cfg.seed;
      spec.global_weight_mean = 0.8f;
      spec.global_weight_std = 0.3f;
      spec.background_current_mean = 5.0f;
      spec.background_current_std = 2.0f;

      RegionSpec l23;
      l23.name = "L23";
      l23.n_neurons = 800;
      l23.internal_density = 0.04f;
      l23.default_nt = kGlut;
      l23.cell_types = {{CellType::kL23_Pyramidal, 0.8f},
                        {CellType::kPV_Basket, 0.12f},
                        {CellType::kSST_Martinotti, 0.08f}};
      spec.regions.push_back(l23);

      RegionSpec l5;
      l5.name = "L5";
      l5.n_neurons = 500;
      l5.internal_density = 0.04f;
      l5.default_nt = kGlut;
      l5.cell_types = {{CellType::kL5_Pyramidal, 0.8f},
                        {CellType::kPV_Basket, 0.1f},
                        {CellType::kSST_Martinotti, 0.1f}};
      spec.regions.push_back(l5);

      ProjectionSpec l23_to_l5;
      l23_to_l5.from_region = "L23";
      l23_to_l5.to_region = "L5";
      l23_to_l5.density = 0.03f;
      l23_to_l5.nt_type = kGlut;
      l23_to_l5.weight_mean = 1.2f;
      l23_to_l5.weight_std = 0.3f;
      spec.projections.push_back(l23_to_l5);
    }

    auto species_defaults = spec.GetDefaults();
    Log(LogLevel::kInfo, "Opto experiment: %s (species=%s, %zu regions)",
        spec.name.c_str(), SpeciesName(spec.species), spec.regions.size());

    // 2. Generate the brain
    uint32_t total = gen.Generate(spec, neurons, synapses, types);
    synapses.AssignPerNeuronTau(neurons);

    // 3. Select target neurons (opsin-expressing population)
    target_neurons.clear();
    std::string target_region = cfg.target_region;
    if (target_region.empty() && !spec.regions.empty()) {
      target_region = spec.regions[0].name;
    }

    int region_idx = -1;
    for (size_t i = 0; i < gen.region_ranges.size(); ++i) {
      if (gen.region_ranges[i].name == target_region) {
        region_idx = static_cast<int>(i);
        break;
      }
    }

    uint32_t target_start = 0, target_end = total;
    if (region_idx >= 0) {
      target_start = gen.region_ranges[region_idx].start;
      target_end = gen.region_ranges[region_idx].end;
    }

    // Randomly select fraction of target region neurons
    std::mt19937 rng(cfg.seed + 1);
    uint32_t n_target_region = target_end - target_start;
    uint32_t n_expressing = static_cast<uint32_t>(
        std::round(cfg.target_fraction * n_target_region));
    n_expressing = std::min(n_expressing, n_target_region);

    std::vector<uint32_t> candidates;
    for (uint32_t i = target_start; i < target_end; ++i) {
      candidates.push_back(i);
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    target_neurons.assign(candidates.begin(),
                          candidates.begin() + n_expressing);
    std::sort(target_neurons.begin(), target_neurons.end());

    Log(LogLevel::kInfo, "Target: %s (%u neurons, %u expressing opsin)",
        target_region.c_str(), n_target_region, n_expressing);

    // 4. Initialize optogenetic writer with opsin and light models
    writer.InitSafety(total);
    writer.InitOpsinModel(total, cfg.excitatory_opsin, cfg.inhibitory_opsin);
    writer.InitLightModel(cfg.laser_power_mw, cfg.objective_na);
    writer.light.tissue = TissueParamsForWavelength(cfg.wavelength_nm);

    // Set up target mapping (digital idx -> bio target idx, same space here)
    writer.target_map.clear();
    for (uint32_t idx : target_neurons) {
      OptogeneticWriter::TargetMapping m;
      m.digital_idx = idx;
      m.bio_target_idx = idx;
      m.has_excitatory = true;
      m.has_inhibitory = false;
      writer.target_map.push_back(m);
    }

    // 5. Run simulation
    float duration = cfg.TotalDuration();
    int n_steps = static_cast<int>(duration / cfg.dt_ms);

    // Background noise
    std::normal_distribution<float> noise_dist(
        spec.background_current_mean, spec.background_current_std);

    // STDP (optional)
    STDPParams stdp_params;
    stdp_params.a_plus = species_defaults.stdp_a_plus;
    stdp_params.a_minus = species_defaults.stdp_a_minus;
    stdp_params.tau_plus = species_defaults.stdp_tau_plus;
    stdp_params.tau_minus = species_defaults.stdp_tau_minus;

    // Tracking
    OptogeneticsResult result;
    int baseline_spikes_target = 0;
    int baseline_spikes_other = 0;
    float baseline_steps = 0;
    int evoked_spikes_target = 0;
    int evoked_spikes_other = 0;
    float evoked_steps = 0;
    int current_pulse = 0;
    result.pulse_spike_counts.resize(cfg.n_pulses, 0);

    float sim_time = 0.0f;
    for (int step = 0; step < n_steps; ++step) {
      sim_time = step * cfg.dt_ms;

      // Clear external input
      neurons.ClearExternalInput();

      // Background noise
      if (spec.background_current_mean != 0.0f ||
          spec.background_current_std != 0.0f) {
        for (size_t i = 0; i < neurons.n; ++i) {
          neurons.i_ext[i] += noise_dist(rng);
        }
      }

      // Determine if light is on (pulsed protocol)
      bool light_on = false;
      float t_relative = sim_time - cfg.baseline_ms;
      if (t_relative >= 0.0f) {
        float pulse_period = cfg.stim_on_ms + cfg.stim_off_ms;
        int pulse_idx = static_cast<int>(t_relative / pulse_period);
        float within_pulse = t_relative - pulse_idx * pulse_period;
        if (pulse_idx < cfg.n_pulses && within_pulse < cfg.stim_on_ms) {
          light_on = true;
          current_pulse = pulse_idx;
        }
      }

      // Optogenetic stimulation
      if (light_on) {
        // Generate stim commands for expressing neurons that are spiking
        // or need to be activated
        std::vector<StimCommand> commands;
        for (uint32_t idx : target_neurons) {
          StimCommand cmd;
          cmd.neuron_idx = idx;
          cmd.intensity = 0.8f;  // 80% max power
          cmd.excitatory = 1;
          cmd.duration_ms = cfg.dt_ms;
          commands.push_back(cmd);
        }

        // Apply opsin kinetics (photocurrent injection)
        writer.ApplyOpsinStep(commands, neurons, cfg.dt_ms);
        result.total_commands += static_cast<int>(commands.size());
      } else {
        // No light: step opsin recovery (desensitized -> closed)
        std::vector<StimCommand> empty;
        writer.ApplyOpsinStep(empty, neurons, cfg.dt_ms);
      }

      // Synaptic decay + propagation
      float tau = species_defaults.tau_syn_excitatory;
      neurons.DecaySynapticInput(cfg.dt_ms, tau);
      synapses.PropagateSpikes(neurons.spiked.data(),
                               neurons.i_syn.data(),
                               cfg.weight_scale);

      // Izhikevich step
      IzhikevichStepHeterogeneous(neurons, cfg.dt_ms, sim_time, types);

      // STDP
      if (cfg.enable_stdp) {
        STDPUpdate(synapses, neurons, sim_time, stdp_params);
      }

      // Track spikes
      bool is_baseline = (sim_time < cfg.baseline_ms);
      bool is_evoked = light_on;

      for (uint32_t idx : target_neurons) {
        if (neurons.spiked[idx]) {
          if (is_baseline) baseline_spikes_target++;
          if (is_evoked) {
            evoked_spikes_target++;
            result.pulse_spike_counts[current_pulse]++;
          }
        }
      }

      if (is_baseline || is_evoked) {
        for (uint32_t i = 0; i < total; ++i) {
          bool is_target = std::binary_search(target_neurons.begin(),
                                              target_neurons.end(), i);
          if (!is_target && neurons.spiked[i]) {
            if (is_baseline) baseline_spikes_other++;
            if (is_evoked) evoked_spikes_other++;
          }
        }
        if (is_baseline) baseline_steps++;
      }
      if (is_evoked) evoked_steps++;
    }

    // 6. Compute results
    float n_target = static_cast<float>(target_neurons.size());
    float n_other = static_cast<float>(total - target_neurons.size());
    float baseline_time_s = baseline_steps * cfg.dt_ms / 1000.0f;
    float evoked_time_s = evoked_steps * cfg.dt_ms / 1000.0f;

    if (n_target > 0 && baseline_time_s > 0) {
      result.baseline_rate_hz = baseline_spikes_target / (n_target * baseline_time_s);
    }
    if (n_target > 0 && evoked_time_s > 0) {
      result.evoked_rate_hz = evoked_spikes_target / (n_target * evoked_time_s);
    }

    float sum_rates = result.evoked_rate_hz + result.baseline_rate_hz;
    if (sum_rates > 0) {
      result.modulation_index =
          (result.evoked_rate_hz - result.baseline_rate_hz) / sum_rates;
    }

    // Off-target rate change: difference between evoked and baseline in non-target neurons
    if (n_other > 0 && baseline_time_s > 0 && evoked_time_s > 0) {
      float baseline_other_rate = baseline_spikes_other / (n_other * baseline_time_s);
      float evoked_other_rate = evoked_spikes_other / (n_other * evoked_time_s);
      result.off_target_rate_change_hz = evoked_other_rate - baseline_other_rate;
    }

    // Network-level modulation: total rate change (target + off-target)
    if (baseline_time_s > 0 && evoked_time_s > 0) {
      float total_n = static_cast<float>(total);
      float baseline_total = (baseline_spikes_target + baseline_spikes_other)
                             / (total_n * baseline_time_s);
      float evoked_total = (evoked_spikes_target + evoked_spikes_other)
                           / (total_n * evoked_time_s);
      float sum = baseline_total + evoked_total;
      result.network_modulation = (sum > 0) ?
          (evoked_total - baseline_total) / sum : 0.0f;
    }

    // Opsin state
    if (!target_neurons.empty()) {
      result.final_open_fraction =
          writer.excitatory_opsin.OpenFraction(target_neurons[0]);
    }

    // Safety
    result.max_energy = 0.0f;
    for (auto e : writer.cumulative_energy) {
      result.max_energy = std::max(result.max_energy, e);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

    Log(LogLevel::kInfo, "Opto result: baseline=%.1f Hz, evoked=%.1f Hz, "
        "modulation=%.3f, commands=%d",
        result.baseline_rate_hz, result.evoked_rate_hz,
        result.modulation_index, result.total_commands);

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_OPTOGENETICS_H_
