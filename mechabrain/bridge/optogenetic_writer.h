#ifndef FWMC_OPTOGENETIC_WRITER_H_
#define FWMC_OPTOGENETIC_WRITER_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "bridge/bridge_channel.h"
#include "bridge/opsin_model.h"
#include "bridge/light_model.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Converts digital twin spike decisions into stimulation commands
// for holographic two-photon optogenetics (Packer et al., 2015).
//
// Safety model:
//   - Per-neuron refractory period (no re-stimulation within window)
//   - Cumulative energy tracking with thermal limit
//   - Nonlinear power curve mapping (CsChrimson activation)
//   - SLM simultaneous target limit with priority scheduling
struct OptogeneticWriter {
  // Hardware constraints
  size_t max_simultaneous_targets = 50;  // SLM diffraction limit
  float pulse_duration_ms = 1.0f;
  float min_power = 0.1f;   // normalized [0, 1]
  float max_power = 1.0f;
  float galvo_switch_ms = 5.0f;  // time between target groups

  // Safety limits
  float refractory_ms = 5.0f;         // min time between stimulations per neuron
  float max_cumulative_energy = 100.0f; // thermal safety limit (arbitrary units)
  float energy_decay_rate = 0.01f;     // energy dissipation per ms

  // Nonlinear power curve parameters (Hill equation for CsChrimson)
  // P_effective = P_max * (V/V_half)^n / (1 + (V/V_half)^n)
  float v_half = 15.0f;  // voltage at 50% activation (mV above rest)
  float hill_n = 2.0f;   // Hill coefficient (cooperativity)

  struct TargetMapping {
    uint32_t digital_idx;
    uint32_t bio_target_idx;
    bool has_excitatory;
    bool has_inhibitory;
  };

  std::vector<TargetMapping> target_map;

  // Per-neuron safety state
  std::vector<float> last_stim_time;
  std::vector<float> cumulative_energy;

  void InitSafety(size_t n_neurons) {
    last_stim_time.assign(n_neurons, -1e9f);
    cumulative_energy.assign(n_neurons, 0.0f);
  }

  // Nonlinear opsin activation curve
  float PowerCurve(float voltage) const {
    float v_rel = std::max(0.0f, voltage + 65.0f);  // relative to rest (-65mV)
    float ratio = v_rel / v_half;
    float ratio_n = std::pow(ratio, hill_n);
    float activation = ratio_n / (1.0f + ratio_n);
    return std::clamp(activation, min_power, max_power);
  }

  float last_command_time_ = 0.0f;

  std::vector<StimCommand> GenerateCommands(
      const NeuronArray& digital,
      const std::vector<BioReading>& bio_state,
      float sim_time_ms = 0.0f) {

    std::vector<StimCommand> commands;

    // Decay cumulative energy based on actual elapsed time
    float elapsed = sim_time_ms - last_command_time_;
    if (elapsed < 0.0f) elapsed = pulse_duration_ms;  // fallback for first call
    for (auto& e : cumulative_energy) {
      e = std::max(0.0f, e - energy_decay_rate * elapsed);
    }
    last_command_time_ = sim_time_ms;

    for (const auto& m : target_map) {
      if (m.digital_idx >= digital.n) continue;

      // Check refractory period
      bool in_refractory = false;
      if (!last_stim_time.empty() && m.digital_idx < last_stim_time.size()) {
        in_refractory = (sim_time_ms - last_stim_time[m.digital_idx]) < refractory_ms;
      }

      // Check thermal safety
      bool over_thermal = false;
      if (!cumulative_energy.empty() && m.digital_idx < cumulative_energy.size()) {
        over_thermal = cumulative_energy[m.digital_idx] >= max_cumulative_energy;
      }

      if (in_refractory || over_thermal) continue;

      // Excitatory: digital twin says fire
      if (digital.spiked[m.digital_idx] && m.has_excitatory) {
        StimCommand cmd;
        cmd.neuron_idx = m.bio_target_idx;
        cmd.intensity = PowerCurve(digital.v[m.digital_idx]);
        cmd.excitatory = true;
        cmd.duration_ms = pulse_duration_ms;
        commands.push_back(cmd);
      }

      // Inhibitory: suppress biological activity when digital says silent
      if (!digital.spiked[m.digital_idx] && m.has_inhibitory) {
        for (const auto& bio : bio_state) {
          if (bio.neuron_idx == m.bio_target_idx && bio.spike_prob > 0.5f) {
            StimCommand cmd;
            cmd.neuron_idx = m.bio_target_idx;
            cmd.intensity = 0.5f;
            cmd.excitatory = false;
            cmd.duration_ms = pulse_duration_ms;
            commands.push_back(cmd);
            break;
          }
        }
      }
    }

    // Respect SLM limit: prioritize excitatory, then by intensity
    if (commands.size() > max_simultaneous_targets) {
      std::sort(commands.begin(), commands.end(),
          [](const StimCommand& a, const StimCommand& b) {
            if (a.excitatory != b.excitatory) return a.excitatory > b.excitatory;
            return a.intensity > b.intensity;
          });
      commands.resize(max_simultaneous_targets);
    }

    // Update safety state for commands that will be sent
    for (const auto& cmd : commands) {
      // Find digital index from bio target
      for (const auto& m : target_map) {
        if (m.bio_target_idx == cmd.neuron_idx &&
            m.digital_idx < last_stim_time.size()) {
          last_stim_time[m.digital_idx] = sim_time_ms;
          cumulative_energy[m.digital_idx] += cmd.intensity * cmd.duration_ms;
          break;
        }
      }
    }

    return commands;
  }

  // Query safety state
  float ThermalLoad(uint32_t idx) const {
    if (idx >= cumulative_energy.size()) return 0.0f;
    return cumulative_energy[idx] / max_cumulative_energy;
  }

  // --- Galvo-SLM hybrid output ---
  // Galvo mirrors can retarget in ~0.1ms (vs ~5ms SLM hologram update).
  // Split stimulation into:
  //   - galvo channel: 1-3 highest-priority neurons (fast retarget)
  //   - slm channel: remaining steady-state neurons (batch hologram)
  struct GalvoSLMSplit {
    std::vector<StimCommand> galvo;  // fast-retarget priority targets
    std::vector<StimCommand> slm;    // batch hologram steady-state
  };

  size_t max_galvo_targets = 3;  // galvo can only address a few spots

  GalvoSLMSplit SplitGalvoSLM(const std::vector<StimCommand>& commands) const {
    GalvoSLMSplit split;
    // Commands are already sorted by priority (excitatory first, then intensity)
    for (size_t i = 0; i < commands.size(); ++i) {
      if (i < max_galvo_targets) {
        split.galvo.push_back(commands[i]);
      } else {
        split.slm.push_back(commands[i]);
      }
    }
    return split;
  }

  // --- Predictive command pre-staging ---
  // Pre-compute the next N hologram patterns during the current step
  // so the SLM can begin phase computation before the actuation tick.
  // Each staged pattern is a set of StimCommands for a predicted future step.
  struct StagedPattern {
    float predicted_time_ms;
    std::vector<StimCommand> commands;
  };

  std::vector<StagedPattern> staged_patterns;
  size_t max_staged = 5;  // look-ahead depth

  // Pre-stage patterns based on predicted spiking from voltage trends.
  // Uses a simple threshold crossing predictor: if v > threshold and rising,
  // predict a spike in the next step.
  void PreStagePatterns(const NeuronArray& digital,
                        const std::vector<BioReading>& bio_state,
                        float sim_time_ms, float dt_ms) {
    staged_patterns.clear();

    // Save safety state before speculative generation (GenerateCommands
    // has side effects on last_stim_time and cumulative_energy)
    auto saved_stim_time = last_stim_time;
    auto saved_energy = cumulative_energy;
    auto saved_cmd_time = last_command_time_;

    if (scratch_predicted_.n != digital.n) {
      scratch_predicted_.Resize(digital.n);
    }

    for (size_t look = 1; look <= max_staged; ++look) {
      float future_time = sim_time_ms + look * dt_ms;
      for (size_t i = 0; i < digital.n; ++i) {
        float dv = (digital.v[i] - (-65.0f)) * 0.1f;
        float v_pred = digital.v[i] + dv * static_cast<float>(look);
        scratch_predicted_.spiked[i] = (v_pred >= 30.0f) ? 1 : 0;
        scratch_predicted_.v[i] = v_pred;
      }
      auto commands = GenerateCommands(scratch_predicted_, bio_state, future_time);
      if (!commands.empty()) {
        staged_patterns.push_back({future_time, std::move(commands)});
      }
    }

    // Restore safety state (speculative commands should not affect real safety)
    last_stim_time = saved_stim_time;
    cumulative_energy = saved_energy;
    last_command_time_ = saved_cmd_time;
  }

  NeuronArray scratch_predicted_;  // reusable scratch buffer for pre-staging

  // --- Opsin kinetics integration ---
  // When enabled, stimulation commands drive a three-state opsin model
  // instead of simple current injection. This produces realistic
  // photocurrent waveforms with desensitization and recovery.
  bool use_opsin_model = false;
  OpsinPopulation excitatory_opsin;  // e.g. ChRmine for excitation
  OpsinPopulation inhibitory_opsin;  // e.g. stGtACR2 for inhibition

  // --- Light model integration ---
  // When enabled, laser power is attenuated by tissue optics before
  // reaching the opsin. This models depth-dependent stimulation efficacy.
  bool use_light_model = false;
  LightModel light;

  void InitOpsinModel(size_t n_neurons,
                      OpsinType excitatory = OpsinType::kChRmine,
                      OpsinType inhibitory = OpsinType::kstGtACR2) {
    use_opsin_model = true;
    excitatory_opsin.Init(n_neurons, excitatory);
    inhibitory_opsin.Init(n_neurons, inhibitory);
  }

  void InitLightModel(float laser_power_mw = 10.0f, float na = 1.0f) {
    use_light_model = true;
    light.laser_power_mw = laser_power_mw;
    light.objective_na = na;
    light.tissue = TissueParamsForWavelength(590.0f);  // default: ChRmine wavelength
  }

  // Apply opsin kinetics: translate StimCommands into irradiance,
  // step the opsin model, and inject photocurrents into i_ext.
  // Call this after GenerateCommands and before the neuron step.
  void ApplyOpsinStep(const std::vector<StimCommand>& commands,
                      NeuronArray& neurons, float dt_ms) {
    if (!use_opsin_model) return;

    excitatory_opsin.ClearIrradiance();
    inhibitory_opsin.ClearIrradiance();

    for (const auto& cmd : commands) {
      // Map bio target index back to digital index for array access
      uint32_t digital_idx = cmd.neuron_idx;  // default: assume same space
      for (const auto& m : target_map) {
        if (m.bio_target_idx == cmd.neuron_idx) {
          digital_idx = m.digital_idx;
          break;
        }
      }
      if (digital_idx >= neurons.n) continue;

      // Convert normalized intensity to irradiance (mW/mm^2)
      float base_irradiance = cmd.intensity * light.laser_power_mw;

      // Apply light model attenuation if enabled
      float irradiance = base_irradiance;
      if (use_light_model) {
        // Convert nm positions to um for the light model
        float x_um = neurons.x[digital_idx] * 0.001f;
        float y_um = neurons.y[digital_idx] * 0.001f;
        float z_um = neurons.z[digital_idx] * 0.001f;

        float lambda = cmd.excitatory
            ? excitatory_opsin.params.lambda_peak_nm
            : inhibitory_opsin.params.lambda_peak_nm;

        // Scale by depth attenuation
        float atten = light.IrradianceAt(x_um, y_um, z_um, lambda);
        float peak = light.laser_power_mw /
            (kPi * std::pow(light.LateralResolution(lambda) * 0.001f, 2.0f));
        if (peak > 0) {
          irradiance = base_irradiance * (atten / peak);
        }
      }

      // Use digital index for opsin arrays (sized by digital neuron count)
      if (cmd.excitatory) {
        excitatory_opsin.SetIrradiance(digital_idx, irradiance);
      } else {
        inhibitory_opsin.SetIrradiance(digital_idx, irradiance);
      }
    }

    // Step both opsin populations
    excitatory_opsin.Step(dt_ms, neurons.v.data(), neurons.i_ext.data(), neurons.n);
    inhibitory_opsin.Step(dt_ms, neurons.v.data(), neurons.i_ext.data(), neurons.n);
  }

  // Retrieve pre-staged pattern closest to the given time
  const StagedPattern* GetStagedPattern(float time_ms,
                                         float tolerance_ms = 0.05f) const {
    for (const auto& sp : staged_patterns) {
      if (std::abs(sp.predicted_time_ms - time_ms) <= tolerance_ms) {
        return &sp;
      }
    }
    return nullptr;
  }
};

}  // namespace mechabrain

#endif  // FWMC_OPTOGENETIC_WRITER_H_
