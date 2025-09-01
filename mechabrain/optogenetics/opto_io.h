#ifndef FWMC_OPTO_IO_H_
#define FWMC_OPTO_IO_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "optogenetics.h"
#include "optimizer.h"

namespace mechabrain {

// Write OptogeneticsResult to a simple key=value text file.
// Format is easily parsed by Python, shell scripts, or any language.
inline bool WriteOptoResult(const std::string& path,
                            const OptogeneticsResult& r) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;

  fprintf(f, "baseline_rate_hz=%.6f\n", r.baseline_rate_hz);
  fprintf(f, "evoked_rate_hz=%.6f\n", r.evoked_rate_hz);
  fprintf(f, "modulation_index=%.6f\n", r.modulation_index);
  fprintf(f, "off_target_rate_change_hz=%.6f\n", r.off_target_rate_change_hz);
  fprintf(f, "network_modulation=%.6f\n", r.network_modulation);
  fprintf(f, "final_open_fraction=%.6f\n", r.final_open_fraction);
  fprintf(f, "total_commands=%d\n", r.total_commands);
  fprintf(f, "max_energy=%.6f\n", r.max_energy);
  fprintf(f, "elapsed_seconds=%.6f\n", r.elapsed_seconds);
  fprintf(f, "has_effect=%d\n", r.has_effect() ? 1 : 0);
  fprintf(f, "n_pulses=%zu\n", r.pulse_spike_counts.size());

  for (size_t i = 0; i < r.pulse_spike_counts.size(); ++i) {
    fprintf(f, "pulse_%zu_spikes=%d\n", i, r.pulse_spike_counts[i]);
  }

  fclose(f);
  return true;
}

// Read OptogeneticsResult from key=value text file.
inline bool ReadOptoResult(const std::string& path, OptogeneticsResult& r) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;

  char line[256];
  int n_pulses = 0;

  try {
    while (fgets(line, sizeof(line), f)) {
      char key[128];
      char val[128];
      if (sscanf(line, "%127[^=]=%127s", key, val) != 2) continue;

      if (strcmp(key, "baseline_rate_hz") == 0) r.baseline_rate_hz = std::stof(val);
      else if (strcmp(key, "evoked_rate_hz") == 0) r.evoked_rate_hz = std::stof(val);
      else if (strcmp(key, "modulation_index") == 0) r.modulation_index = std::stof(val);
      else if (strcmp(key, "off_target_rate_change_hz") == 0) r.off_target_rate_change_hz = std::stof(val);
      else if (strcmp(key, "network_modulation") == 0) r.network_modulation = std::stof(val);
      else if (strcmp(key, "final_open_fraction") == 0) r.final_open_fraction = std::stof(val);
      else if (strcmp(key, "total_commands") == 0) r.total_commands = std::stoi(val);
      else if (strcmp(key, "max_energy") == 0) r.max_energy = std::stof(val);
      else if (strcmp(key, "elapsed_seconds") == 0) r.elapsed_seconds = std::stod(val);
      else if (strcmp(key, "n_pulses") == 0) {
        n_pulses = std::stoi(val);
        r.pulse_spike_counts.resize(n_pulses, 0);
      }
      else if (strncmp(key, "pulse_", 6) == 0) {
        int idx = 0;
        if (sscanf(key, "pulse_%d_spikes", &idx) == 1 &&
            idx >= 0 && idx < n_pulses) {
          r.pulse_spike_counts[idx] = std::stoi(val);
        }
      }
    }
  } catch (const std::exception&) {
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}

// Write OptogeneticsConfig to key=value text file.
inline bool WriteOptoConfig(const std::string& path,
                            const OptogeneticsConfig& c) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;

  if (!c.brain_spec_path.empty())
    fprintf(f, "brain_spec_path=%s\n", c.brain_spec_path.c_str());
  fprintf(f, "excitatory_opsin=%d\n", static_cast<int>(c.excitatory_opsin));
  fprintf(f, "inhibitory_opsin=%d\n", static_cast<int>(c.inhibitory_opsin));
  fprintf(f, "laser_power_mw=%.4f\n", c.laser_power_mw);
  fprintf(f, "objective_na=%.4f\n", c.objective_na);
  fprintf(f, "wavelength_nm=%.4f\n", c.wavelength_nm);
  if (!c.target_region.empty())
    fprintf(f, "target_region=%s\n", c.target_region.c_str());
  fprintf(f, "target_fraction=%.4f\n", c.target_fraction);
  fprintf(f, "seed=%u\n", c.seed);
  fprintf(f, "baseline_ms=%.4f\n", c.baseline_ms);
  fprintf(f, "stim_on_ms=%.4f\n", c.stim_on_ms);
  fprintf(f, "stim_off_ms=%.4f\n", c.stim_off_ms);
  fprintf(f, "n_pulses=%d\n", c.n_pulses);
  fprintf(f, "post_stim_ms=%.4f\n", c.post_stim_ms);
  fprintf(f, "dt_ms=%.4f\n", c.dt_ms);
  fprintf(f, "weight_scale=%.4f\n", c.weight_scale);
  fprintf(f, "enable_stdp=%d\n", c.enable_stdp ? 1 : 0);

  fclose(f);
  return true;
}

// Read OptogeneticsConfig from key=value text file.
inline bool ReadOptoConfig(const std::string& path, OptogeneticsConfig& c) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;

  char line[512];
  try {
    while (fgets(line, sizeof(line), f)) {
      char key[128];
      char val[384];
      if (sscanf(line, "%127[^=]=%383[^\n]", key, val) != 2) continue;

      if (strcmp(key, "brain_spec_path") == 0) c.brain_spec_path = val;
      else if (strcmp(key, "excitatory_opsin") == 0) c.excitatory_opsin = static_cast<OpsinType>(std::stoi(val));
      else if (strcmp(key, "inhibitory_opsin") == 0) c.inhibitory_opsin = static_cast<OpsinType>(std::stoi(val));
      else if (strcmp(key, "laser_power_mw") == 0) c.laser_power_mw = std::stof(val);
      else if (strcmp(key, "objective_na") == 0) c.objective_na = std::stof(val);
      else if (strcmp(key, "wavelength_nm") == 0) c.wavelength_nm = std::stof(val);
      else if (strcmp(key, "target_region") == 0) c.target_region = val;
      else if (strcmp(key, "target_fraction") == 0) c.target_fraction = std::stof(val);
      else if (strcmp(key, "seed") == 0) c.seed = static_cast<uint32_t>(std::stoul(val));
      else if (strcmp(key, "baseline_ms") == 0) c.baseline_ms = std::stof(val);
      else if (strcmp(key, "stim_on_ms") == 0) c.stim_on_ms = std::stof(val);
      else if (strcmp(key, "stim_off_ms") == 0) c.stim_off_ms = std::stof(val);
      else if (strcmp(key, "n_pulses") == 0) c.n_pulses = std::stoi(val);
      else if (strcmp(key, "post_stim_ms") == 0) c.post_stim_ms = std::stof(val);
      else if (strcmp(key, "dt_ms") == 0) c.dt_ms = std::stof(val);
      else if (strcmp(key, "weight_scale") == 0) c.weight_scale = std::stof(val);
      else if (strcmp(key, "enable_stdp") == 0) c.enable_stdp = (std::stoi(val) != 0);
    }
  } catch (const std::exception&) {
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}

// Write optimizer result summary (best params + all trial scores).
inline bool WriteOptimizerResult(const std::string& path,
                                 const std::vector<OptoParam>& space,
                                 const OptimizerResult& r) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;

  fprintf(f, "[best]\n");
  fprintf(f, "score=%.6f\n", r.best.score);
  for (size_t i = 0; i < space.size() && i < r.best.params.size(); ++i) {
    fprintf(f, "%s=%.6f\n", space[i].name.c_str(), r.best.params[i]);
  }
  fprintf(f, "modulation_index=%.6f\n", r.best.result.modulation_index);
  fprintf(f, "evoked_rate_hz=%.6f\n", r.best.result.evoked_rate_hz);
  fprintf(f, "baseline_rate_hz=%.6f\n", r.best.result.baseline_rate_hz);
  fprintf(f, "max_energy=%.6f\n", r.best.result.max_energy);
  fprintf(f, "\n[summary]\n");
  fprintf(f, "n_trials=%d\n", r.n_trials);
  fprintf(f, "total_seconds=%.3f\n", r.total_seconds);

  fprintf(f, "\n[trials]\n");
  for (size_t i = 0; i < r.all_trials.size(); ++i) {
    fprintf(f, "%zu=%.6f\n", i, r.all_trials[i].score);
  }

  fclose(f);
  return true;
}

}  // namespace mechabrain

#endif  // FWMC_OPTO_IO_H_
