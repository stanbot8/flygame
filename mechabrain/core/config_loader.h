#ifndef FWMC_CONFIG_LOADER_H_
#define FWMC_CONFIG_LOADER_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "core/error.h"
#include "core/experiment_config.h"

namespace mechabrain {

// Minimal JSON-like config parser. No external dependencies.
// Supports the subset needed for experiment configs:
//   key: value pairs (strings, numbers, booleans)
//   arrays of numbers
//   stimulus events
//
// Format example (experiment.cfg):
//   name = phase1_odor_A
//   fly_strain = w1118_GCaMP8f
//   duration_ms = 30000
//   dt_ms = 0.1
//   bridge_mode = 1
//   enable_stdp = true
//   connectome_dir = data/MB_fly01
//   output_dir = results/fly01_odorA
//   recording_input = recordings/fly01_session1.bin
//   monitor_neurons = 0 1 2 3 4 5
//   stimulus: odor_A 500 1500 0.8 0,1,2,3
//   stimulus: shock 2000 2100 1.0 10,11,12
struct ConfigLoader {
  static Result<ExperimentConfig> Load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound, "Cannot open config: " + path);
    }

    ExperimentConfig cfg;
    char line[4096];

    while (fgets(line, sizeof(line), f)) {
      // Skip comments and blank lines
      char* p = line;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '#' || *p == '\n' || *p == '\0') continue;

      // Remove trailing newline
      size_t len = strlen(p);
      while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r')) p[--len] = '\0';

      // Parse stimulus events
      if (strncmp(p, "stimulus:", 9) == 0) {
        StimulusEvent ev;
        if (ParseStimulus(p + 9, ev)) {
          cfg.stimulus_protocol.push_back(std::move(ev));
        }
        continue;
      }

      // Parse key = value
      char* eq = strchr(p, '=');
      if (!eq) continue;

      std::string key(p, eq);
      std::string val(eq + 1);

      // Trim whitespace
      auto trim = [](std::string& s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(0, 1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
      };
      trim(key);
      trim(val);

      // Assign to config fields (with safe parsing)
      try {
        if (key == "name") cfg.name = val;
        else if (key == "fly_strain") cfg.fly_strain = val;
        else if (key == "date") cfg.date = val;
        else if (key == "notes") cfg.notes = val;
        else if (key == "dt_ms") cfg.dt_ms = std::stof(val);
        else if (key == "duration_ms") cfg.duration_ms = std::stof(val);
        else if (key == "weight_scale") cfg.weight_scale = std::stof(val);
        else if (key == "metrics_interval") cfg.metrics_interval = std::stoi(val);
        else if (key == "enable_stdp") cfg.enable_stdp = (val == "true" || val == "1");
        else if (key == "bridge_mode") cfg.bridge_mode = std::stoi(val);
        else if (key == "monitor_threshold") cfg.monitor_threshold = std::stof(val);
        else if (key == "bridge_threshold") cfg.bridge_threshold = std::stof(val);
        else if (key == "resync_threshold") cfg.resync_threshold = std::stof(val);
        else if (key == "min_observation_ms") cfg.min_observation_ms = std::stof(val);
        else if (key == "calibration_interval") cfg.calibration_interval = std::stoi(val);
        else if (key == "calibration_lr") cfg.calibration_lr = std::stof(val);
        else if (key == "connectome_dir") cfg.connectome_dir = val;
        else if (key == "recording_input") cfg.recording_input = val;
        else if (key == "output_dir") cfg.output_dir = val;
        else if (key == "record_spikes") cfg.record_spikes = (val == "true" || val == "1");
        else if (key == "record_voltages") cfg.record_voltages = (val == "true" || val == "1");
        else if (key == "record_shadow_metrics") cfg.record_shadow_metrics = (val == "true" || val == "1");
        else if (key == "record_per_neuron_error") cfg.record_per_neuron_error = (val == "true" || val == "1");
        else if (key == "recording_interval") cfg.recording_interval = std::stoi(val);
        else if (key == "monitor_neurons") {
          cfg.monitor_neurons = ParseUintList(val);
        }
      } catch (const std::exception&) {
        fclose(f);
        return MakeError(ErrorCode::kInvalidParam,
            "Invalid value for '" + key + "': " + val);
      }
    }

    fclose(f);

    // Validate required ranges
    if (cfg.dt_ms <= 0.0f)
      return MakeError(ErrorCode::kInvalidParam, "dt_ms must be > 0");
    if (cfg.duration_ms <= 0.0f)
      return MakeError(ErrorCode::kInvalidParam, "duration_ms must be > 0");
    if (cfg.weight_scale < 0.0f)
      return MakeError(ErrorCode::kInvalidParam, "weight_scale must be >= 0");
    if (cfg.metrics_interval <= 0)
      return MakeError(ErrorCode::kInvalidParam, "metrics_interval must be > 0");
    if (cfg.recording_interval <= 0)
      return MakeError(ErrorCode::kInvalidParam, "recording_interval must be > 0");
    if (cfg.bridge_mode < 0 || cfg.bridge_mode > 2)
      return MakeError(ErrorCode::kInvalidParam, "bridge_mode must be 0, 1, or 2");
    if (cfg.monitor_threshold < 0.0f || cfg.monitor_threshold > 1.0f)
      return MakeError(ErrorCode::kInvalidParam, "monitor_threshold must be in [0, 1]");
    if (cfg.bridge_threshold < 0.0f || cfg.bridge_threshold > 1.0f)
      return MakeError(ErrorCode::kInvalidParam, "bridge_threshold must be in [0, 1]");
    if (cfg.resync_threshold < 0.0f || cfg.resync_threshold > 1.0f)
      return MakeError(ErrorCode::kInvalidParam, "resync_threshold must be in [0, 1]");

    return cfg;
  }

 private:
  static std::vector<uint32_t> ParseUintList(const std::string& s) {
    std::vector<uint32_t> result;
    size_t pos = 0;
    while (pos < s.size()) {
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) pos++;
      if (pos >= s.size()) break;
      size_t end = pos;
      while (end < s.size() && s[end] >= '0' && s[end] <= '9') end++;
      if (end > pos) {
        result.push_back(static_cast<uint32_t>(std::stoul(s.substr(pos, end - pos))));
      }
      pos = end;
    }
    return result;
  }

  static bool ParseStimulus(const char* s, StimulusEvent& ev) {
    // Format: label start_ms end_ms intensity target1,target2,...
    char label[256];
    float start, end, intensity;
    char targets_str[4096];

    if (sscanf(s, " %255s %f %f %f %4095s", label, &start, &end, &intensity, targets_str) != 5) {
      return false;
    }

    ev.label = label;
    ev.start_ms = start;
    ev.end_ms = end;
    ev.intensity = intensity;
    ev.target_neurons = ParseUintList(targets_str);
    return true;
  }
};

}  // namespace mechabrain

#endif  // FWMC_CONFIG_LOADER_H_
