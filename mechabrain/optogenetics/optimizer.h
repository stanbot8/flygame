#ifndef FWMC_OPTOGENETICS_OPTIMIZER_H_
#define FWMC_OPTOGENETICS_OPTIMIZER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "optogenetics.h"
#include "core/log.h"

namespace mechabrain {

// A named parameter with bounds, used to define the search space.
struct OptoParam {
  std::string name;
  float min_val;
  float max_val;
  float default_val;

  // Clamp value to valid range.
  float Clamp(float v) const {
    return std::clamp(v, min_val, max_val);
  }

  // Normalize to [0, 1].
  float Normalize(float v) const {
    if (max_val <= min_val) return 0.5f;
    return (v - min_val) / (max_val - min_val);
  }

  // Denormalize from [0, 1].
  float Denormalize(float n) const {
    return min_val + n * (max_val - min_val);
  }
};

// A single trial: parameter values + resulting score.
struct OptoTrial {
  std::vector<float> params;      // parameter values (same order as param_space)
  OptogeneticsResult result;       // full experiment result
  float score = 0.0f;             // objective function output (higher = better)
};

// Objective function type: maps experiment result to scalar score.
// The optimizer maximizes this score.
using OptoObjective = std::function<float(const OptogeneticsResult&)>;

// Common objective functions for optogenetics optimization.
struct OptoObjectives {
  // Maximize excitatory modulation (positive modulation index).
  static float MaxExcitation(const OptogeneticsResult& r) {
    return r.modulation_index;
  }

  // Maximize inhibition (negative modulation index, returned as positive score).
  static float MaxInhibition(const OptogeneticsResult& r) {
    return -r.modulation_index;
  }

  // Maximize modulation while minimizing off-target effects.
  // Weighted sum: modulation - penalty * off_target.
  static OptoObjective SelectiveModulation(float off_target_penalty = 0.5f) {
    return [off_target_penalty](const OptogeneticsResult& r) -> float {
      return std::abs(r.modulation_index)
             - off_target_penalty * std::abs(r.off_target_rate_change_hz);
    };
  }

  // Target a specific evoked firing rate (minimize distance to target).
  static OptoObjective TargetRate(float target_hz) {
    return [target_hz](const OptogeneticsResult& r) -> float {
      float err = std::abs(r.evoked_rate_hz - target_hz);
      return 1.0f / (1.0f + err);  // sigmoid-like: 1.0 at target, 0 far away
    };
  }

  // Maximize modulation while staying under energy safety limit.
  static OptoObjective SafeModulation(float max_energy_mj = 50.0f) {
    return [max_energy_mj](const OptogeneticsResult& r) -> float {
      if (r.max_energy > max_energy_mj) return -1.0f;  // hard constraint
      return std::abs(r.modulation_index);
    };
  }
};

// Configuration for the optimizer.
struct OptimizerConfig {
  // Search budget
  int max_trials = 50;
  int initial_random = 10;    // random exploration before guided search

  // CMA-ES-lite parameters (simplified covariance matrix adaptation)
  float sigma_init = 0.3f;   // initial step size in normalized space
  float sigma_decay = 0.95f;  // step size decay per generation
  int population_size = 8;    // candidates per generation

  // Random seed
  uint32_t seed = 123;

  // Early stopping: stop if score exceeds this threshold
  float target_score = std::numeric_limits<float>::max();

  // Verbosity
  bool verbose = true;
};

// Applies a parameter vector to an OptogeneticsConfig.
// Maps named parameters to config fields.
inline void ApplyParams(const std::vector<OptoParam>& space,
                        const std::vector<float>& values,
                        OptogeneticsConfig& cfg) {
  for (size_t i = 0; i < space.size(); ++i) {
    const auto& p = space[i];
    float v = p.Clamp(values[i]);

    if (p.name == "laser_power_mw") cfg.laser_power_mw = v;
    else if (p.name == "wavelength_nm") cfg.wavelength_nm = v;
    else if (p.name == "objective_na") cfg.objective_na = v;
    else if (p.name == "target_fraction") cfg.target_fraction = v;
    else if (p.name == "stim_on_ms") cfg.stim_on_ms = v;
    else if (p.name == "stim_off_ms") cfg.stim_off_ms = v;
    else if (p.name == "n_pulses") cfg.n_pulses = static_cast<int>(v);
    else if (p.name == "intensity") {}  // applied via stim commands, not config
    else if (p.name == "weight_scale") cfg.weight_scale = v;
  }
}

// Result of an optimization run.
struct OptimizerResult {
  OptoTrial best;
  std::vector<OptoTrial> all_trials;
  int n_trials = 0;
  double total_seconds = 0.0;

  bool converged() const {
    return !all_trials.empty() && best.score > 0.0f;
  }
};

// Parameter search optimizer for optogenetics experiments.
//
// Wraps OptogeneticsExperiment in a search loop that explores the parameter
// space to maximize an objective function. Uses a simplified CMA-ES-like
// strategy: random exploration followed by gaussian perturbation around
// the current best, with adaptive step size.
//
// This is the core building block for AI-driven experiment design:
// an agent calls Optimize() with an objective, observes the result,
// and can refine the search space or objective for the next round.
struct OptogeneticsOptimizer {
  // The parameter search space.
  std::vector<OptoParam> param_space;

  // Base config (non-optimized parameters stay fixed).
  OptogeneticsConfig base_config;

  // Default parameter space for standard optogenetics optimization.
  static std::vector<OptoParam> DefaultParamSpace() {
    return {
      {"laser_power_mw", 1.0f, 50.0f, 10.0f},
      {"target_fraction", 0.05f, 0.5f, 0.2f},
      {"stim_on_ms", 50.0f, 500.0f, 200.0f},
      {"stim_off_ms", 100.0f, 1000.0f, 300.0f},
      {"n_pulses", 1.0f, 20.0f, 5.0f},
    };
  }

  // Extended parameter space including optical parameters.
  static std::vector<OptoParam> ExtendedParamSpace() {
    return {
      {"laser_power_mw", 1.0f, 50.0f, 10.0f},
      {"wavelength_nm", 450.0f, 640.0f, 590.0f},
      {"objective_na", 0.4f, 1.4f, 1.0f},
      {"target_fraction", 0.05f, 0.5f, 0.2f},
      {"stim_on_ms", 50.0f, 500.0f, 200.0f},
      {"stim_off_ms", 100.0f, 1000.0f, 300.0f},
      {"n_pulses", 1.0f, 20.0f, 5.0f},
      {"weight_scale", 0.5f, 2.0f, 1.0f},
    };
  }

  // Run optimization. Returns best parameters found.
  OptimizerResult Optimize(const OptoObjective& objective,
                           const OptimizerConfig& opt_cfg = {}) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (param_space.empty()) {
      param_space = DefaultParamSpace();
    }

    const size_t n_params = param_space.size();
    std::mt19937 rng(opt_cfg.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    OptimizerResult result;
    result.best.score = -std::numeric_limits<float>::max();

    // Current best in normalized space
    std::vector<float> best_normalized(n_params, 0.5f);
    float sigma = opt_cfg.sigma_init;

    int trial_idx = 0;

    // Phase 1: random exploration
    int n_random = std::min(opt_cfg.initial_random, opt_cfg.max_trials);
    for (int i = 0; i < n_random; ++i) {
      std::vector<float> normalized(n_params);
      for (size_t j = 0; j < n_params; ++j) {
        normalized[j] = uniform(rng);
      }
      auto trial = RunTrial(normalized, objective, trial_idx++);
      result.all_trials.push_back(trial);

      if (trial.score > result.best.score) {
        result.best = trial;
        best_normalized = normalized;
      }

      if (opt_cfg.verbose) {
        Log(LogLevel::kInfo, "Opto trial %d (random): score=%.4f [best=%.4f]",
            trial_idx, trial.score, result.best.score);
      }

      if (result.best.score >= opt_cfg.target_score) break;
    }

    // Phase 2: guided search (gaussian perturbation around best)
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    while (trial_idx < opt_cfg.max_trials &&
           result.best.score < opt_cfg.target_score) {
      // Generate population around current best
      std::vector<std::pair<std::vector<float>, float>> candidates;

      for (int p = 0; p < opt_cfg.population_size &&
                       trial_idx < opt_cfg.max_trials; ++p) {
        std::vector<float> normalized(n_params);
        for (size_t j = 0; j < n_params; ++j) {
          normalized[j] = std::clamp(
              best_normalized[j] + sigma * gauss(rng), 0.0f, 1.0f);
        }

        auto trial = RunTrial(normalized, objective, trial_idx++);
        result.all_trials.push_back(trial);
        candidates.push_back({normalized, trial.score});

        if (trial.score > result.best.score) {
          result.best = trial;
        }

        if (opt_cfg.verbose) {
          Log(LogLevel::kInfo,
              "Opto trial %d (guided, sigma=%.3f): score=%.4f [best=%.4f]",
              trial_idx, sigma, trial.score, result.best.score);
        }
      }

      // Update best_normalized: weighted mean of top candidates
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                  return a.second > b.second;
                });

      int n_elite = std::max(1, static_cast<int>(candidates.size()) / 2);
      std::fill(best_normalized.begin(), best_normalized.end(), 0.0f);
      for (int e = 0; e < n_elite; ++e) {
        for (size_t j = 0; j < n_params; ++j) {
          best_normalized[j] += candidates[e].first[j] / n_elite;
        }
      }

      sigma *= opt_cfg.sigma_decay;
    }

    result.n_trials = trial_idx;
    auto t1 = std::chrono::high_resolution_clock::now();
    result.total_seconds = std::chrono::duration<double>(t1 - t0).count();

    if (opt_cfg.verbose) {
      Log(LogLevel::kInfo,
          "Optimization complete: %d trials, best score=%.4f, %.1fs",
          result.n_trials, result.best.score, result.total_seconds);
      LogBestParams(result.best);
    }

    return result;
  }

 private:
  OptoTrial RunTrial(const std::vector<float>& normalized,
                     const OptoObjective& objective,
                     int trial_idx) {
    OptoTrial trial;
    trial.params.resize(param_space.size());
    for (size_t j = 0; j < param_space.size(); ++j) {
      trial.params[j] = param_space[j].Denormalize(normalized[j]);
    }

    // Build config from base + trial params
    OptogeneticsConfig cfg = base_config;
    cfg.seed = base_config.seed + trial_idx;
    ApplyParams(param_space, trial.params, cfg);

    // Run experiment
    OptogeneticsExperiment experiment;
    trial.result = experiment.Run(cfg);
    trial.score = objective(trial.result);

    return trial;
  }

  void LogBestParams(const OptoTrial& best) const {
    for (size_t i = 0; i < param_space.size(); ++i) {
      Log(LogLevel::kInfo, "  %s = %.3f",
          param_space[i].name.c_str(), best.params[i]);
    }
    Log(LogLevel::kInfo, "  modulation_index = %.4f",
        best.result.modulation_index);
    Log(LogLevel::kInfo, "  evoked_rate = %.1f Hz",
        best.result.evoked_rate_hz);
  }
};

}  // namespace mechabrain

#endif  // FWMC_OPTOGENETICS_OPTIMIZER_H_
