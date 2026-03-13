#ifndef FWMC_PARAM_SWEEP_H_
#define FWMC_PARAM_SWEEP_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/experiment_config.h"
#include "core/izhikevich.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// A single point in parameter space with its score.
struct SweepPoint {
  IzhikevichParams params;
  CellType cell_type = CellType::kGeneric;
  float score = 0.0f;  // higher = better (e.g., spike correlation with bio)
};

// Defines the bounds for a single Izhikevich parameter.
struct ParamRange {
  float a_min = 0.01f, a_max = 0.15f;
  float b_min = 0.15f, b_max = 0.30f;
  float c_min = -70.0f, c_max = -45.0f;
  float d_min = 1.0f,  d_max = 10.0f;
};

// Scoring function: given neurons after simulation, returns a fitness score.
// The sweep engine calls this after running a short simulation with each
// parameter set. Higher scores are better.
using ScoreFn = std::function<float(const NeuronArray& neurons,
                                     float sim_time_ms)>;

// Parameter sweep engine. Explores Izhikevich parameter space for one cell type
// by running short simulations and scoring the output.
struct ParamSweep {
  // Sweep configuration
  ParamRange range;
  int grid_steps = 5;       // steps per dimension for grid sweep (total = steps^4)
  int random_samples = 100; // samples for random sweep
  float sim_duration_ms = 500.0f;
  float dt_ms = 0.1f;
  float weight_scale = 1.0f;
  uint32_t seed = 42;

  // Results (sorted best-first after sweep)
  std::vector<SweepPoint> results;

  // Run a grid sweep over (a, b, c, d) space.
  // For each point, creates neurons of the target cell type, runs a short sim,
  // and scores the output.
  void GridSweep(CellType target_type,
                 const NeuronArray& base_neurons,
                 SynapseTable& synapses,
                 ScoreFn score_fn) {
    results.clear();
    int steps = grid_steps;

    float da = (steps > 1) ? (range.a_max - range.a_min) / (steps - 1) : 0;
    float db = (steps > 1) ? (range.b_max - range.b_min) / (steps - 1) : 0;
    float dc = (steps > 1) ? (range.c_max - range.c_min) / (steps - 1) : 0;
    float dd = (steps > 1) ? (range.d_max - range.d_min) / (steps - 1) : 0;

    size_t total = static_cast<size_t>(steps) * steps * steps * steps;
    Log(LogLevel::kInfo, "ParamSweep: grid sweep %d^4 = %zu points for %s",
        steps, total, CellTypeName(target_type));

    for (int ia = 0; ia < steps; ++ia) {
      for (int ib = 0; ib < steps; ++ib) {
        for (int ic = 0; ic < steps; ++ic) {
          for (int id = 0; id < steps; ++id) {
            IzhikevichParams p;
            p.a = range.a_min + ia * da;
            p.b = range.b_min + ib * db;
            p.c = range.c_min + ic * dc;
            p.d = range.d_min + id * dd;

            float s = EvaluateParams(p, target_type, base_neurons,
                                     synapses, score_fn);
            results.push_back({p, target_type, s});
          }
        }
      }
    }

    SortResults();
    LogTopResults(5);
  }

  // Run a random sweep with uniform sampling in parameter space.
  void RandomSweep(CellType target_type,
                   const NeuronArray& base_neurons,
                   SynapseTable& synapses,
                   ScoreFn score_fn) {
    results.clear();
    std::mt19937 rng(seed);

    Log(LogLevel::kInfo, "ParamSweep: random sweep %d samples for %s",
        random_samples, CellTypeName(target_type));

    std::uniform_real_distribution<float> dist_a(range.a_min, range.a_max);
    std::uniform_real_distribution<float> dist_b(range.b_min, range.b_max);
    std::uniform_real_distribution<float> dist_c(range.c_min, range.c_max);
    std::uniform_real_distribution<float> dist_d(range.d_min, range.d_max);

    for (int i = 0; i < random_samples; ++i) {
      IzhikevichParams p;
      p.a = dist_a(rng);
      p.b = dist_b(rng);
      p.c = dist_c(rng);
      p.d = dist_d(rng);

      float s = EvaluateParams(p, target_type, base_neurons,
                               synapses, score_fn);
      results.push_back({p, target_type, s});
    }

    SortResults();
    LogTopResults(5);
  }

  // Hill-climb from the best point found so far.
  // Perturbs each parameter by step_size, keeps improvement.
  void Refine(CellType target_type,
              const NeuronArray& base_neurons,
              SynapseTable& synapses,
              ScoreFn score_fn,
              int iterations = 50,
              float step_size = 0.5f) {
    if (results.empty()) return;

    SweepPoint best = results[0];
    Log(LogLevel::kInfo, "ParamSweep: refining from score=%.4f", best.score);

    std::mt19937 rng(seed + 1000);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    for (int iter = 0; iter < iterations; ++iter) {
      IzhikevichParams candidate = best.params;

      // Perturb one random parameter
      int dim = rng() % 4;
      float perturbation = noise(rng) * step_size;
      switch (dim) {
        case 0: candidate.a = std::clamp(candidate.a + perturbation * 0.01f,
                                          range.a_min, range.a_max); break;
        case 1: candidate.b = std::clamp(candidate.b + perturbation * 0.02f,
                                          range.b_min, range.b_max); break;
        case 2: candidate.c = std::clamp(candidate.c + perturbation * 2.0f,
                                          range.c_min, range.c_max); break;
        case 3: candidate.d = std::clamp(candidate.d + perturbation * 1.0f,
                                          range.d_min, range.d_max); break;
      }

      float s = EvaluateParams(candidate, target_type, base_neurons,
                               synapses, score_fn);
      if (s > best.score) {
        best = {candidate, target_type, s};
      }
    }

    // Insert refined result at front
    results.insert(results.begin(), best);
    Log(LogLevel::kInfo, "ParamSweep: refined to score=%.4f  a=%.4f b=%.4f c=%.2f d=%.2f",
        best.score, best.params.a, best.params.b, best.params.c, best.params.d);
  }

  // Get the best parameters found.
  IzhikevichParams BestParams() const {
    if (results.empty()) return {};
    return results[0].params;
  }

  float BestScore() const {
    if (results.empty()) return 0.0f;
    return results[0].score;
  }

 private:
  float EvaluateParams(const IzhikevichParams& p, CellType target_type,
                       const NeuronArray& base_neurons,
                       SynapseTable& synapses,
                       const ScoreFn& score_fn) {
    NeuronArray trial;
    trial.Resize(base_neurons.n);
    trial.i_ext = base_neurons.i_ext;
    trial.type = base_neurons.type;
    trial.region = base_neurons.region;

    // Build CellTypeManager with the candidate params for the target type
    CellTypeManager types;
    types.SetOverride(target_type, p);
    types.AssignFromTypes(trial);

    // Run simulation
    int n_steps = static_cast<int>(sim_duration_ms / dt_ms);
    float sim_time = 0.0f;

    for (int step = 0; step < n_steps; ++step) {
      trial.ClearSynapticInput();
      synapses.PropagateSpikes(trial.spiked.data(), trial.i_syn.data(),
                               weight_scale);
      IzhikevichStepHeterogeneous(trial, dt_ms, sim_time, types);
      sim_time += dt_ms;
    }

    return score_fn(trial, sim_time);
  }

  void SortResults() {
    std::sort(results.begin(), results.end(),
              [](const SweepPoint& a, const SweepPoint& b) {
                return a.score > b.score;
              });
  }

  void LogTopResults(int n) const {
    int show = std::min(n, static_cast<int>(results.size()));
    for (int i = 0; i < show; ++i) {
      const auto& r = results[static_cast<size_t>(i)];
      Log(LogLevel::kInfo, "  #%d score=%.4f  a=%.4f b=%.4f c=%.2f d=%.2f",
          i + 1, r.score, r.params.a, r.params.b, r.params.c, r.params.d);
    }
  }

  static const char* CellTypeName(CellType ct) {
    switch (ct) {
      case CellType::kKenyonCell: return "KC";
      case CellType::kMBON_cholinergic: return "MBON_ACh";
      case CellType::kMBON_gabaergic: return "MBON_GABA";
      case CellType::kDAN_PPL1: return "DAN_PPL1";
      case CellType::kDAN_PAM: return "DAN_PAM";
      case CellType::kPN_excitatory: return "PN_exc";
      case CellType::kPN_inhibitory: return "PN_inh";
      case CellType::kLN_local: return "LN";
      case CellType::kORN: return "ORN";
      case CellType::kFastSpiking: return "FS";
      case CellType::kBursting: return "Burst";
      default: return "Generic";
    }
  }
};

// Built-in scoring functions for common use cases.
namespace scoring {

// Target a specific firing rate (Hz). Score is inverse of error.
inline ScoreFn TargetFiringRate(float target_hz, float dt_ms) {
  return [target_hz, dt_ms](const NeuronArray& neurons, float sim_time_ms) -> float {
    int spikes = neurons.CountSpikes();
    float duration_s = sim_time_ms / 1000.0f;
    float rate_hz = (duration_s > 0) ?
        static_cast<float>(spikes) / (static_cast<float>(neurons.n) * duration_s)
        : 0.0f;
    float error = std::abs(rate_hz - target_hz);
    return 1.0f / (1.0f + error);
  };
}

// Reward spiking activity in a target range. Score = fraction of neurons that
// spiked at least once during the sim, penalized if too high.
inline ScoreFn ActivityInRange(float min_fraction, float max_fraction) {
  return [min_fraction, max_fraction](const NeuronArray& neurons, float /*sim_time_ms*/) -> float {
    int active = 0;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.last_spike_time[i] > 0.0f) active++;
    }
    float frac = static_cast<float>(active) / static_cast<float>(neurons.n);
    if (frac < min_fraction) return frac / min_fraction;
    if (frac > max_fraction) return max_fraction / frac;
    return 1.0f;
  };
}

// Reward biologically realistic coefficient of variation of ISI.
// Fly neurons typically have CV_ISI ≈ 0.5-1.0.
inline ScoreFn RealisticCV(float target_cv = 0.7f) {
  return [target_cv](const NeuronArray& neurons, float /*sim_time_ms*/) -> float {
    // Use last_spike_time distribution as proxy for ISI regularity
    std::vector<float> spike_times;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.last_spike_time[i] > 0.0f) {
        spike_times.push_back(neurons.last_spike_time[i]);
      }
    }
    if (spike_times.size() < 5) return 0.0f;

    float mean = 0;
    for (float t : spike_times) mean += t;
    mean /= static_cast<float>(spike_times.size());

    float var = 0;
    for (float t : spike_times) var += (t - mean) * (t - mean);
    var /= static_cast<float>(spike_times.size());

    float cv = (mean > 0) ? std::sqrt(var) / mean : 0.0f;
    float error = std::abs(cv - target_cv);
    return 1.0f / (1.0f + error * 5.0f);
  };
}

}  // namespace scoring

// ===================================================================
// Experiment-level parameter sweep engine
// Sweeps arbitrary experiment parameters (not just Izhikevich params)
// across grid, random, or hill-climbing search strategies.
// ===================================================================

// A single axis in experiment parameter space
struct SweepAxis {
    std::string param_name;  // e.g. "weight_scale", "stdp.a_plus", "dt_ms"
    std::vector<float> values;  // explicit values to try

    // Generate linearly-spaced values
    static SweepAxis Linear(const std::string& name, float lo, float hi, int n) {
        SweepAxis axis;
        axis.param_name = name;
        axis.values.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1)
                              : 0.5f;
            axis.values[static_cast<size_t>(i)] = lo + t * (hi - lo);
        }
        return axis;
    }

    // Generate logarithmically-spaced values (lo and hi must be > 0)
    static SweepAxis Log(const std::string& name, float lo, float hi, int n) {
        SweepAxis axis;
        axis.param_name = name;
        axis.values.resize(static_cast<size_t>(n));
        float log_lo = std::log10(lo);
        float log_hi = std::log10(hi);
        for (int i = 0; i < n; ++i) {
            float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1)
                              : 0.5f;
            axis.values[static_cast<size_t>(i)] = std::pow(10.0f, log_lo + t * (log_hi - log_lo));
        }
        return axis;
    }
};

// Applies a named parameter to an ExperimentConfig.
// Supports the most commonly swept parameters.
inline void ApplySweepParam(ExperimentConfig& cfg,
                             const std::string& name, float value) {
    if (name == "weight_scale") cfg.weight_scale = value;
    else if (name == "dt_ms") cfg.dt_ms = value;
    else if (name == "monitor_threshold") cfg.monitor_threshold = value;
    else if (name == "bridge_threshold") cfg.bridge_threshold = value;
    else if (name == "resync_threshold") cfg.resync_threshold = value;
    else if (name == "min_observation_ms") cfg.min_observation_ms = value;
    else if (name == "calibration_lr") cfg.calibration_lr = value;
    else if (name == "duration_ms") cfg.duration_ms = value;
    else {
        Log(LogLevel::kWarn, "SweepParam: unknown parameter '%s'", name.c_str());
    }
}

// Callback that runs one experiment config and returns a metric value.
// The sweep engine is agnostic to the simulation backend. The caller
// provides this function, which typically creates an ExperimentRunner,
// calls Run(), and extracts the desired metric (e.g., final correlation).
using ExperimentEvalFn = std::function<float(const ExperimentConfig& config,
                                              const std::string& run_dir)>;

// Experiment-level parameter sweep
struct ExperimentSweep {
    std::vector<SweepAxis> axes;
    std::string output_dir = "sweep_results";
    std::string metric = "correlation";  // metric to optimize (informational)

    struct SweepResult {
        std::vector<float> params;       // one value per axis
        float metric_value;
        std::string result_dir;
    };

    std::vector<SweepResult> all_results;

    // -------------------------------------------------------------------
    // Grid search: evaluate all combinations of axis values
    // -------------------------------------------------------------------
    std::vector<SweepResult> GridSearch(const ExperimentConfig& base_config,
                                         ExperimentEvalFn eval_fn) {
        all_results.clear();

        // Compute total combinations
        size_t total = 1;
        for (const auto& axis : axes) total *= axis.values.size();

        Log(LogLevel::kInfo, "ExperimentSweep: grid search over %zu combinations",
            total);

        // Enumerate all combinations using a mixed-radix counter
        std::vector<size_t> indices(axes.size(), 0);

        for (size_t combo = 0; combo < total; ++combo) {
            // Build config for this combination
            ExperimentConfig cfg = base_config;
            std::vector<float> param_vals;

            for (size_t a = 0; a < axes.size(); ++a) {
                float val = axes[a].values[indices[a]];
                ApplySweepParam(cfg, axes[a].param_name, val);
                param_vals.push_back(val);
            }

            std::string run_dir = output_dir + "/run_" + std::to_string(combo);
            cfg.output_dir = run_dir;

            Log(LogLevel::kInfo, "  Grid run %zu/%zu", combo + 1, total);
            float m = eval_fn(cfg, run_dir);

            all_results.push_back({param_vals, m, run_dir});

            // Advance mixed-radix counter
            for (int a = static_cast<int>(axes.size()) - 1; a >= 0; --a) {
                indices[static_cast<size_t>(a)]++;
                if (indices[static_cast<size_t>(a)] < axes[static_cast<size_t>(a)].values.size())
                    break;
                indices[static_cast<size_t>(a)] = 0;
            }
        }

        SortResults();
        LogBest();
        return all_results;
    }

    // -------------------------------------------------------------------
    // Random search: sample N random combinations uniformly
    // -------------------------------------------------------------------
    std::vector<SweepResult> RandomSearch(const ExperimentConfig& base_config,
                                           ExperimentEvalFn eval_fn,
                                           int n_samples,
                                           uint32_t seed = 42) {
        all_results.clear();
        std::mt19937 rng(seed);

        Log(LogLevel::kInfo, "ExperimentSweep: random search %d samples", n_samples);

        for (int i = 0; i < n_samples; ++i) {
            ExperimentConfig cfg = base_config;
            std::vector<float> param_vals;

            for (const auto& axis : axes) {
                std::uniform_int_distribution<size_t> dist(0, axis.values.size() - 1);
                float val = axis.values[dist(rng)];
                ApplySweepParam(cfg, axis.param_name, val);
                param_vals.push_back(val);
            }

            std::string run_dir = output_dir + "/random_" + std::to_string(i);
            cfg.output_dir = run_dir;

            Log(LogLevel::kInfo, "  Random run %d/%d", i + 1, n_samples);
            float m = eval_fn(cfg, run_dir);

            all_results.push_back({param_vals, m, run_dir});
        }

        SortResults();
        LogBest();
        return all_results;
    }

    // -------------------------------------------------------------------
    // Hill climbing: start from center of each axis, greedily move to
    // the best neighbor at each iteration
    // -------------------------------------------------------------------
    SweepResult HillClimb(const ExperimentConfig& base_config,
                           ExperimentEvalFn eval_fn,
                           int max_iters = 50) {
        // Start from the midpoint index of each axis
        std::vector<size_t> current(axes.size());
        for (size_t a = 0; a < axes.size(); ++a)
            current[a] = axes[a].values.size() / 2;

        // Evaluate starting point
        auto evaluate = [&](const std::vector<size_t>& idx) -> SweepResult {
            ExperimentConfig cfg = base_config;
            std::vector<float> param_vals;
            for (size_t a = 0; a < axes.size(); ++a) {
                float val = axes[a].values[idx[a]];
                ApplySweepParam(cfg, axes[a].param_name, val);
                param_vals.push_back(val);
            }
            std::string run_dir = output_dir + "/hill_" +
                std::to_string(all_results.size());
            cfg.output_dir = run_dir;
            float m = eval_fn(cfg, run_dir);
            SweepResult r{param_vals, m, run_dir};
            all_results.push_back(r);
            return r;
        };

        Log(LogLevel::kInfo, "ExperimentSweep: hill climb, max %d iterations",
            max_iters);

        SweepResult best = evaluate(current);
        Log(LogLevel::kInfo, "  Hill climb start: metric=%.4f", best.metric_value);

        for (int iter = 0; iter < max_iters; ++iter) {
            bool improved = false;

            // Try moving +1 and -1 along each axis
            for (size_t a = 0; a < axes.size(); ++a) {
                for (int delta : {-1, +1}) {
                    int new_idx = static_cast<int>(current[a]) + delta;
                    if (new_idx < 0 ||
                        new_idx >= static_cast<int>(axes[a].values.size()))
                        continue;

                    auto neighbor = current;
                    neighbor[a] = static_cast<size_t>(new_idx);

                    SweepResult r = evaluate(neighbor);
                    if (r.metric_value > best.metric_value) {
                        best = r;
                        current = neighbor;
                        improved = true;
                        Log(LogLevel::kInfo,
                            "  Hill climb iter %d: improved to %.4f "
                            "(moved %s %+d)",
                            iter, best.metric_value,
                            axes[a].param_name.c_str(), delta);
                    }
                }
            }

            if (!improved) {
                Log(LogLevel::kInfo, "  Hill climb converged at iter %d", iter);
                break;
            }
        }

        Log(LogLevel::kInfo, "  Hill climb best: metric=%.4f", best.metric_value);
        return best;
    }

    // -------------------------------------------------------------------
    // Results I/O
    // -------------------------------------------------------------------

    // Save all results to CSV
    void SaveResultsCSV(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) {
            Log(LogLevel::kError, "Failed to open %s for CSV output", path.c_str());
            return;
        }

        // Header
        for (size_t a = 0; a < axes.size(); ++a) {
            fprintf(f, "%s,", axes[a].param_name.c_str());
        }
        fprintf(f, "%s,result_dir\n", metric.c_str());

        // Data rows
        for (const auto& r : all_results) {
            for (size_t a = 0; a < r.params.size(); ++a) {
                fprintf(f, "%g,", r.params[a]);
            }
            fprintf(f, "%g,%s\n", r.metric_value, r.result_dir.c_str());
        }

        fclose(f);
        Log(LogLevel::kInfo, "Sweep results saved to %s (%zu rows)",
            path.c_str(), all_results.size());
    }

    // Best result by metric (highest value)
    SweepResult Best() const {
        if (all_results.empty()) return {{}, 0.0f, ""};
        return all_results.front();  // sorted best-first after search
    }

private:
    void SortResults() {
        std::sort(all_results.begin(), all_results.end(),
                  [](const SweepResult& a, const SweepResult& b) {
                      return a.metric_value > b.metric_value;
                  });
    }

    void LogBest() const {
        if (all_results.empty()) return;
        const auto& best = all_results.front();
        std::string params_str;
        for (size_t a = 0; a < axes.size() && a < best.params.size(); ++a) {
            if (a > 0) params_str += ", ";
            char buf[64];
            snprintf(buf, sizeof(buf), "%s=%g",
                     axes[a].param_name.c_str(), best.params[a]);
            params_str += buf;
        }
        Log(LogLevel::kInfo, "ExperimentSweep best: %s -> %s=%.4f",
            params_str.c_str(), metric.c_str(), best.metric_value);
    }
};

}  // namespace mechabrain

#endif  // FWMC_PARAM_SWEEP_H_
