#ifndef PARABRAIN_EXPERIMENT_OPTIMIZER_H_
#define PARABRAIN_EXPERIMENT_OPTIMIZER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/log.h"

namespace mechabrain {

// A single parameter axis with name and range.
struct OptParam {
  std::string name;
  float lo, hi;
};

// Result from a single evaluation.
struct OptResult {
  std::vector<float> params;
  float fitness = -1e9f;
};

// General-purpose experiment optimizer using random search with refinement.
//
// Works with ANY experiment via a callable fitness function. The optimizer
// generates parameter vectors, passes them to the fitness function, and
// tracks the best result. Multi-seed evaluation ensures robustness against
// chaotic dynamics in spiking networks.
//
// Usage:
//   ExperimentOptimizer opt;
//   opt.axes = {{"ring_weight", 5, 40}, {"tonic", 2, 15}};
//   opt.n_iterations = 200;
//   opt.n_seeds = 3;  // evaluate each config with 3 different seeds
//   auto best = opt.Run([](const std::vector<float>& params, uint32_t seed) {
//     MyExperiment exp;
//     exp.ring_weight = params[0];
//     exp.tonic = params[1];
//     auto result = exp.Run(seed);
//     return -result.error;  // higher = better
//   });
//
struct ExperimentOptimizer {
  std::vector<OptParam> axes;
  int n_iterations = 200;       // total parameter samples
  int n_seeds = 3;              // seeds per evaluation (for robustness)
  uint32_t base_seed = 42;
  float refine_fraction = 0.4f; // fraction of iterations spent refining
  float refine_radius = 0.15f;  // fraction of range for perturbation

  // Fitness function: takes parameter vector + seed, returns fitness (higher = better).
  // Called n_seeds times per parameter set; the MINIMUM fitness across seeds is used
  // (worst-case robustness).
  using FitnessFn = std::function<float(const std::vector<float>& params, uint32_t seed)>;

  OptResult Run(FitnessFn fitness_fn) {
    std::mt19937 rng(base_seed + 999);
    size_t n_params = axes.size();
    int broad_n = static_cast<int>(n_iterations * (1.0f - refine_fraction));

    OptResult best;
    best.params.resize(n_params);
    best.fitness = -1e9f;

    auto uniform = [&](float lo, float hi) {
      return std::uniform_real_distribution<float>(lo, hi)(rng);
    };

    // Generate seed list for multi-seed evaluation
    std::vector<uint32_t> eval_seeds(static_cast<size_t>(n_seeds));
    for (int s = 0; s < n_seeds; ++s) {
      eval_seeds[static_cast<size_t>(s)] = base_seed + static_cast<uint32_t>(s) * 7919;
    }

    auto evaluate = [&](const std::vector<float>& params) -> float {
      // Worst-case fitness across seeds (robustness)
      float min_fitness = 1e9f;
      for (uint32_t s : eval_seeds) {
        float f = fitness_fn(params, s);
        min_fitness = std::min(min_fitness, f);
      }
      return min_fitness;
    };

    for (int iter = 0; iter < n_iterations; ++iter) {
      std::vector<float> params(n_params);

      if (iter < broad_n) {
        // Broad exploration: uniform random
        for (size_t i = 0; i < n_params; ++i) {
          params[i] = uniform(axes[i].lo, axes[i].hi);
        }
      } else {
        // Refinement: perturb around best
        for (size_t i = 0; i < n_params; ++i) {
          float range = (axes[i].hi - axes[i].lo) * refine_radius;
          float v = uniform(best.params[i] - range, best.params[i] + range);
          params[i] = std::clamp(v, axes[i].lo, axes[i].hi);
        }
      }

      float fitness = evaluate(params);

      if (fitness > best.fitness) {
        best.params = params;
        best.fitness = fitness;

        // Log improvements during broad phase
        if (iter < broad_n) {
          Log(LogLevel::kInfo, "[optimizer] iter %d/%d: fitness=%.1f",
              iter, n_iterations, fitness);
        }
      }
    }

    // Log best result with parameter names
    std::string params_str;
    for (size_t i = 0; i < n_params; ++i) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%s=%.2f", axes[i].name.c_str(), best.params[i]);
      if (i > 0) params_str += " ";
      params_str += buf;
    }
    Log(LogLevel::kInfo, "[optimizer] BEST (robust over %d seeds): fitness=%.1f  %s",
        n_seeds, best.fitness, params_str.c_str());

    return best;
  }
};

}  // namespace mechabrain

#endif  // PARABRAIN_EXPERIMENT_OPTIMIZER_H_
