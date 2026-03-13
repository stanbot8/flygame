#ifndef FWMC_PARAMETRIC_GEN_H_
#define FWMC_PARAMETRIC_GEN_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/species.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Specifies a cell type and its proportion within a region.
struct CellTypeFraction {
  CellType type;
  float fraction;  // [0, 1], fractions within a region should sum to ~1
};

// Specifies a brain region with neuron count, cell type mix, and internal wiring.
struct RegionSpec {
  std::string name;
  uint32_t n_neurons = 100;
  float internal_density = 0.05f;       // probability of internal connection
  NTType default_nt = kACh;             // default NT for internal synapses
  float release_probability = 1.0f;     // vesicle release probability [0,1]
  std::vector<CellTypeFraction> cell_types;  // cell type distribution

  // NT type distribution for internal synapses (overrides default_nt if set)
  // Maps NTType -> fraction. If empty, all synapses use default_nt.
  struct NTFraction { NTType nt; float fraction; };
  std::vector<NTFraction> nt_distribution;

  // Per-region background current (overrides global if >= 0).
  // Set to -1 to use the global background_current_mean.
  float background_mean = -1.0f;

  // Scan data sources (optional). When set, parametric generation fills
  // the region first, then scan data overwrites positions/types/connectivity.
  // Paths are relative to the .brain file directory.
  std::string scan_neurons_path;   // path to neurons.bin for this region
  std::string scan_synapses_path;  // path to synapses.bin for this region
};

// Specifies a projection (long-range connection) between two regions.
struct ProjectionSpec {
  std::string from_region;
  std::string to_region;
  float density = 0.01f;    // connection probability
  NTType nt_type = kACh;
  float weight_mean = 1.0f;
  float weight_std = 0.2f;  // weight is sampled from N(mean, std), clamped > 0
  float release_probability = 1.0f;  // vesicle release probability [0,1]
};

// Specifies a timed stimulus pattern targeting a region or neuron range.
struct StimulusSpec {
  std::string label;                // e.g., "odor_A", "shock"
  std::string target_region;        // region name (maps to neuron index range)
  float start_ms = 0.0f;
  float end_ms = 100.0f;
  float intensity = 1.0f;           // current injection magnitude
  float fraction = 1.0f;            // fraction of region neurons targeted [0,1]
};

// Complete brain specification: regions + projections + stimuli.
struct BrainSpec {
  std::string name = "parametric_brain";
  Species species = Species::kGeneric;  // organism; drives default params
  std::vector<RegionSpec> regions;
  std::vector<ProjectionSpec> projections;
  std::vector<StimulusSpec> stimuli;
  float global_weight_mean = 1.0f;
  float global_weight_std = 0.3f;
  uint32_t seed = 42;

  // Background drive: Gaussian noise current injected every step.
  // Models tonic synaptic bombardment from unmodeled inputs.
  // mean=5-8 keeps neurons near threshold; std=2-3 adds variability.
  float background_current_mean = 0.0f;
  float background_current_std = 0.0f;

  // Get species-appropriate biophysical defaults.
  SpeciesDefaults GetDefaults() const {
    return SpeciesDefaults::For(species);
  }
};

// Generates a NeuronArray + SynapseTable from a BrainSpec.
// Each region occupies a contiguous slice of the neuron array.
struct ParametricGenerator {
  struct RegionRange {
    std::string name;
    uint32_t start;  // first neuron index
    uint32_t end;    // one past last
  };
  std::vector<RegionRange> region_ranges;

  // Generate neurons and synapses from a brain spec.
  // Returns total neuron count.
  uint32_t Generate(const BrainSpec& spec, NeuronArray& neurons,
                    SynapseTable& synapses, CellTypeManager& types) {
    std::mt19937 rng(spec.seed);

    // Count total neurons and assign region ranges
    uint32_t total = 0;
    region_ranges.clear();
    for (const auto& reg : spec.regions) {
      region_ranges.push_back({reg.name, total, total + reg.n_neurons});
      total += reg.n_neurons;
    }

    // Allocate neurons
    neurons.Resize(total);

    // Assign cell types and regions
    for (size_t r = 0; r < spec.regions.size(); ++r) {
      const auto& reg = spec.regions[r];
      uint32_t start = region_ranges[r].start;
      uint32_t end = region_ranges[r].end;
      uint32_t count = end - start;

      // Set region index
      for (uint32_t i = start; i < end; ++i) {
        neurons.region[i] = static_cast<uint8_t>(r);
      }

      // Assign cell types by fraction
      if (reg.cell_types.empty()) {
        for (uint32_t i = start; i < end; ++i) {
          neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
        }
      } else {
        uint32_t assigned = 0;
        for (const auto& ctf : reg.cell_types) {
          uint32_t n_this = static_cast<uint32_t>(
              std::round(ctf.fraction * count));
          n_this = std::min(n_this, count - assigned);
          for (uint32_t j = 0; j < n_this; ++j) {
            neurons.type[start + assigned + j] =
                static_cast<uint8_t>(ctf.type);
          }
          assigned += n_this;
        }
        // Fill remainder with last type
        if (assigned < count && !reg.cell_types.empty()) {
          auto last_type = reg.cell_types.back().type;
          for (uint32_t j = assigned; j < count; ++j) {
            neurons.type[start + j] = static_cast<uint8_t>(last_type);
          }
        }
      }

      // Assign random positions within a bounding box per region
      // (simple spread for visualization, not anatomically precise)
      std::uniform_real_distribution<float> pos_dist(0.0f, 100.0f);
      float region_offset = r * 150.0f;
      for (uint32_t i = start; i < end; ++i) {
        neurons.x[i] = region_offset + pos_dist(rng);
        neurons.y[i] = pos_dist(rng);
        neurons.z[i] = pos_dist(rng);
      }
    }

    // Generate synapses (COO format, then build CSR)
    std::vector<uint32_t> pre_vec, post_vec;
    std::vector<float> weight_vec;
    std::vector<uint8_t> nt_vec;
    std::vector<float> p_release_vec;
    bool any_stochastic = false;

    // Estimate total synapses for reserve (avoids repeated reallocation)
    {
      size_t est = 0;
      for (const auto& reg : spec.regions)
        est += static_cast<size_t>(static_cast<double>(reg.n_neurons) * reg.n_neurons * reg.internal_density);
      for (const auto& proj : spec.projections) {
        int fi = FindRegion(proj.from_region), ti = FindRegion(proj.to_region);
        if (fi >= 0 && ti >= 0) {
          auto& fr = region_ranges[static_cast<size_t>(fi)];
          auto& tr = region_ranges[static_cast<size_t>(ti)];
          est += static_cast<size_t>(static_cast<double>(fr.end - fr.start) * (tr.end - tr.start) * proj.density);
        }
      }
      pre_vec.reserve(est);
      post_vec.reserve(est);
      weight_vec.reserve(est);
      nt_vec.reserve(est);
      p_release_vec.reserve(est);
    }

    // Internal connections within each region
    for (size_t r = 0; r < spec.regions.size(); ++r) {
      const auto& reg = spec.regions[r];
      uint32_t start = region_ranges[r].start;
      uint32_t end = region_ranges[r].end;

      size_t before = pre_vec.size();
      GenerateConnections(rng, start, end, start, end,
                          reg.internal_density,
                          reg.default_nt, reg.nt_distribution,
                          spec.global_weight_mean, spec.global_weight_std,
                          pre_vec, post_vec, weight_vec, nt_vec,
                          true);  // skip self-loops
      size_t added = pre_vec.size() - before;
      p_release_vec.insert(p_release_vec.end(), added, reg.release_probability);
      if (reg.release_probability < 1.0f) any_stochastic = true;
    }

    // Inter-region projections
    for (const auto& proj : spec.projections) {
      int from_idx = FindRegion(proj.from_region);
      int to_idx = FindRegion(proj.to_region);
      if (from_idx < 0 || to_idx < 0) {
        Log(LogLevel::kWarn, "ParametricGen: unknown region in projection %s -> %s",
            proj.from_region.c_str(), proj.to_region.c_str());
        continue;
      }

      auto& from = region_ranges[static_cast<size_t>(from_idx)];
      auto& to = region_ranges[static_cast<size_t>(to_idx)];

      size_t before = pre_vec.size();
      std::vector<RegionSpec::NTFraction> empty_nt;
      GenerateConnections(rng, from.start, from.end, to.start, to.end,
                          proj.density, proj.nt_type, empty_nt,
                          proj.weight_mean, proj.weight_std,
                          pre_vec, post_vec, weight_vec, nt_vec,
                          false);
      size_t added = pre_vec.size() - before;
      p_release_vec.insert(p_release_vec.end(), added, proj.release_probability);
      if (proj.release_probability < 1.0f) any_stochastic = true;
    }

    // Build CSR (with release probabilities if any are non-deterministic)
    if (any_stochastic) {
      synapses.BuildFromCOO(total, pre_vec, post_vec, weight_vec, nt_vec,
                            p_release_vec);
    } else {
      synapses.BuildFromCOO(total, pre_vec, post_vec, weight_vec, nt_vec);
    }

    // Assign per-neuron params
    types.AssignFromTypes(neurons);

    Log(LogLevel::kInfo, "ParametricGen: %u neurons, %zu synapses across %zu regions",
        total, synapses.Size(), spec.regions.size());

    return total;
  }

 private:
  int FindRegion(const std::string& name) const {
    for (size_t i = 0; i < region_ranges.size(); ++i) {
      if (region_ranges[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }

  // Generate random connections between neuron ranges.
  // For large regions with low density, uses geometric skip sampling
  // (O(expected_edges) instead of O(n²)) to avoid iterating all pairs.
  static void GenerateConnections(
      std::mt19937& rng,
      uint32_t pre_start, uint32_t pre_end,
      uint32_t post_start, uint32_t post_end,
      float density, NTType default_nt,
      const std::vector<RegionSpec::NTFraction>& nt_dist,
      float w_mean, float w_std,
      std::vector<uint32_t>& pre_vec,
      std::vector<uint32_t>& post_vec,
      std::vector<float>& weight_vec,
      std::vector<uint8_t>& nt_vec,
      bool skip_self_loops) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::normal_distribution<float> weight_dist(w_mean, w_std);

    // Build NT CDF for sampling
    std::vector<float> nt_cdf;
    std::vector<NTType> nt_types;
    if (!nt_dist.empty()) {
      float cumulative = 0.0f;
      for (const auto& ntf : nt_dist) {
        cumulative += ntf.fraction;
        nt_cdf.push_back(cumulative);
        nt_types.push_back(ntf.nt);
      }
    }

    auto emit_synapse = [&](uint32_t pre, uint32_t post) {
      pre_vec.push_back(pre);
      post_vec.push_back(post);
      weight_vec.push_back(std::max(0.01f, weight_dist(rng)));
      if (!nt_cdf.empty()) {
        float r = coin(rng);
        NTType nt = nt_types.back();
        for (size_t k = 0; k < nt_cdf.size(); ++k) {
          if (r <= nt_cdf[k]) { nt = nt_types[k]; break; }
        }
        nt_vec.push_back(static_cast<uint8_t>(nt));
      } else {
        nt_vec.push_back(static_cast<uint8_t>(default_nt));
      }
    };

    uint64_t n_pre = pre_end - pre_start;
    uint64_t n_post = post_end - post_start;
    uint64_t total_pairs = n_pre * n_post;

    // Use geometric skip sampling when density is low and space is large.
    // Instead of flipping a coin for each of n² pairs, we sample the
    // geometric distribution to skip directly to the next edge.
    // This gives O(expected_edges) time instead of O(n²).
    if (density <= 0.0f) return;  // no edges to generate
    if (density >= 1.0f) density = 1.0f;
    bool use_sparse = (total_pairs > 100000 && density < 0.1f);

    if (use_sparse) {
      // Geometric skip: gap between consecutive edges ~ Geometric(density)
      // Skip = floor(log(U) / log(1 - density)), where U ~ Uniform(0,1)
      double log_complement = std::log(1.0 - static_cast<double>(density));

      int64_t idx = -1;
      while (true) {
        // Sample skip from geometric distribution
        double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
        int64_t skip = static_cast<int64_t>(std::log(u) / log_complement);
        idx += skip + 1;

        if (idx >= static_cast<int64_t>(total_pairs)) break;

        uint32_t pre = pre_start + static_cast<uint32_t>(idx / n_post);
        uint32_t post = post_start + static_cast<uint32_t>(idx % n_post);
        if (skip_self_loops && pre == post) continue;

        emit_synapse(pre, post);
      }
    } else {
      // Dense path: iterate all pairs (small regions)
      for (uint32_t pre = pre_start; pre < pre_end; ++pre) {
        for (uint32_t post = post_start; post < post_end; ++post) {
          if (skip_self_loops && pre == post) continue;
          if (coin(rng) >= density) continue;
          emit_synapse(pre, post);
        }
      }
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_PARAMETRIC_GEN_H_
