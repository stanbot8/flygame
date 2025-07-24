#ifndef FWMC_LOD_TRANSITION_H_
#define FWMC_LOD_TRANSITION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/compartmental_neuron.h"
#include "core/izhikevich.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "tissue/brain_sdf.h"
#include "tissue/lod_manager.h"
#include "tissue/neural_field.h"
#include "tissue/population_model.h"
#include "tissue/voxel_grid.h"

namespace mechabrain {

// LOD Transition Engine: manages state conversion between simulation levels.
//
// Four-level hierarchy for scaling from fly to human brain:
//
//   LOD 0: Wilson-Cowan neural field on voxel grid (cheapest)
//   LOD 1: Region-level population statistics (reserved)
//   LOD 2: Individual Izhikevich spiking neurons with CSR synapses
//   LOD 3: 3-compartment pyramidal neurons with active channels (finest)
//
// The core insight: run a cheap Wilson-Cowan field everywhere as a global
// backbone, then dynamically spawn detailed spiking or compartmental neurons
// only where resolution matters. As the focus point moves, regions
// escalate/de-escalate automatically with state transfer between scales.
//
// Transitions:
//   LOD 0 -> LOD 2: spawn Izhikevich neurons from field E/I state
//   LOD 2 -> LOD 3: convert point neurons to 3-compartment models
//   LOD 3 -> LOD 2: collapse compartments back to point neurons
//   LOD 2 -> LOD 0: write spike rates back to field, discard neurons
//
// Boundary coupling (every timestep):
//   Field -> Neurons: inject E activity as external current into boundary neurons
//   Neurons -> Field: overwrite E in covered voxels from local spike rates
//
// This is analogous to adaptive mesh refinement in CFD or LOD in game engines.

// A chunk of spiking neurons spawned for one region at LOD 2.
struct NeuronChunk {
  std::string name;
  uint32_t region_idx = 0;       // index into LODManager::region_lods

  NeuronArray neurons;
  SynapseTable synapses;
  IzhikevichParams params;

  // Spatial extent (from BrainSDF primitive)
  float center_x = 0, center_y = 0, center_z = 0;
  float radius_x = 0, radius_y = 0, radius_z = 0;

  // Voxel indices this chunk covers (indices into VoxelGrid)
  std::vector<size_t> covered_voxels;

  // Per-neuron: which voxel does this neuron belong to?
  // Maps neuron index -> index into covered_voxels.
  std::vector<uint32_t> neuron_voxel_map;

  // Boundary neurons: those near the edge of the region, which receive
  // input from the neural field to represent unmodeled external connections.
  std::vector<uint32_t> boundary_indices;

  // Firing rate estimation (sliding window)
  std::vector<int> voxel_spike_counts;   // per covered voxel
  int step_count = 0;
  int rate_window_steps = 100;  // compute rate over this many steps

  // Reset spike counters for a new rate estimation window.
  void ResetRateCounters() {
    std::fill(voxel_spike_counts.begin(), voxel_spike_counts.end(), 0);
    step_count = 0;
  }

  // Accumulate spikes into per-voxel counters.
  void AccumulateSpikes() {
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.spiked[i]) {
        voxel_spike_counts[neuron_voxel_map[i]]++;
      }
    }
    step_count++;
  }

  // Compute mean firing rate (Hz) for a covered voxel.
  // neurons_per_voxel: how many neurons were placed in that voxel.
  float VoxelRate(size_t voxel_local_idx, int neurons_in_voxel,
                  float dt_ms) const {
    if (neurons_in_voxel <= 0 || step_count <= 0) return 0.0f;
    float window_s = step_count * dt_ms / 1000.0f;
    return static_cast<float>(voxel_spike_counts[voxel_local_idx]) /
           (static_cast<float>(neurons_in_voxel) * window_s);
  }

  // Total spike count this step.
  int CountSpikes() const { return neurons.CountSpikes(); }
};

// A chunk of compartmental neurons for one region at LOD 3.
// Provides multi-compartment biophysical detail (soma + apical + basal
// dendrites with active channels, NMDA, Ca2+ dynamics, bAPs).
// Used for regions closest to the LOD focus where cellular-resolution
// fidelity matters, such as cortical pyramidal cells during bridging.
struct CompartmentalChunk {
  std::string name;
  uint32_t region_idx = 0;

  CompartmentalArray neurons;
  SynapseTable synapses;        // synapses deliver to soma compartment
  CompartmentalParams params;

  // Spatial extent (from BrainSDF primitive)
  float center_x = 0, center_y = 0, center_z = 0;
  float radius_x = 0, radius_y = 0, radius_z = 0;

  // Per-neuron world position (um). Not in CompartmentalArray because
  // positions are a spatial concern, not a dynamics concern.
  std::vector<float> x, y, z;

  // Voxel coverage (same semantics as NeuronChunk)
  std::vector<size_t> covered_voxels;
  std::vector<uint32_t> neuron_voxel_map;
  std::vector<uint32_t> boundary_indices;

  // Firing rate estimation
  std::vector<int> voxel_spike_counts;
  int step_count = 0;
  int rate_window_steps = 100;

  void ResetRateCounters() {
    std::fill(voxel_spike_counts.begin(), voxel_spike_counts.end(), 0);
    step_count = 0;
  }

  void AccumulateSpikes() {
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.spiked[i]) {
        voxel_spike_counts[neuron_voxel_map[i]]++;
      }
    }
    step_count++;
  }

  float VoxelRate(size_t voxel_local_idx, int neurons_in_voxel,
                  float dt_ms) const {
    if (neurons_in_voxel <= 0 || step_count <= 0) return 0.0f;
    float window_s = step_count * dt_ms / 1000.0f;
    return static_cast<float>(voxel_spike_counts[voxel_local_idx]) /
           (static_cast<float>(neurons_in_voxel) * window_s);
  }

  int CountSpikes() const { return neurons.CountSpikes(); }
};

// A population-level chunk for one region at LOD 1.
// Uses the Montbrio-Pazo-Roxin exact mean-field reduction of QIF neurons.
// Much cheaper than individual neurons (6 ODEs per region) while capturing
// adaptation, refractory dynamics, and E/I population rates.
struct PopulationChunk {
  std::string name;
  uint32_t region_idx = 0;

  PopulationState state;
  PopulationParams params;

  // Spatial extent (from BrainSDF primitive)
  float center_x = 0, center_y = 0, center_z = 0;
  float radius_x = 0, radius_y = 0, radius_z = 0;

  // Voxel indices this chunk covers
  std::vector<size_t> covered_voxels;

  // Nominal neuron count for this region (used for spike probability scaling).
  // When escalating to LOD 2, this many neurons are spawned.
  uint32_t nominal_neurons = 0;

  // E/I ratio (fraction of neurons that are excitatory)
  float ei_ratio = 0.8f;  // 80% E, 20% I (Drosophila and cortical default)
};

struct LODTransitionConfig {
  // Neuron density when escalating: neurons per inside-brain voxel.
  // Drosophila: ~140k neurons in ~500x300x200um brain. At 10um voxels,
  // that's ~3000 voxels inside, so ~47 neurons/voxel on average.
  // We default to a lower density for performance; user can increase.
  float neurons_per_voxel = 10.0f;

  // Boundary layer thickness (um). Neurons within this distance of the
  // region edge receive field-derived input.
  float boundary_width_um = 20.0f;

  // Coupling scale: neural field E activity [0,1] -> external current (pA).
  // Tuned so that E=0.5 (moderate activity) produces ~5 pA, enough to
  // bring a resting neuron near threshold.
  float field_to_neuron_scale = 10.0f;

  // Coupling scale: neuron firing rate (Hz) -> field E activity [0,1].
  // Typical Drosophila rates 1-30 Hz map to E in [0, 0.3].
  float neuron_to_field_scale = 0.01f;

  // Max firing rate for normalization (Hz). Rates above this are clamped.
  float max_rate_hz = 100.0f;

  // Intra-region synapse density (connection probability).
  float synapse_density = 0.05f;

  // Synapse weight statistics.
  float weight_mean = 1.0f;
  float weight_std = 0.3f;

  // Weight scale applied during spike propagation.
  float weight_scale = 1.0f;

  // RNG seed.
  uint32_t seed = 42;

  // Default Izhikevich params for spawned neurons.
  IzhikevichParams default_params;

  // Default population params for LOD 1 regions.
  PopulationParams population_params = PopulationParams::Drosophila();

  // Default compartmental params for LOD 3 neurons (L5 pyramidal).
  CompartmentalParams compartmental_params = DefaultPyramidalParams();

  // When true, Step() automatically transitions LOD levels based on
  // focus distance. Set false for tests that manually control escalation.
  bool auto_lod = true;
};

// Defines a long-range projection between two brain regions for the LOD system.
// When both source and target regions are at LOD 2+ (spiking/compartmental),
// the engine instantiates synapses connecting neurons across the two chunks.
// This models axonal tracts (e.g., antennal lobe -> mushroom body calyx,
// optic lobe -> central brain) that are lost in the continuum field.
struct LODProjection {
  std::string from_region;
  std::string to_region;
  float density = 0.01f;     // connection probability
  float weight_mean = 1.0f;
  float weight_std = 0.2f;
  uint8_t nt_type = 0;       // NTType (kACh, kGABA, etc.)
};

// Active cross-chunk synaptic connection. Created when both endpoints of
// an LODProjection are at LOD 2+. Destroyed when either endpoint de-escalates.
// Pre indices in the SynapseTable are local to the source chunk.
// Post indices are local to the target chunk.
struct CrossChunkLink {
  uint32_t src_region;   // index into LODManager::region_lods
  uint32_t dst_region;
  SynapseTable synapses;
};

// Pre-resolved projection: region names resolved to indices once at Init().
struct ResolvedProjection {
  uint32_t src_region = 0;
  uint32_t dst_region = 0;
  float density = 0.01f;
};

struct LODTransitionEngine {
  VoxelGrid grid;
  NeuralField field;
  BrainSDF brain_sdf;
  LODManager lod;
  LODTransitionConfig config;

  // Active population chunks (one per region at LOD 1).
  std::vector<PopulationChunk> pop_chunks;

  // Active neuron chunks (one per region at LOD 2).
  std::vector<NeuronChunk> chunks;

  // Active compartmental chunks (one per region at LOD 3).
  std::vector<CompartmentalChunk> comp_chunks;

  // Inter-region projection definitions (set by caller before Init).
  std::vector<LODProjection> projections;

  // Active cross-chunk synaptic connections (instantiated on escalation).
  std::vector<CrossChunkLink> cross_links;

  // Pre-resolved projection indices (avoids per-timestep FindRegion string lookups).
  std::vector<ResolvedProjection> resolved_projections;

  // SDF and region channels on the grid.
  size_t ch_sdf = SIZE_MAX;
  size_t ch_region = SIZE_MAX;

  // Per-voxel neuron count (for rate normalization). Populated during escalation.
  std::vector<int> voxel_neuron_count;

  // Pre-computed region -> voxel index map. Built once during Init().
  // region_voxels[region_idx] = sorted list of global voxel indices inside that region.
  // Eliminates O(total_voxels) scans during escalation.
  std::vector<std::vector<size_t>> region_voxels;

  // Track which regions currently have active chunks.
  // Maps region_idx -> index into pop_chunks vector, or -1 if not active.
  std::vector<int> region_pop_map;

  // Maps region_idx -> index into chunks vector, or -1 if not active.
  std::vector<int> region_chunk_map;

  // Maps region_idx -> index into comp_chunks vector, or -1 if not active.
  std::vector<int> region_comp_map;

  bool initialized = false;

  // Initialize the engine from a brain SDF.
  // grid_spacing_um: voxel size. Smaller = more voxels = finer field resolution
  //                  but more memory and compute. 10um is a good default for
  //                  Drosophila; 100-500um for human cortex regions.
  void Init(float grid_spacing_um = 10.0f) {
    // Compute bounding box from SDF primitives
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    for (const auto& p : brain_sdf.primitives) {
      min_x = std::min(min_x, p.cx - p.rx - 10.0f);
      min_y = std::min(min_y, p.cy - p.ry - 10.0f);
      min_z = std::min(min_z, p.cz - p.rz - 10.0f);
      max_x = std::max(max_x, p.cx + p.rx + 10.0f);
      max_y = std::max(max_y, p.cy + p.ry + 10.0f);
      max_z = std::max(max_z, p.cz + p.rz + 10.0f);
    }

    // Clamp to non-negative
    min_x = std::max(0.0f, min_x);
    min_y = std::max(0.0f, min_y);
    min_z = std::max(0.0f, min_z);

    uint32_t nx = static_cast<uint32_t>(std::ceil((max_x - min_x) / grid_spacing_um));
    uint32_t ny = static_cast<uint32_t>(std::ceil((max_y - min_y) / grid_spacing_um));
    uint32_t nz = static_cast<uint32_t>(std::ceil((max_z - min_z) / grid_spacing_um));
    nx = std::max(nx, 2u);
    ny = std::max(ny, 2u);
    nz = std::max(nz, 2u);

    grid.Init(nx, ny, nz, grid_spacing_um);
    grid.origin_x = min_x;
    grid.origin_y = min_y;
    grid.origin_z = min_z;

    // Bake SDF and region IDs
    ch_sdf = grid.AddChannel("sdf");
    ch_region = grid.AddChannel("region_id");
    brain_sdf.BakeToGrid(grid, ch_sdf, ch_region);

    // Initialize neural field
    field.ch_sdf = ch_sdf;
    field.Init(grid);

    // Pre-compute region-to-voxel index map (avoids O(n) scans on escalation).
    {
      size_t n_regions = brain_sdf.primitives.size();
      region_voxels.assign(n_regions, {});
      auto& sdf_data = grid.channels[ch_sdf].data;
      auto& region_data = grid.channels[ch_region].data;
      for (size_t i = 0; i < grid.NumVoxels(); ++i) {
        if (sdf_data[i] < 0.0f) {
          for (size_t r = 0; r < n_regions; ++r) {
            if (std::abs(region_data[i] - static_cast<float>(r)) < 0.5f) {
              region_voxels[r].push_back(i);
              break;  // each voxel belongs to at most one region
            }
          }
        }
      }
    }

    // Register LOD regions from SDF primitives
    lod.region_lods.clear();
    region_pop_map.clear();
    region_chunk_map.clear();
    region_comp_map.clear();
    for (const auto& p : brain_sdf.primitives) {
      LODManager::RegionLOD rl;
      rl.name = p.name;
      rl.center_x = p.cx;
      rl.center_y = p.cy;
      rl.center_z = p.cz;
      rl.current_lod = LODLevel::kContinuum;
      lod.region_lods.push_back(rl);
      region_pop_map.push_back(-1);
      region_chunk_map.push_back(-1);
      region_comp_map.push_back(-1);
    }

    // Init per-voxel neuron count tracker
    voxel_neuron_count.assign(grid.NumVoxels(), 0);

    pop_chunks.clear();
    chunks.clear();
    comp_chunks.clear();
    cross_links.clear();

    // Pre-resolve projection region names to indices (avoids string lookups per timestep)
    resolved_projections.clear();
    for (const auto& proj : projections) {
      int src_idx = FindRegion(proj.from_region);
      int dst_idx = FindRegion(proj.to_region);
      if (src_idx < 0 || dst_idx < 0) continue;
      ResolvedProjection rp;
      rp.src_region = static_cast<uint32_t>(src_idx);
      rp.dst_region = static_cast<uint32_t>(dst_idx);
      rp.density = proj.density;
      resolved_projections.push_back(rp);
    }

    initialized = true;

    Log(LogLevel::kInfo,
        "LODTransition: grid %ux%ux%u (%.0fum spacing), %zu regions, %zu projections",
        nx, ny, nz, grid_spacing_um, brain_sdf.primitives.size(),
        projections.size());
  }

  // Find a region index by name. Returns -1 if not found.
  int FindRegion(const std::string& name) const {
    for (size_t i = 0; i < lod.region_lods.size(); ++i) {
      if (lod.region_lods[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }

  // Check if a region is active (at LOD 1, 2, or 3).
  bool IsRegionActive(uint32_t region_idx) const {
    if (region_idx >= region_chunk_map.size()) return false;
    return region_pop_map[region_idx] >= 0 ||
           region_chunk_map[region_idx] >= 0 ||
           region_comp_map[region_idx] >= 0;
  }

  // Check if a region is at spiking resolution (LOD 2 or 3).
  bool IsRegionSpiking(uint32_t region_idx) const {
    if (region_idx >= region_chunk_map.size()) return false;
    return region_chunk_map[region_idx] >= 0 ||
           region_comp_map[region_idx] >= 0;
  }

  // Get the neuron count for an active region (LOD 2 or LOD 3).
  uint32_t GetRegionNeuronCount(uint32_t region_idx) const {
    int cidx = region_chunk_map[region_idx];
    if (cidx >= 0) return static_cast<uint32_t>(chunks[static_cast<size_t>(cidx)].neurons.n);
    int comp_idx = region_comp_map[region_idx];
    if (comp_idx >= 0) return static_cast<uint32_t>(comp_chunks[static_cast<size_t>(comp_idx)].neurons.n);
    return 0;
  }

  // Get spiked array for an active region (LOD 2 or LOD 3).
  const uint8_t* GetRegionSpiked(uint32_t region_idx) const {
    int cidx = region_chunk_map[region_idx];
    if (cidx >= 0) return chunks[static_cast<size_t>(cidx)].neurons.spiked.data();
    int comp_idx = region_comp_map[region_idx];
    if (comp_idx >= 0) return comp_chunks[static_cast<size_t>(comp_idx)].neurons.spiked.data();
    return nullptr;
  }

  // Get i_syn array for an active region (LOD 2: i_syn; LOD 3: i_syn_soma).
  float* GetRegionISyn(uint32_t region_idx) {
    int cidx = region_chunk_map[region_idx];
    if (cidx >= 0) return chunks[static_cast<size_t>(cidx)].neurons.i_syn.data();
    int comp_idx = region_comp_map[region_idx];
    if (comp_idx >= 0) return comp_chunks[static_cast<size_t>(comp_idx)].neurons.i_syn_soma.data();
    return nullptr;
  }

  // Instantiate cross-chunk projections for a newly escalated region.
  // Checks all defined projections: if this region is the source or target
  // and the other endpoint is already active, creates the cross-chunk link.
  void InstantiateProjections(uint32_t region_idx) {
    for (const auto& proj : projections) {
      int src_idx = FindRegion(proj.from_region);
      int dst_idx = FindRegion(proj.to_region);
      if (src_idx < 0 || dst_idx < 0) continue;

      uint32_t src = static_cast<uint32_t>(src_idx);
      uint32_t dst = static_cast<uint32_t>(dst_idx);

      // This projection involves the newly escalated region?
      if (src != region_idx && dst != region_idx) continue;

      // Both endpoints must be active
      if (!IsRegionActive(src) || !IsRegionActive(dst)) continue;

      // Check if this link already exists
      bool exists = false;
      for (const auto& link : cross_links) {
        if (link.src_region == src && link.dst_region == dst) {
          exists = true;
          break;
        }
      }
      if (exists) continue;

      // Create cross-chunk synapses
      uint32_t n_src = GetRegionNeuronCount(src);
      uint32_t n_dst = GetRegionNeuronCount(dst);
      if (n_src == 0 || n_dst == 0) continue;

      std::mt19937 rng(config.seed + src * 10000 + dst * 100);

      std::vector<uint32_t> pre_vec, post_vec;
      std::vector<float> weight_vec;
      std::vector<uint8_t> nt_vec;

      float density = proj.density;
      std::normal_distribution<float> w_dist(proj.weight_mean, proj.weight_std);

      uint64_t total_pairs = static_cast<uint64_t>(n_src) * n_dst;
      bool use_sparse = (total_pairs > 100000 && density < 0.1f);

      if (use_sparse) {
        // Geometric skip sampling: O(expected_edges) instead of O(n^2)
        double log_comp = std::log(1.0 - static_cast<double>(density));
        int64_t idx = -1;
        while (true) {
          double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
          int64_t skip = static_cast<int64_t>(std::log(u) / log_comp);
          idx += skip + 1;
          if (idx >= static_cast<int64_t>(total_pairs)) break;
          uint32_t pre = static_cast<uint32_t>(idx / n_dst);
          uint32_t post_n = static_cast<uint32_t>(idx % n_dst);
          pre_vec.push_back(pre);
          post_vec.push_back(post_n);
          weight_vec.push_back(std::max(0.01f, w_dist(rng)));
          nt_vec.push_back(proj.nt_type);
        }
      } else {
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        for (uint32_t pre = 0; pre < n_src; ++pre) {
          for (uint32_t post_n = 0; post_n < n_dst; ++post_n) {
            if (coin(rng) >= density) continue;
            pre_vec.push_back(pre);
            post_vec.push_back(post_n);
            weight_vec.push_back(std::max(0.01f, w_dist(rng)));
            nt_vec.push_back(proj.nt_type);
          }
        }
      }

      if (pre_vec.empty()) continue;

      CrossChunkLink link;
      link.src_region = src;
      link.dst_region = dst;
      link.synapses.BuildFromCOO(n_src, pre_vec, post_vec, weight_vec, nt_vec);
      cross_links.push_back(std::move(link));

      Log(LogLevel::kInfo,
          "LODTransition: projection '%s' -> '%s': %zu cross-chunk synapses",
          proj.from_region.c_str(), proj.to_region.c_str(),
          cross_links.back().synapses.Size());
    }
  }

  // Remove all cross-chunk links involving a region about to de-escalate.
  void RemoveProjections(uint32_t region_idx) {
    cross_links.erase(
        std::remove_if(cross_links.begin(), cross_links.end(),
            [region_idx](const CrossChunkLink& link) {
              return link.src_region == region_idx ||
                     link.dst_region == region_idx;
            }),
        cross_links.end());
  }

  // Propagate spikes across all active cross-chunk links.
  // Source spiked[] comes from the source region's chunk.
  // Target i_syn[] goes to the target region's chunk.
  void PropagateCrossChunkSpikes(float weight_scale) {
    for (auto& link : cross_links) {
      const uint8_t* src_spiked = GetRegionSpiked(link.src_region);
      float* dst_i_syn = GetRegionISyn(link.dst_region);
      if (!src_spiked || !dst_i_syn) continue;
      link.synapses.PropagateSpikes(src_spiked, dst_i_syn, weight_scale);
    }
  }

  // Number of active cross-chunk links.
  size_t ActiveCrossLinks() const { return cross_links.size(); }

  // Total cross-chunk synapses across all active links.
  size_t TotalCrossChunkSynapses() const {
    size_t total = 0;
    for (const auto& link : cross_links) total += link.synapses.Size();
    return total;
  }

  // Escalate a region from LOD 0 (neural field) to LOD 1 (population model).
  // Initializes MPR state from the mean E/I activity of covered voxels.
  bool EscalateToPopulation(uint32_t region_idx) {
    if (region_idx >= lod.region_lods.size()) return false;
    if (region_pop_map[region_idx] >= 0) return false;   // already at LOD 1
    if (region_chunk_map[region_idx] >= 0) return false;  // already at LOD 2
    if (region_comp_map[region_idx] >= 0) return false;   // already at LOD 3

    const auto& prim = brain_sdf.primitives[region_idx];

    PopulationChunk pop;
    pop.name = prim.name;
    pop.region_idx = region_idx;
    pop.center_x = prim.cx;
    pop.center_y = prim.cy;
    pop.center_z = prim.cz;
    pop.radius_x = prim.rx;
    pop.radius_y = prim.ry;
    pop.radius_z = prim.rz;
    pop.params = config.population_params;

    // Use pre-computed region-to-voxel map (O(region_voxels) instead of O(total_voxels))
    pop.covered_voxels = region_voxels[region_idx];
    if (pop.covered_voxels.empty()) return false;

    float E_sum = 0.0f, I_sum = 0.0f;
    int n_inside = static_cast<int>(pop.covered_voxels.size());
    for (size_t i : pop.covered_voxels) {
      E_sum += grid.channels[field.ch_e].data[i];
      I_sum += grid.channels[field.ch_i].data[i];
    }

    // Compute mean field activity for initialization
    float E_mean = E_sum / static_cast<float>(n_inside);
    float I_mean = I_sum / static_cast<float>(n_inside);

    // Initialize population state from field
    pop.state = PopulationFromField(E_mean, I_mean, pop.params);

    // Compute nominal neuron count (for later escalation to LOD 2)
    pop.nominal_neurons = static_cast<uint32_t>(
        pop.covered_voxels.size() * config.neurons_per_voxel);
    if (pop.nominal_neurons == 0) pop.nominal_neurons = 1;

    // Store
    size_t pop_idx = pop_chunks.size();
    pop_chunks.push_back(std::move(pop));
    region_pop_map[region_idx] = static_cast<int>(pop_idx);

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kRegion;

    Log(LogLevel::kInfo,
        "LODTransition: escalated '%s' -> LOD 1 (population, E=%.2f I=%.2f)",
        prim.name.c_str(), E_mean, I_mean);

    return true;
  }

  // De-escalate a region from LOD 1 (population) back to LOD 0 (field).
  bool DeEscalateFromPopulation(uint32_t region_idx) {
    if (region_idx >= region_pop_map.size()) return false;
    int pidx = region_pop_map[region_idx];
    if (pidx < 0) return false;

    auto& pop = pop_chunks[static_cast<size_t>(pidx)];
    auto& E_data = grid.channels[field.ch_e].data;
    auto& I_data = grid.channels[field.ch_i].data;

    // Collapse population rates back to field
    FieldValues fv = FieldFromPopulation(pop.state, pop.params);

    for (size_t voxel_global : pop.covered_voxels) {
      E_data[voxel_global] = fv.E;
      I_data[voxel_global] = fv.I;
    }

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kContinuum;

    Log(LogLevel::kInfo,
        "LODTransition: de-escalated '%s' LOD 1 -> LOD 0",
        pop.name.c_str());

    // Remove chunk (swap with last)
    region_pop_map[region_idx] = -1;
    if (static_cast<size_t>(pidx) < pop_chunks.size() - 1) {
      uint32_t swapped_region = pop_chunks.back().region_idx;
      region_pop_map[swapped_region] = pidx;
      pop_chunks[static_cast<size_t>(pidx)] = std::move(pop_chunks.back());
    }
    pop_chunks.pop_back();

    return true;
  }

  // Escalate a region from LOD 1 (population) to LOD 2 (spiking neurons).
  // Spawns neurons initialized from population state instead of field.
  bool EscalateFromPopulation(uint32_t region_idx) {
    if (region_idx >= region_pop_map.size()) return false;
    int pidx = region_pop_map[region_idx];
    if (pidx < 0) return false;
    if (region_chunk_map[region_idx] >= 0) return false;

    auto& pop = pop_chunks[static_cast<size_t>(pidx)];
    const auto& prim = brain_sdf.primitives[region_idx];
    std::mt19937 rng(config.seed + region_idx * 1000);

    NeuronChunk chunk;
    chunk.name = prim.name;
    chunk.region_idx = region_idx;
    chunk.center_x = prim.cx;
    chunk.center_y = prim.cy;
    chunk.center_z = prim.cz;
    chunk.radius_x = prim.rx;
    chunk.radius_y = prim.ry;
    chunk.radius_z = prim.rz;
    chunk.params = config.default_params;

    chunk.covered_voxels = std::move(pop.covered_voxels);
    if (chunk.covered_voxels.empty()) return false;

    uint32_t n_neurons = pop.nominal_neurons;
    chunk.neurons.Resize(n_neurons);
    chunk.neuron_voxel_map.resize(n_neurons);
    chunk.voxel_spike_counts.assign(chunk.covered_voxels.size(), 0);

    // Initialize neuron voltages from population mean voltage + jitter.
    // MPR v_pop gives the mean; spread with Lorentzian-like jitter.
    float v_pop = pop.state.v_e;
    (void)pop.state.r_e;  // used only for logging, not initialization
    std::normal_distribution<float> v_jitter(0.0f, 3.0f);  // +-3mV jitter
    std::uniform_real_distribution<float> pos_jitter(-0.4f, 0.4f);

    uint32_t neurons_per_vox = n_neurons /
        static_cast<uint32_t>(chunk.covered_voxels.size());
    if (neurons_per_vox == 0) neurons_per_vox = 1;
    uint32_t neuron_idx = 0;

    for (size_t vi = 0; vi < chunk.covered_voxels.size() && neuron_idx < n_neurons; ++vi) {
      size_t voxel_global = chunk.covered_voxels[vi];

      uint32_t gx = static_cast<uint32_t>(voxel_global % grid.nx);
      uint32_t gy = static_cast<uint32_t>((voxel_global / grid.nx) % grid.ny);
      uint32_t gz = static_cast<uint32_t>(voxel_global / (grid.nx * grid.ny));

      float wx, wy, wz;
      grid.GridToWorld(gx, gy, gz, wx, wy, wz);

      uint32_t n_here = (vi < chunk.covered_voxels.size() - 1) ?
          neurons_per_vox :
          (n_neurons - neuron_idx);

      for (uint32_t j = 0; j < n_here && neuron_idx < n_neurons; ++j) {
        chunk.neurons.x[neuron_idx] = wx + pos_jitter(rng) * grid.dx;
        chunk.neurons.y[neuron_idx] = wy + pos_jitter(rng) * grid.dx;
        chunk.neurons.z[neuron_idx] = wz + pos_jitter(rng) * grid.dx;

        // Initialize from population state:
        // Map MPR v (dimensionless) to Izhikevich v (mV).
        // MPR v range is roughly [-10, 10], Izh v range [-70, 30].
        // Use: v_izh = v_rest + (v_pop - eta) * scale
        float v_rest = chunk.params.c;
        float v_thresh = chunk.params.v_thresh;
        float v_range = v_thresh - v_rest;
        float v_normalized = std::clamp(
            (v_pop - pop.params.eta_e) / 10.0f, 0.0f, 0.8f);
        float v_init = v_rest + v_normalized * v_range + v_jitter(rng);
        chunk.neurons.v[neuron_idx] = std::clamp(v_init, v_rest - 10.0f, v_thresh);
        chunk.neurons.u[neuron_idx] = chunk.params.b * chunk.neurons.v[neuron_idx];

        chunk.neuron_voxel_map[neuron_idx] = static_cast<uint32_t>(vi);
        neuron_idx++;
      }

      voxel_neuron_count[voxel_global] = static_cast<int>(n_here);
    }

    // Generate synapses (same as Escalate)
    {
      std::vector<uint32_t> pre_vec, post_vec;
      std::vector<float> weight_vec;
      std::vector<uint8_t> nt_vec;

      std::normal_distribution<float> w_dist(config.weight_mean, config.weight_std);
      float density = config.synapse_density;
      uint64_t total_pairs = static_cast<uint64_t>(n_neurons) * n_neurons;
      bool use_sparse = (total_pairs > 100000 && density < 0.1f);

      if (use_sparse) {
        double log_comp = std::log(1.0 - static_cast<double>(density));
        int64_t idx = -1;
        while (true) {
          double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
          int64_t skip = static_cast<int64_t>(std::log(u) / log_comp);
          idx += skip + 1;
          if (idx >= static_cast<int64_t>(total_pairs)) break;
          uint32_t pre = static_cast<uint32_t>(idx / n_neurons);
          uint32_t post_n = static_cast<uint32_t>(idx % n_neurons);
          if (pre == post_n) continue;
          pre_vec.push_back(pre);
          post_vec.push_back(post_n);
          weight_vec.push_back(std::max(0.01f, w_dist(rng)));
          nt_vec.push_back(static_cast<uint8_t>(kACh));
        }
      } else {
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        for (uint32_t pre = 0; pre < n_neurons; ++pre) {
          for (uint32_t post_n = 0; post_n < n_neurons; ++post_n) {
            if (pre == post_n) continue;
            if (coin(rng) >= density) continue;
            pre_vec.push_back(pre);
            post_vec.push_back(post_n);
            weight_vec.push_back(std::max(0.01f, w_dist(rng)));
            nt_vec.push_back(static_cast<uint8_t>(kACh));
          }
        }
      }

      chunk.synapses.BuildFromCOO(n_neurons, pre_vec, post_vec, weight_vec, nt_vec);
    }

    // Identify boundary neurons
    for (uint32_t i = 0; i < n_neurons; ++i) {
      float nx = chunk.neurons.x[i];
      float ny = chunk.neurons.y[i];
      float nz = chunk.neurons.z[i];
      float d = prim.Evaluate(nx, ny, nz);
      if (d > -config.boundary_width_um && d < 0.0f) {
        chunk.boundary_indices.push_back(i);
      }
    }

    // Store neuron chunk
    size_t chunk_idx = chunks.size();
    chunks.push_back(std::move(chunk));
    region_chunk_map[region_idx] = static_cast<int>(chunk_idx);

    // Remove population chunk (swap with last)
    region_pop_map[region_idx] = -1;
    if (static_cast<size_t>(pidx) < pop_chunks.size() - 1) {
      uint32_t swapped_region = pop_chunks.back().region_idx;
      region_pop_map[swapped_region] = pidx;
      pop_chunks[static_cast<size_t>(pidx)] = std::move(pop_chunks.back());
    }
    pop_chunks.pop_back();

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kNeuron;

    Log(LogLevel::kInfo,
        "LODTransition: escalated '%s' LOD 1 -> LOD 2 (%u neurons, %zu synapses)",
        prim.name.c_str(), n_neurons,
        chunks.back().synapses.Size());

    // Instantiate cross-chunk projections
    InstantiateProjections(region_idx);

    return true;
  }

  // De-escalate a region from LOD 2 (spiking) to LOD 1 (population).
  // Computes population statistics from the neuron ensemble.
  bool DeEscalateToPopulation(uint32_t region_idx) {
    if (region_idx >= region_chunk_map.size()) return false;
    int cidx = region_chunk_map[region_idx];
    if (cidx < 0) return false;
    if (region_pop_map[region_idx] >= 0) return false;

    // Remove cross-chunk projections
    RemoveProjections(region_idx);

    auto& chunk = chunks[static_cast<size_t>(cidx)];

    PopulationChunk pop;
    pop.name = chunk.name;
    pop.region_idx = region_idx;
    pop.center_x = chunk.center_x;
    pop.center_y = chunk.center_y;
    pop.center_z = chunk.center_z;
    pop.radius_x = chunk.radius_x;
    pop.radius_y = chunk.radius_y;
    pop.radius_z = chunk.radius_z;
    pop.params = config.population_params;
    pop.covered_voxels = std::move(chunk.covered_voxels);
    pop.nominal_neurons = static_cast<uint32_t>(chunk.neurons.n);

    // Compute population firing rate from neuron spike data
    float dt_ms = 0.1f;  // typical timestep
    if (chunk.step_count > 0) {
      int total_spikes = 0;
      for (auto c : chunk.voxel_spike_counts) total_spikes += c;
      float window_s = chunk.step_count * dt_ms / 1000.0f;
      float rate_hz = static_cast<float>(total_spikes) /
                      (static_cast<float>(chunk.neurons.n) * window_s);
      pop.state.r_e = std::clamp(rate_hz, 0.0f, pop.params.max_rate_hz);
    }

    // Compute mean voltage
    float v_sum = 0.0f;
    for (size_t i = 0; i < chunk.neurons.n; ++i) {
      v_sum += chunk.neurons.v[i];
    }
    float v_mean = v_sum / static_cast<float>(chunk.neurons.n);

    // Map Izhikevich v (mV) to MPR v (dimensionless)
    float v_rest = config.default_params.c;
    float v_thresh = config.default_params.v_thresh;
    float v_normalized = (v_mean - v_rest) / (v_thresh - v_rest);
    pop.state.v_e = pop.params.eta_e + v_normalized * 10.0f;

    // Derive inhibitory rate (approximate from E/I balance)
    pop.state.r_i = pop.state.r_e * 0.3f;  // rough E/I rate ratio
    pop.state.v_i = pop.state.v_e;

    // Steady-state adaptation
    pop.state.a_e = pop.params.alpha_e * pop.params.tau_a * pop.state.r_e;
    pop.state.a_i = pop.params.alpha_i * pop.params.tau_a * pop.state.r_i;

    // Clear voxel neuron counts
    for (size_t voxel_global : pop.covered_voxels) {
      voxel_neuron_count[voxel_global] = 0;
    }

    // Store population chunk
    size_t pop_idx = pop_chunks.size();
    pop_chunks.push_back(std::move(pop));
    region_pop_map[region_idx] = static_cast<int>(pop_idx);

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kRegion;

    Log(LogLevel::kInfo,
        "LODTransition: de-escalated '%s' LOD 2 -> LOD 1 (r_e=%.1f Hz)",
        chunk.name.c_str(), pop_chunks.back().state.r_e);

    // Remove neuron chunk (swap with last)
    region_chunk_map[region_idx] = -1;
    if (static_cast<size_t>(cidx) < chunks.size() - 1) {
      uint32_t swapped_region = chunks.back().region_idx;
      region_chunk_map[swapped_region] = cidx;
      chunks[static_cast<size_t>(cidx)] = std::move(chunks.back());
    }
    chunks.pop_back();

    return true;
  }

  // Escalate a region from LOD 0 (neural field) to LOD 2 (spiking neurons).
  // Returns true on success, false if region is invalid or already escalated.
  bool Escalate(uint32_t region_idx) {
    if (region_idx >= lod.region_lods.size()) return false;
    if (region_pop_map[region_idx] >= 0) return false;   // at LOD 1
    if (region_chunk_map[region_idx] >= 0) return false;  // already at LOD 2
    if (region_comp_map[region_idx] >= 0) return false;   // already at LOD 3

    const auto& prim = brain_sdf.primitives[region_idx];
    std::mt19937 rng(config.seed + region_idx * 1000);

    NeuronChunk chunk;
    chunk.name = prim.name;
    chunk.region_idx = region_idx;
    chunk.center_x = prim.cx;
    chunk.center_y = prim.cy;
    chunk.center_z = prim.cz;
    chunk.radius_x = prim.rx;
    chunk.radius_y = prim.ry;
    chunk.radius_z = prim.rz;
    chunk.params = config.default_params;

    // Use pre-computed region-to-voxel map (O(region_voxels) instead of O(total_voxels))
    chunk.covered_voxels = region_voxels[region_idx];
    if (chunk.covered_voxels.empty()) return false;

    // Determine neuron count
    uint32_t n_neurons = static_cast<uint32_t>(
        chunk.covered_voxels.size() * config.neurons_per_voxel);
    if (n_neurons == 0) n_neurons = 1;

    chunk.neurons.Resize(n_neurons);
    chunk.neuron_voxel_map.resize(n_neurons);
    chunk.voxel_spike_counts.assign(chunk.covered_voxels.size(), 0);

    // Read field E activity for initialization
    auto& E_data = grid.channels[field.ch_e].data;

    // Distribute neurons across covered voxels (round-robin with jitter).
    std::uniform_real_distribution<float> jitter(-0.4f, 0.4f);
    uint32_t neurons_per_vox = n_neurons /
        static_cast<uint32_t>(chunk.covered_voxels.size());
    if (neurons_per_vox == 0) neurons_per_vox = 1;
    uint32_t neuron_idx = 0;

    for (size_t vi = 0; vi < chunk.covered_voxels.size() && neuron_idx < n_neurons; ++vi) {
      size_t voxel_global = chunk.covered_voxels[vi];

      // Convert flat index to 3D grid coordinates
      uint32_t gx = static_cast<uint32_t>(voxel_global % grid.nx);
      uint32_t gy = static_cast<uint32_t>((voxel_global / grid.nx) % grid.ny);
      uint32_t gz = static_cast<uint32_t>(voxel_global / (grid.nx * grid.ny));

      float wx, wy, wz;
      grid.GridToWorld(gx, gy, gz, wx, wy, wz);

      // Read field activity at this voxel
      float E = E_data[voxel_global];

      // Neurons to place in this voxel
      uint32_t n_here = (vi < chunk.covered_voxels.size() - 1) ?
          neurons_per_vox :
          (n_neurons - neuron_idx);  // remainder in last voxel

      for (uint32_t j = 0; j < n_here && neuron_idx < n_neurons; ++j) {
        // Position: voxel center + jitter
        chunk.neurons.x[neuron_idx] = wx + jitter(rng) * grid.dx;
        chunk.neurons.y[neuron_idx] = wy + jitter(rng) * grid.dx;
        chunk.neurons.z[neuron_idx] = wz + jitter(rng) * grid.dx;

        // Initialize membrane potential from field E activity.
        // E in [0,1] maps to v in [v_rest, v_thresh].
        // This ensures spawned neurons start at activity levels consistent
        // with the mean-field representation they replace.
        float v_rest = chunk.params.c;        // resting potential
        float v_thresh = chunk.params.v_thresh;
        float E_clamped = std::clamp(E, 0.0f, 0.8f);  // cap below threshold
        chunk.neurons.v[neuron_idx] = v_rest + E_clamped * (v_thresh - v_rest);
        chunk.neurons.u[neuron_idx] = chunk.params.b * chunk.neurons.v[neuron_idx];

        // Map neuron to its voxel (local index into covered_voxels)
        chunk.neuron_voxel_map[neuron_idx] = static_cast<uint32_t>(vi);

        neuron_idx++;
      }

      // Track per-voxel neuron count (global)
      voxel_neuron_count[voxel_global] = static_cast<int>(n_here);
    }

    // Generate intra-region synapses (Erdos-Renyi random graph).
    // For large chunks, use geometric skip sampling (same as ParametricGenerator).
    {
      std::vector<uint32_t> pre_vec, post_vec;
      std::vector<float> weight_vec;
      std::vector<uint8_t> nt_vec;

      std::uniform_real_distribution<float> coin(0.0f, 1.0f);
      std::normal_distribution<float> w_dist(config.weight_mean, config.weight_std);

      float density = config.synapse_density;
      uint64_t total_pairs = static_cast<uint64_t>(n_neurons) * n_neurons;
      bool use_sparse = (total_pairs > 100000 && density < 0.1f);

      if (use_sparse) {
        double log_comp = std::log(1.0 - static_cast<double>(density));
        int64_t idx = -1;
        while (true) {
          double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
          int64_t skip = static_cast<int64_t>(std::log(u) / log_comp);
          idx += skip + 1;
          if (idx >= static_cast<int64_t>(total_pairs)) break;
          uint32_t pre = static_cast<uint32_t>(idx / n_neurons);
          uint32_t post_n = static_cast<uint32_t>(idx % n_neurons);
          if (pre == post_n) continue;
          pre_vec.push_back(pre);
          post_vec.push_back(post_n);
          weight_vec.push_back(std::max(0.01f, w_dist(rng)));
          nt_vec.push_back(static_cast<uint8_t>(kACh));  // default excitatory
        }
      } else {
        for (uint32_t pre = 0; pre < n_neurons; ++pre) {
          for (uint32_t post_n = 0; post_n < n_neurons; ++post_n) {
            if (pre == post_n) continue;
            if (coin(rng) >= density) continue;
            pre_vec.push_back(pre);
            post_vec.push_back(post_n);
            weight_vec.push_back(std::max(0.01f, w_dist(rng)));
            nt_vec.push_back(static_cast<uint8_t>(kACh));
          }
        }
      }

      chunk.synapses.BuildFromCOO(n_neurons, pre_vec, post_vec, weight_vec, nt_vec);
    }

    // Identify boundary neurons: those near the edge of the region.
    // "Near the edge" = the SDF primitive value at the neuron position
    // is between -boundary_width and 0 (inside but close to surface).
    for (uint32_t i = 0; i < n_neurons; ++i) {
      float nx = chunk.neurons.x[i];
      float ny = chunk.neurons.y[i];
      float nz = chunk.neurons.z[i];
      float d = prim.Evaluate(nx, ny, nz);
      if (d > -config.boundary_width_um && d < 0.0f) {
        chunk.boundary_indices.push_back(i);
      }
    }

    // Store chunk
    size_t chunk_idx = chunks.size();
    chunks.push_back(std::move(chunk));
    region_chunk_map[region_idx] = static_cast<int>(chunk_idx);

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kNeuron;

    Log(LogLevel::kInfo,
        "LODTransition: escalated '%s' -> LOD 2 (%u neurons, %zu synapses, %zu boundary)",
        prim.name.c_str(), n_neurons,
        chunks.back().synapses.Size(),
        chunks.back().boundary_indices.size());

    // Instantiate cross-chunk projections now that this region is active
    InstantiateProjections(region_idx);

    return true;
  }

  // De-escalate a region from LOD 2 back to LOD 0.
  // Collapses neuron spike rates into the neural field.
  bool DeEscalate(uint32_t region_idx) {
    if (region_idx >= region_chunk_map.size()) return false;
    int cidx = region_chunk_map[region_idx];
    if (cidx < 0) return false;

    // Remove cross-chunk projections involving this region
    RemoveProjections(region_idx);

    auto& chunk = chunks[static_cast<size_t>(cidx)];
    auto& E_data = grid.channels[field.ch_e].data;
    auto& I_data = grid.channels[field.ch_i].data;

    // Write mean firing rates back into the neural field.
    for (size_t vi = 0; vi < chunk.covered_voxels.size(); ++vi) {
      size_t voxel_global = chunk.covered_voxels[vi];
      int n_in_voxel = voxel_neuron_count[voxel_global];

      if (n_in_voxel > 0 && chunk.step_count > 0) {
        float rate_hz = chunk.VoxelRate(vi, n_in_voxel, config.default_params.tau_syn_ms > 0 ? 0.1f : 0.1f);
        // Normalize rate to [0,1] for field E
        float E_val = std::clamp(rate_hz / config.max_rate_hz, 0.0f, 1.0f);
        E_data[voxel_global] = E_val;
        // Set I from E/I balance (typical cortical ratio: I ~ 0.4 * E)
        I_data[voxel_global] = E_val * 0.4f;
      }

      // Clear voxel neuron count
      voxel_neuron_count[voxel_global] = 0;
    }

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kContinuum;

    Log(LogLevel::kInfo,
        "LODTransition: de-escalated '%s' -> LOD 0",
        chunk.name.c_str());

    // Remove chunk: swap with last and pop
    region_chunk_map[region_idx] = -1;
    if (static_cast<size_t>(cidx) < chunks.size() - 1) {
      // Update the swapped chunk's region mapping
      uint32_t swapped_region = chunks.back().region_idx;
      region_chunk_map[swapped_region] = cidx;
      chunks[static_cast<size_t>(cidx)] = std::move(chunks.back());
    }
    chunks.pop_back();

    return true;
  }

  // Escalate a region from LOD 2 (spiking) to LOD 3 (compartmental).
  // Converts each Izhikevich point neuron into a 3-compartment model:
  //   soma voltage <- Izhikevich v
  //   apical/basal dendrites <- leak potential (resting)
  //   gating variables <- steady-state at soma voltage
  // Preserves spatial layout, synapse connectivity, and boundary assignments.
  bool EscalateToCompartmental(uint32_t region_idx) {
    if (region_idx >= region_chunk_map.size()) return false;
    int cidx = region_chunk_map[region_idx];
    if (cidx < 0) return false;  // not at LOD 2
    if (region_comp_map[region_idx] >= 0) return false;  // already at LOD 3

    auto& src = chunks[static_cast<size_t>(cidx)];

    CompartmentalChunk comp;
    comp.name = src.name;
    comp.region_idx = region_idx;
    comp.center_x = src.center_x;
    comp.center_y = src.center_y;
    comp.center_z = src.center_z;
    comp.radius_x = src.radius_x;
    comp.radius_y = src.radius_y;
    comp.radius_z = src.radius_z;
    comp.params = config.compartmental_params;

    uint32_t n = static_cast<uint32_t>(src.neurons.n);
    comp.neurons.Resize(n);

    // Transfer positions
    comp.x = std::move(src.neurons.x);
    comp.y = std::move(src.neurons.y);
    comp.z = std::move(src.neurons.z);

    // Convert Izhikevich state to compartmental state.
    // Soma voltage inherits from Izhikevich v.
    // Dendrites start near their leak potential.
    // Gating variables are initialized at steady state for the soma voltage.
    for (uint32_t i = 0; i < n; ++i) {
      float v = src.neurons.v[i];
      comp.neurons.v_soma[i] = v;
      comp.neurons.v_apical[i] = comp.params.apical.E_leak;
      comp.neurons.v_basal[i] = comp.params.basal.E_leak;

      // Gating at steady state for current soma voltage
      comp.neurons.m_Na[i] = mNaInf(v);
      comp.neurons.h_Na[i] = hNaInf(v);
      comp.neurons.n_KDR[i] = nKDRInf(v);

      // Dendritic gating at rest
      float va = comp.params.apical.E_leak;
      comp.neurons.m_CaHVA[i] = mCaHVAInf(va);
      comp.neurons.h_CaHVA[i] = hCaHVAInf(va);
      comp.neurons.m_Ih[i] = mIhInf(va);

      float vb = comp.params.basal.E_leak;
      comp.neurons.m_Na_b[i] = mNaInf(vb);
      comp.neurons.h_Na_b[i] = hNaInf(vb);

      // Spike time continuity
      comp.neurons.last_spike_time[i] = src.neurons.last_spike_time[i];
      comp.neurons.spiked[i] = src.neurons.spiked[i];
    }

    // Transfer spatial metadata
    comp.covered_voxels = std::move(src.covered_voxels);
    comp.neuron_voxel_map = std::move(src.neuron_voxel_map);
    comp.boundary_indices = std::move(src.boundary_indices);
    comp.voxel_spike_counts = std::move(src.voxel_spike_counts);
    comp.step_count = src.step_count;

    // Transfer synapse table (spikes now deliver to soma compartment)
    comp.synapses = std::move(src.synapses);

    // Store compartmental chunk
    size_t comp_idx = comp_chunks.size();
    comp_chunks.push_back(std::move(comp));
    region_comp_map[region_idx] = static_cast<int>(comp_idx);

    // Remove the spiking chunk (swap with last)
    region_chunk_map[region_idx] = -1;
    if (static_cast<size_t>(cidx) < chunks.size() - 1) {
      uint32_t swapped_region = chunks.back().region_idx;
      region_chunk_map[swapped_region] = cidx;
      chunks[static_cast<size_t>(cidx)] = std::move(chunks.back());
    }
    chunks.pop_back();

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kCompartmental;

    Log(LogLevel::kInfo,
        "LODTransition: escalated '%s' LOD 2 -> LOD 3 (%u compartmental neurons)",
        comp_chunks.back().name.c_str(), n);

    return true;
  }

  // De-escalate a region from LOD 3 (compartmental) back to LOD 2 (spiking).
  // Collapses 3-compartment state back to a point neuron:
  //   Izhikevich v <- soma voltage
  //   Izhikevich u <- b * v (steady-state approximation)
  // Preserves spatial layout and synapse connectivity.
  bool DeEscalateFromCompartmental(uint32_t region_idx) {
    if (region_idx >= region_comp_map.size()) return false;
    int cidx = region_comp_map[region_idx];
    if (cidx < 0) return false;

    auto& src = comp_chunks[static_cast<size_t>(cidx)];

    NeuronChunk chunk;
    chunk.name = src.name;
    chunk.region_idx = region_idx;
    chunk.center_x = src.center_x;
    chunk.center_y = src.center_y;
    chunk.center_z = src.center_z;
    chunk.radius_x = src.radius_x;
    chunk.radius_y = src.radius_y;
    chunk.radius_z = src.radius_z;
    chunk.params = config.default_params;

    uint32_t n = static_cast<uint32_t>(src.neurons.n);
    chunk.neurons.Resize(n);

    // Transfer positions back
    chunk.neurons.x = std::move(src.x);
    chunk.neurons.y = std::move(src.y);
    chunk.neurons.z = std::move(src.z);

    // Convert compartmental state back to Izhikevich.
    for (uint32_t i = 0; i < n; ++i) {
      chunk.neurons.v[i] = src.neurons.v_soma[i];
      chunk.neurons.u[i] = chunk.params.b * src.neurons.v_soma[i];
      chunk.neurons.last_spike_time[i] = src.neurons.last_spike_time[i];
      chunk.neurons.spiked[i] = src.neurons.spiked[i];
    }

    // Transfer spatial metadata
    chunk.covered_voxels = std::move(src.covered_voxels);
    chunk.neuron_voxel_map = std::move(src.neuron_voxel_map);
    chunk.boundary_indices = std::move(src.boundary_indices);
    chunk.voxel_spike_counts = std::move(src.voxel_spike_counts);
    chunk.step_count = src.step_count;

    // Transfer synapse table back
    chunk.synapses = std::move(src.synapses);

    // Store spiking chunk
    size_t spike_idx = chunks.size();
    chunks.push_back(std::move(chunk));
    region_chunk_map[region_idx] = static_cast<int>(spike_idx);

    // Remove the compartmental chunk (swap with last)
    region_comp_map[region_idx] = -1;
    if (static_cast<size_t>(cidx) < comp_chunks.size() - 1) {
      uint32_t swapped_region = comp_chunks.back().region_idx;
      region_comp_map[swapped_region] = cidx;
      comp_chunks[static_cast<size_t>(cidx)] = std::move(comp_chunks.back());
    }
    comp_chunks.pop_back();

    // Update LOD level
    lod.region_lods[region_idx].current_lod = LODLevel::kNeuron;

    Log(LogLevel::kInfo,
        "LODTransition: de-escalated '%s' LOD 3 -> LOD 2 (%u spiking neurons)",
        chunks.back().name.c_str(), n);

    return true;
  }

  // De-escalate a compartmental region (LOD 3) all the way back to field (LOD 0).
  // Steps through: LOD 3 -> LOD 2 -> LOD 1 -> LOD 0.
  bool DeEscalateCompartmentalToField(uint32_t region_idx) {
    if (!DeEscalateFromCompartmental(region_idx)) return false;
    if (!DeEscalateToPopulation(region_idx)) return false;
    return DeEscalateFromPopulation(region_idx);
  }

  // Inject neural field activity as external input into LOD 1 populations.
  // Population chunks receive mean E/I field activity from covered voxels.
  void CoupleFieldToPopulations() {
    for (auto& pop : pop_chunks) {
      if (pop.covered_voxels.empty()) continue;
      float E_sum = 0.0f, I_sum = 0.0f;
      for (size_t voxel : pop.covered_voxels) {
        E_sum += grid.channels[field.ch_e].data[voxel];
        I_sum += grid.channels[field.ch_i].data[voxel];
      }
      float n = static_cast<float>(pop.covered_voxels.size());
      // Convert field E/I to external current for the population
      // Scale: E in [0,1] maps to moderate current
      pop.state.I_ext_e += config.field_to_neuron_scale * (E_sum / n);
      pop.state.I_ext_i += config.field_to_neuron_scale * (I_sum / n);
    }
  }

  // Write population firing rates back into the neural field.
  void CouplePopulationsToField() {
    auto& E_data = grid.channels[field.ch_e].data;
    auto& I_data = grid.channels[field.ch_i].data;

    for (const auto& pop : pop_chunks) {
      FieldValues fv = FieldFromPopulation(pop.state, pop.params);
      for (size_t voxel : pop.covered_voxels) {
        E_data[voxel] = fv.E;
        I_data[voxel] = fv.I;
      }
    }
  }

  // Couple LOD 1 population regions to LOD 2/3 spiking regions (and vice versa)
  // via defined projections. When a projection connects a population region to
  // a spiking region, the population rate drives external current into spiking
  // neurons, and spiking rates drive external input into the population.
  void CouplePopulationsToNeurons(float dt_ms) {
    // Use pre-resolved projection indices (no FindRegion string lookups)
    for (const auto& rp : resolved_projections) {
      uint32_t src = rp.src_region;
      uint32_t dst = rp.dst_region;

      int src_pop = region_pop_map[src];
      int dst_pop = region_pop_map[dst];
      int src_chunk = region_chunk_map[src];
      int dst_chunk = region_chunk_map[dst];
      int src_comp = region_comp_map[src];
      int dst_comp = region_comp_map[dst];

      // Population -> Spiking: inject rate-proportional current
      if (src_pop >= 0 && (dst_chunk >= 0 || dst_comp >= 0)) {
        float rate = pop_chunks[static_cast<size_t>(src_pop)].state.r_e;
        float drive = rate * config.neuron_to_field_scale *
                      config.field_to_neuron_scale * rp.density * 100.0f;
        if (dst_chunk >= 0) {
          auto& ch = chunks[static_cast<size_t>(dst_chunk)];
          for (uint32_t bi : ch.boundary_indices) {
            ch.neurons.i_ext[bi] += drive;
          }
        }
        if (dst_comp >= 0) {
          auto& cc = comp_chunks[static_cast<size_t>(dst_comp)];
          for (uint32_t bi : cc.boundary_indices) {
            cc.neurons.i_ext_soma[bi] += drive;
          }
        }
      }

      // Spiking -> Population: inject firing rate as external drive
      if ((src_chunk >= 0 || src_comp >= 0) && dst_pop >= 0) {
        float rate_hz = 0.0f;
        if (src_chunk >= 0) {
          auto& ch = chunks[static_cast<size_t>(src_chunk)];
          if (ch.step_count > 0 && ch.neurons.n > 0) {
            int total_spikes = 0;
            for (auto c : ch.voxel_spike_counts) total_spikes += c;
            float window_s = ch.step_count * dt_ms / 1000.0f;
            rate_hz = static_cast<float>(total_spikes) /
                      (static_cast<float>(ch.neurons.n) * window_s);
          }
        }
        if (src_comp >= 0) {
          auto& cc = comp_chunks[static_cast<size_t>(src_comp)];
          if (cc.step_count > 0 && cc.neurons.n > 0) {
            int total_spikes = 0;
            for (auto c : cc.voxel_spike_counts) total_spikes += c;
            float window_s = cc.step_count * dt_ms / 1000.0f;
            rate_hz = static_cast<float>(total_spikes) /
                      (static_cast<float>(cc.neurons.n) * window_s);
          }
        }
        float drive = rate_hz * config.neuron_to_field_scale *
                      rp.density * 100.0f;
        pop_chunks[static_cast<size_t>(dst_pop)].state.I_ext_e += drive;
      }

      // Population -> Population: rate-based coupling
      if (src_pop >= 0 && dst_pop >= 0) {
        float rate = pop_chunks[static_cast<size_t>(src_pop)].state.r_e;
        float drive = rate * config.neuron_to_field_scale *
                      rp.density * 100.0f;
        pop_chunks[static_cast<size_t>(dst_pop)].state.I_ext_e += drive;
      }
    }
  }

  // Inject neural field activity as external current into boundary neurons.
  // Only affects neurons marked as boundary (near the region edge).
  // Handles both LOD 2 (spiking) and LOD 3 (compartmental) chunks.
  void CoupleFieldToNeurons() {
    // LOD 2 spiking chunks
    for (auto& chunk : chunks) {
      for (uint32_t bi : chunk.boundary_indices) {
        float wx = chunk.neurons.x[bi];
        float wy = chunk.neurons.y[bi];
        float wz = chunk.neurons.z[bi];
        float E = grid.Sample(field.ch_e, wx, wy, wz);
        chunk.neurons.i_ext[bi] += config.field_to_neuron_scale * E;
      }
    }

    // LOD 3 compartmental chunks: inject into soma compartment
    for (auto& comp : comp_chunks) {
      for (uint32_t bi : comp.boundary_indices) {
        float wx = comp.x[bi];
        float wy = comp.y[bi];
        float wz = comp.z[bi];
        float E = grid.Sample(field.ch_e, wx, wy, wz);
        comp.neurons.i_ext_soma[bi] += config.field_to_neuron_scale * E;
      }
    }
  }

  // Write neuron spike rates back into the neural field for covered voxels.
  // This replaces the Wilson-Cowan dynamics in active regions with
  // actual spiking/compartmental neuron data.
  void CoupleNeuronsToField(float dt_ms) {
    auto& E_data = grid.channels[field.ch_e].data;
    auto& I_data = grid.channels[field.ch_i].data;

    // LOD 2 spiking chunks
    for (auto& chunk : chunks) {
      if (chunk.step_count < 1) continue;
      for (size_t vi = 0; vi < chunk.covered_voxels.size(); ++vi) {
        size_t voxel_global = chunk.covered_voxels[vi];
        int n_in_voxel = voxel_neuron_count[voxel_global];
        if (n_in_voxel <= 0) continue;
        float rate_hz = chunk.VoxelRate(vi, n_in_voxel, dt_ms);
        float E_val = std::clamp(rate_hz * config.neuron_to_field_scale,
                                  0.0f, 1.0f);
        E_data[voxel_global] = E_val;
        I_data[voxel_global] = E_val * 0.4f;
      }
    }

    // LOD 3 compartmental chunks
    for (auto& comp : comp_chunks) {
      if (comp.step_count < 1) continue;
      for (size_t vi = 0; vi < comp.covered_voxels.size(); ++vi) {
        size_t voxel_global = comp.covered_voxels[vi];
        int n_in_voxel = voxel_neuron_count[voxel_global];
        if (n_in_voxel <= 0) continue;
        float rate_hz = comp.VoxelRate(vi, n_in_voxel, dt_ms);
        float E_val = std::clamp(rate_hz * config.neuron_to_field_scale,
                                  0.0f, 1.0f);
        E_data[voxel_global] = E_val;
        I_data[voxel_global] = E_val * 0.4f;
      }
    }
  }

  // Full simulation step.
  // 1. Check LOD transitions
  // 2. Step neural field (global backbone)
  // 3. Couple field -> populations, populations <-> neurons, field -> neurons
  // 4. Clear synaptic input, propagate intra-chunk + cross-chunk spikes
  // 5. Step all population and neuron chunks
  // 6. Couple population/neuron rates -> field
  int Step(float dt_ms, float sim_time_ms) {
    // 1. Check LOD transitions (skip if auto_lod disabled)
    int transitions = config.auto_lod ? UpdateLOD() : 0;

    // 2. Step the neural field everywhere
    field.Step(grid, dt_ms);

    // 3a. Couple field -> LOD 1 populations
    CoupleFieldToPopulations();

    // 3b. Couple LOD 1 populations <-> LOD 2/3 neurons (bidirectional)
    CouplePopulationsToNeurons(dt_ms);

    // 3c. Inject field activity into LOD 2/3 boundary neurons
    CoupleFieldToNeurons();

    // 4a. Clear synaptic input for all neuron chunks
    for (auto& chunk : chunks) chunk.neurons.ClearSynapticInput();
    for (auto& comp : comp_chunks) comp.neurons.ClearSynapticInput();

    // 4b. Propagate intra-chunk spikes (within each region)
    for (auto& chunk : chunks) {
      chunk.synapses.PropagateSpikes(
          chunk.neurons.spiked.data(),
          chunk.neurons.i_syn.data(),
          config.weight_scale);
    }
    for (auto& comp : comp_chunks) {
      comp.synapses.PropagateSpikes(
          comp.neurons.spiked.data(),
          comp.neurons.i_syn_soma.data(),
          config.weight_scale);
    }

    // 4c. Propagate cross-chunk spikes (inter-region projections)
    PropagateCrossChunkSpikes(config.weight_scale);

    // 5a. Step LOD 1 population chunks
    for (auto& pop : pop_chunks) {
      PopulationStep(pop.state, dt_ms, pop.params);
    }

    // 5b. Step LOD 2 spiking chunks
    for (auto& chunk : chunks) {
      IzhikevichStep(chunk.neurons, dt_ms, sim_time_ms, chunk.params);
      chunk.AccumulateSpikes();
    }

    // 5c. Step LOD 3 compartmental chunks
    for (auto& comp : comp_chunks) {
      CompartmentalStep(comp.neurons, dt_ms, sim_time_ms, comp.params);
      comp.AccumulateSpikes();
    }

    // 6a. Write LOD 1 population rates back into field
    CouplePopulationsToField();

    // 6b. Write LOD 2/3 neuron rates back into field
    CoupleNeuronsToField(dt_ms);

    // Clear external input for next step
    for (auto& chunk : chunks) chunk.neurons.ClearExternalInput();
    for (auto& comp : comp_chunks) comp.neurons.ClearExternalInput();

    return transitions;
  }

  // Check for LOD transitions based on the LOD manager focus and zones.
  // Handles all transitions across the full 4-level hierarchy:
  //   LOD 0 (field) <-> LOD 1 (population) <-> LOD 2 (spiking) <-> LOD 3 (compartmental)
  //
  // Stepwise transitions preserve state continuity:
  //   Escalation:   0 -> 1 -> 2 -> 3 (each step initializes from previous level)
  //   De-escalation: 3 -> 2 -> 1 -> 0 (each step collapses state to coarser level)
  //   Multi-step jumps are executed as sequential single steps within one call.
  int UpdateLOD() {
    int transitions = 0;
    for (size_t r = 0; r < lod.region_lods.size(); ++r) {
      auto& rl = lod.region_lods[r];
      LODLevel prev = rl.current_lod;
      LODLevel next = lod.GetLODWithHysteresis(
          rl.center_x, rl.center_y, rl.center_z, prev);

      if (next == prev) continue;
      uint32_t ri = static_cast<uint32_t>(r);

      // --- Escalation (coarser -> finer) ---

      // LOD 0 -> LOD 1: field to population
      if (prev == LODLevel::kContinuum && next == LODLevel::kRegion) {
        if (EscalateToPopulation(ri)) transitions++;
      }
      // LOD 0 -> LOD 2: field -> population -> spiking (two steps)
      else if (prev == LODLevel::kContinuum && next == LODLevel::kNeuron) {
        if (EscalateToPopulation(ri)) {
          transitions++;
          if (EscalateFromPopulation(ri)) transitions++;
        }
      }
      // LOD 0 -> LOD 3: field -> population -> spiking -> compartmental
      else if (prev == LODLevel::kContinuum && next == LODLevel::kCompartmental) {
        if (EscalateToPopulation(ri)) {
          transitions++;
          if (EscalateFromPopulation(ri)) {
            transitions++;
            if (EscalateToCompartmental(ri)) transitions++;
          }
        }
      }
      // LOD 1 -> LOD 2: population to spiking
      else if (prev == LODLevel::kRegion && next == LODLevel::kNeuron) {
        if (EscalateFromPopulation(ri)) transitions++;
      }
      // LOD 1 -> LOD 3: population -> spiking -> compartmental
      else if (prev == LODLevel::kRegion && next == LODLevel::kCompartmental) {
        if (EscalateFromPopulation(ri)) {
          transitions++;
          if (EscalateToCompartmental(ri)) transitions++;
        }
      }
      // LOD 2 -> LOD 3: spiking to compartmental
      else if (prev == LODLevel::kNeuron && next == LODLevel::kCompartmental) {
        if (EscalateToCompartmental(ri)) transitions++;
      }

      // --- De-escalation (finer -> coarser) ---

      // LOD 3 -> LOD 2: compartmental to spiking
      else if (prev == LODLevel::kCompartmental && next == LODLevel::kNeuron) {
        if (DeEscalateFromCompartmental(ri)) transitions++;
      }
      // LOD 3 -> LOD 1: compartmental -> spiking -> population
      else if (prev == LODLevel::kCompartmental && next == LODLevel::kRegion) {
        if (DeEscalateFromCompartmental(ri)) {
          transitions++;
          if (DeEscalateToPopulation(ri)) transitions++;
        }
      }
      // LOD 3 -> LOD 0: compartmental -> spiking -> population -> field
      else if (prev == LODLevel::kCompartmental && next == LODLevel::kContinuum) {
        if (DeEscalateFromCompartmental(ri)) {
          transitions++;
          if (DeEscalateToPopulation(ri)) {
            transitions++;
            if (DeEscalateFromPopulation(ri)) transitions++;
          }
        }
      }
      // LOD 2 -> LOD 1: spiking to population
      else if (prev == LODLevel::kNeuron && next == LODLevel::kRegion) {
        if (DeEscalateToPopulation(ri)) transitions++;
      }
      // LOD 2 -> LOD 0: spiking -> population -> field
      else if (prev == LODLevel::kNeuron && next == LODLevel::kContinuum) {
        if (DeEscalateToPopulation(ri)) {
          transitions++;
          if (DeEscalateFromPopulation(ri)) transitions++;
        }
      }
      // LOD 1 -> LOD 0: population to field
      else if (prev == LODLevel::kRegion && next == LODLevel::kContinuum) {
        if (DeEscalateFromPopulation(ri)) transitions++;
      }
    }
    return transitions;
  }

  // Read activity at a world position. Checks LOD 3 compartmental chunks
  // first (highest resolution), then LOD 2 spiking, then neural field.
  float ReadActivity(float wx, float wy, float wz) const {
    // Check LOD 3 compartmental chunks first
    for (const auto& comp : comp_chunks) {
      const auto& prim = brain_sdf.primitives[comp.region_idx];
      float d = prim.Evaluate(wx, wy, wz);
      if (d < 0.0f) {
        float sum = 0.0f;
        int count = 0;
        float r2 = grid.dx * grid.dx;
        for (size_t i = 0; i < comp.neurons.n; ++i) {
          float ddx = comp.x[i] - wx;
          float ddy = comp.y[i] - wy;
          float ddz = comp.z[i] - wz;
          if (ddx*ddx + ddy*ddy + ddz*ddz < r2) {
            sum += CompartmentalActivity(comp.neurons, i, comp.params);
            count++;
          }
        }
        if (count > 0) return sum / static_cast<float>(count);
      }
    }

    // Check LOD 2 spiking chunks
    for (const auto& chunk : chunks) {
      const auto& prim = brain_sdf.primitives[chunk.region_idx];
      float d = prim.Evaluate(wx, wy, wz);
      if (d < 0.0f) {
        float sum = 0.0f;
        int count = 0;
        float r2 = grid.dx * grid.dx;
        for (size_t i = 0; i < chunk.neurons.n; ++i) {
          float ddx = chunk.neurons.x[i] - wx;
          float ddy = chunk.neurons.y[i] - wy;
          float ddz = chunk.neurons.z[i] - wz;
          if (ddx*ddx + ddy*ddy + ddz*ddz < r2) {
            sum += chunk.neurons.v[i];
            count++;
          }
        }
        if (count > 0) {
          float v_mean = sum / static_cast<float>(count);
          float v_rest = chunk.params.c;
          float v_thresh = chunk.params.v_thresh;
          return std::clamp((v_mean - v_rest) / (v_thresh - v_rest), 0.0f, 1.0f);
        }
      }
    }

    // Check LOD 1 population chunks: return normalized rate as activity.
    for (const auto& pop : pop_chunks) {
      const auto& prim = brain_sdf.primitives[pop.region_idx];
      float d = prim.Evaluate(wx, wy, wz);
      if (d < 0.0f) {
        return std::clamp(pop.state.r_e / pop.params.max_rate_hz, 0.0f, 1.0f);
      }
    }

    // Not in any active region: read from neural field
    return field.ReadActivity(grid, wx, wy, wz);
  }

  // Total neuron count across all active chunks (LOD 2 + LOD 3).
  size_t TotalActiveNeurons() const {
    size_t total = 0;
    for (const auto& chunk : chunks) total += chunk.neurons.n;
    for (const auto& comp : comp_chunks) total += comp.neurons.n;
    return total;
  }

  // Nominal neuron count across all active population chunks (LOD 1).
  size_t TotalPopulationNeurons() const {
    size_t total = 0;
    for (const auto& pop : pop_chunks) total += pop.nominal_neurons;
    return total;
  }

  // Total spikes this step across all chunks (LOD 2 + LOD 3).
  int TotalSpikes() const {
    int total = 0;
    for (const auto& chunk : chunks) total += chunk.CountSpikes();
    for (const auto& comp : comp_chunks) total += comp.CountSpikes();
    return total;
  }

  // Number of active LOD 1 population regions.
  size_t ActivePopulations() const { return pop_chunks.size(); }

  // Number of active LOD 2 spiking regions.
  size_t ActiveChunks() const { return chunks.size(); }

  // Number of active LOD 3 compartmental regions.
  size_t ActiveCompartmentalChunks() const { return comp_chunks.size(); }
};

}  // namespace mechabrain

#endif  // FWMC_LOD_TRANSITION_H_
