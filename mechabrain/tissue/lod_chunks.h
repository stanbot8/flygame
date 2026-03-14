#ifndef FWMC_LOD_CHUNKS_H_
#define FWMC_LOD_CHUNKS_H_

// Chunk representations for each LOD level in the multi-scale brain simulator.
//
// Each chunk owns the simulation state for one brain region at a specific
// level of detail:
//   NeuronChunk       (LOD 2): Izhikevich spiking neurons with CSR synapses
//   CompartmentalChunk (LOD 3): 3-compartment pyramidal neurons
//   PopulationChunk   (LOD 1): Montbrio-Pazo-Roxin mean-field model

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/compartmental_neuron.h"
#include "core/izhikevich.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "tissue/population_model.h"

namespace mechabrain {

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

  // Compute mean firing rate (Hz) for a covered voxel.
  float VoxelRate(size_t voxel_local_idx, int neurons_in_voxel,
                  float dt_ms) const {
    if (neurons_in_voxel <= 0 || step_count <= 0) return 0.0f;
    float window_s = step_count * dt_ms / 1000.0f;
    return static_cast<float>(voxel_spike_counts[voxel_local_idx]) /
           (static_cast<float>(neurons_in_voxel) * window_s);
  }

  int CountSpikes() const { return neurons.CountSpikes(); }
};

// A chunk of compartmental neurons for one region at LOD 3.
// Provides multi-compartment biophysical detail (soma + apical + basal
// dendrites with active channels, NMDA, Ca2+ dynamics, bAPs).
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
  uint32_t nominal_neurons = 0;

  // E/I ratio (fraction of neurons that are excitatory)
  float ei_ratio = 0.8f;  // 80% E, 20% I (Drosophila and cortical default)
};

}  // namespace mechabrain

#endif  // FWMC_LOD_CHUNKS_H_
