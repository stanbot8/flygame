#ifndef FWMC_LOD_CONFIG_H_
#define FWMC_LOD_CONFIG_H_

// Configuration and support types for the LOD transition engine.

#include <cstdint>
#include <string>

#include "core/compartmental_neuron.h"
#include "core/izhikevich.h"
#include "core/synapse_table.h"
#include "tissue/population_model.h"

namespace mechabrain {

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
  float field_to_neuron_scale = 10.0f;

  // Coupling scale: neuron firing rate (Hz) -> field E activity [0,1].
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

}  // namespace mechabrain

#endif  // FWMC_LOD_CONFIG_H_
