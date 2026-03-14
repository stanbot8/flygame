#ifndef PARABRAIN_NEUROMODULATOR_FIELD_H_
#define PARABRAIN_NEUROMODULATOR_FIELD_H_

// Volumetric neuromodulator diffusion fields.
//
// Instead of tracking DA/5HT/OA per-neuron with simple decay,
// this models neuromodulators as 3D diffusion fields on a VoxelGrid.
// Neurons that release neuromodulators inject into their local voxel;
// the field diffuses through the volume; all neurons sample from
// their spatial position.
//
// This is biologically accurate: neuromodulators operate via volume
// transmission: they're released from varicosities and diffuse
// through the neuropil, affecting all neurons within range.
//
// Architecture follows the UWYN pattern from SkiBiDy (Skin BioDynaMo):
// agents (neurons) read from continuous fields (diffusion grids),
// and only inject into fields when they spike.
//
// Diffusion parameters from literature:
//   DA diffusion in brain tissue: D ~2.4 um^2/ms (Nicholson 1995)
//   DA decay (reuptake + MAO): tau ~50-200ms (Cragg & Rice 2004)
//   5HT diffusion: D ~1.7 um^2/ms (Bunin & Wightman 1998)
//   5HT decay: tau ~200-1000ms (Bunin et al. 1998)
//   OA diffusion: estimated ~2.0 um^2/ms (similar molecular weight to NE)
//   OA decay: tau ~500ms (estimated, Roeder 2005)
//
// References:
//   Nicholson 1995 (extracellular diffusion)
//   Cragg & Rice 2004 (DA volume transmission)
//   Bunin & Wightman 1998 (5HT diffusion in brain)
//   Venton et al. 2003 (DA release radius ~7 um)
//   Rice & Cragg 2008 (volume vs synaptic transmission)

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "tissue/voxel_grid.h"

namespace mechabrain {

struct NeuromodulatorFieldConfig {
  // Grid resolution. Coarser = faster, finer = more accurate.
  // 10 um is a good default: captures release plumes (~7 um radius)
  // while keeping the grid small for a fly brain (50x30x20 = 30K voxels).
  float voxel_spacing_um = 10.0f;

  // Diffusion coefficients (um^2/ms) from literature
  float da_diffusion = 2.4f;    // Nicholson 1995
  float sht_diffusion = 1.7f;   // Bunin & Wightman 1998
  float oa_diffusion = 2.0f;    // estimated (similar to NE)

  // Decay rates (1/ms). tau = 1/decay.
  float da_decay = 0.01f;       // tau ~100ms (Cragg & Rice 2004)
  float sht_decay = 0.002f;     // tau ~500ms (Bunin et al. 1998)
  float oa_decay = 0.002f;      // tau ~500ms (estimated, Roeder 2005)

  // Release amount per spike (concentration units deposited in voxel)
  float da_release_per_spike = 0.5f;
  float sht_release_per_spike = 0.3f;
  float oa_release_per_spike = 0.4f;

  // Brain extents in um (Drosophila defaults)
  float brain_width_um = 500.0f;
  float brain_height_um = 300.0f;
  float brain_depth_um = 200.0f;
  float origin_x_um = 0.0f;
  float origin_y_um = 0.0f;
  float origin_z_um = 0.0f;
};

struct NeuromodulatorField {
  VoxelGrid grid;
  size_t ch_da = 0;   // dopamine channel index
  size_t ch_sht = 0;  // serotonin channel index
  size_t ch_oa = 0;   // octopamine channel index

  // Cached neuron-to-voxel mapping (avoids repeated WorldToGrid lookups)
  std::vector<uint32_t> neuron_voxel_x;
  std::vector<uint32_t> neuron_voxel_y;
  std::vector<uint32_t> neuron_voxel_z;
  std::vector<size_t> neuron_flat_idx;  // flat voxel index for direct lookup
  std::vector<bool> neuron_in_grid;

  // Which NT types map to which neuromodulator field
  // DA neurons: NTType::kDA (3)
  // 5HT neurons: NTType::k5HT (4)
  // OA neurons: NTType::kOA (5)

  void Init(const NeuromodulatorFieldConfig& cfg, const NeuronArray& neurons) {
    float sp = cfg.voxel_spacing_um;
    uint32_t w = static_cast<uint32_t>(std::ceil(cfg.brain_width_um / sp));
    uint32_t h = static_cast<uint32_t>(std::ceil(cfg.brain_height_um / sp));
    uint32_t d = static_cast<uint32_t>(std::ceil(cfg.brain_depth_um / sp));

    grid.Init(w, h, d, sp);
    grid.origin_x = cfg.origin_x_um;
    grid.origin_y = cfg.origin_y_um;
    grid.origin_z = cfg.origin_z_um;

    ch_da  = grid.AddChannel("dopamine",   cfg.da_diffusion,  cfg.da_decay);
    ch_sht = grid.AddChannel("serotonin",  cfg.sht_diffusion, cfg.sht_decay);
    ch_oa  = grid.AddChannel("octopamine", cfg.oa_diffusion,  cfg.oa_decay);

    // Pre-compute neuron-to-voxel mapping
    uint32_t n = static_cast<uint32_t>(neurons.n);
    neuron_voxel_x.resize(n);
    neuron_voxel_y.resize(n);
    neuron_voxel_z.resize(n);
    neuron_flat_idx.resize(n, 0);
    neuron_in_grid.resize(n, false);

    for (uint32_t i = 0; i < n; ++i) {
      // NeuronArray stores positions in nm; VoxelGrid uses um
      float wx = neurons.x[i] / 1000.0f;
      float wy = neurons.y[i] / 1000.0f;
      float wz = neurons.z[i] / 1000.0f;
      uint32_t gx, gy, gz;
      if (grid.WorldToGrid(wx, wy, wz, gx, gy, gz)) {
        neuron_voxel_x[i] = gx;
        neuron_voxel_y[i] = gy;
        neuron_voxel_z[i] = gz;
        neuron_flat_idx[i] = grid.Idx(gx, gy, gz);
        neuron_in_grid[i] = true;
      }
    }
  }

  // Convenience: init with default config and assign random positions
  // within the brain volume (for parametric circuits without real coordinates).
  void InitWithRandomPositions(const NeuromodulatorFieldConfig& cfg,
                                NeuronArray& neurons, uint32_t seed = 42) {
    // Assign random positions within brain volume
    std::mt19937 rng(seed);
    for (uint32_t i = 0; i < neurons.n; ++i) {
      // Position in nm (NeuronArray convention)
      neurons.x[i] = std::uniform_real_distribution<float>(
          cfg.origin_x_um * 1000.0f,
          (cfg.origin_x_um + cfg.brain_width_um) * 1000.0f)(rng);
      neurons.y[i] = std::uniform_real_distribution<float>(
          cfg.origin_y_um * 1000.0f,
          (cfg.origin_y_um + cfg.brain_height_um) * 1000.0f)(rng);
      neurons.z[i] = std::uniform_real_distribution<float>(
          cfg.origin_z_um * 1000.0f,
          (cfg.origin_z_um + cfg.brain_depth_um) * 1000.0f)(rng);
    }
    Init(cfg, neurons);
  }

  // Step 1: Inject neuromodulator release from spiking neurons into the grid.
  // Call after spike detection, before diffusion.
  void InjectRelease(const NeuronArray& neurons, const SynapseTable& synapses,
                     const NeuromodulatorFieldConfig& cfg) {
    for (uint32_t i = 0; i < neurons.n; ++i) {
      if (!neurons.spiked[i] || !neuron_in_grid[i]) continue;

      size_t voxel = grid.Idx(neuron_voxel_x[i], neuron_voxel_y[i],
                               neuron_voxel_z[i]);

      // Check neuron's NT type to determine what it releases
      // DA neurons release dopamine, etc.
      // We check the outgoing synapse NT type for this neuron.
      uint32_t syn_start = synapses.row_ptr[i];
      uint32_t syn_end = synapses.row_ptr[i + 1];
      if (syn_start >= syn_end) continue;

      uint8_t nt = synapses.nt_type[syn_start];
      if (nt == 3)  // kDA
        grid.channels[ch_da].data[voxel] += cfg.da_release_per_spike;
      else if (nt == 4)  // k5HT
        grid.channels[ch_sht].data[voxel] += cfg.sht_release_per_spike;
      else if (nt == 5)  // kOA
        grid.channels[ch_oa].data[voxel] += cfg.oa_release_per_spike;
    }
  }

  // Step 2: Diffuse and decay all neuromodulator fields.
  void Diffuse(float dt_ms) {
    grid.Diffuse(dt_ms);
  }

  // Step 3: Sample fields back to per-neuron concentrations.
  // Uses cached flat voxel indices for O(1) direct lookup (nearest-neighbor)
  // instead of full trilinear interpolation with repeated nm->um conversion.
  void SampleToNeurons(NeuronArray& neurons) const {
    const float* da_data  = grid.channels[ch_da].data.data();
    const float* sht_data = grid.channels[ch_sht].data.data();
    const float* oa_data  = grid.channels[ch_oa].data.data();

    for (uint32_t i = 0; i < neurons.n; ++i) {
      if (!neuron_in_grid[i]) continue;
      size_t idx = neuron_flat_idx[i];
      neurons.dopamine[i]  = da_data[idx];
      neurons.serotonin[i] = sht_data[idx];
      neurons.octopamine[i] = oa_data[idx];
    }
  }

  // All-in-one update: inject from spikes, diffuse, sample back.
  // Replaces the per-neuron NeuromodulatorUpdate() with volumetric diffusion.
  void Update(NeuronArray& neurons, const SynapseTable& synapses,
              float dt_ms, const NeuromodulatorFieldConfig& cfg) {
    InjectRelease(neurons, synapses, cfg);
    Diffuse(dt_ms);
    SampleToNeurons(neurons);
  }

  // Read a channel value at a world position (for external queries/visualization)
  float SampleDA(float wx_um, float wy_um, float wz_um) const {
    return grid.Sample(ch_da, wx_um, wy_um, wz_um);
  }
  float SampleSHT(float wx_um, float wy_um, float wz_um) const {
    return grid.Sample(ch_sht, wx_um, wy_um, wz_um);
  }
  float SampleOA(float wx_um, float wy_um, float wz_um) const {
    return grid.Sample(ch_oa, wx_um, wy_um, wz_um);
  }
};

}  // namespace mechabrain

#endif  // PARABRAIN_NEUROMODULATOR_FIELD_H_
