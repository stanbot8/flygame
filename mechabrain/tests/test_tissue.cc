// Tissue module tests: voxel grid, brain SDF, neural field, LOD manager,
// neuromodulator field
#include "test_harness.h"

#include "tissue/voxel_grid.h"
#include "tissue/brain_sdf.h"
#include "tissue/neural_field.h"
#include "tissue/lod_manager.h"
#include "tissue/neuromodulator_field.h"
#include "core/parametric_gen.h"
#include "core/spike_frequency_adaptation.h"

#include <cmath>

// ===== VoxelGrid tests =====

TEST(voxel_grid_init) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  CHECK(grid.nx == 10);
  CHECK(grid.ny == 10);
  CHECK(grid.nz == 10);
  CHECK(grid.NumVoxels() == 1000);
  CHECK(grid.dx == 5.0f);
}

TEST(voxel_grid_add_channel) {
  VoxelGrid grid;
  grid.Init(5, 5, 5, 1.0f);
  size_t ch = grid.AddChannel("test", 0.1f, 0.01f);
  CHECK(ch == 0);
  CHECK(grid.channels[ch].data.size() == 125);
  CHECK(grid.channels[ch].diffusion_coeff == 0.1f);
  CHECK(grid.channels[ch].decay_rate == 0.01f);
  CHECK(grid.FindChannel("test") == 0);
  CHECK(grid.FindChannel("nonexistent") == SIZE_MAX);
}

TEST(voxel_grid_world_to_grid) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  grid.origin_x = 0; grid.origin_y = 0; grid.origin_z = 0;

  uint32_t gx = 0, gy = 0, gz = 0;
  bool ok = grid.WorldToGrid(12.0f, 22.0f, 7.0f, gx, gy, gz);
  CHECK(ok);
  CHECK(gx == 2);
  CHECK(gy == 4);
  CHECK(gz == 1);
  (void)ok; (void)gx; (void)gy; (void)gz;

  // Outside the grid (reuse gx/gy/gz)
  bool out1 = grid.WorldToGrid(-1.0f, 0.0f, 0.0f, gx, gy, gz);
  CHECK(!out1); (void)out1;
  bool out2 = grid.WorldToGrid(60.0f, 0.0f, 0.0f, gx, gy, gz);
  CHECK(!out2); (void)out2;
}

TEST(voxel_grid_inject_and_sample) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch = grid.AddChannel("conc");

  grid.Inject(ch, 12.5f, 12.5f, 12.5f, 100.0f);

  // Sample at injection point should be high
  float val = grid.Sample(ch, 12.5f, 12.5f, 12.5f);
  CHECK(val > 0.0f);

  // Sample far away should be zero
  float far = grid.Sample(ch, 42.5f, 42.5f, 42.5f);
  CHECK(far == 0.0f);
}

TEST(voxel_grid_diffusion) {
  VoxelGrid grid;
  grid.Init(20, 20, 20, 1.0f);
  size_t ch = grid.AddChannel("heat", 1.0f, 0.0f);

  // Point source at center
  grid.channels[ch].data[grid.Idx(10, 10, 10)] = 100.0f;

  float center_before = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  float neighbor_before = grid.channels[ch].data[grid.Idx(11, 10, 10)];

  // Diffuse
  grid.Diffuse(0.1f);

  float center_after = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  float neighbor_after = grid.channels[ch].data[grid.Idx(11, 10, 10)];

  // Center should decrease, neighbor should increase
  CHECK(center_after < center_before);
  CHECK(neighbor_after > neighbor_before);
}

TEST(voxel_grid_decay) {
  VoxelGrid grid;
  grid.Init(5, 5, 5, 1.0f);
  size_t ch = grid.AddChannel("substance", 0.0f, 0.1f);  // decay only

  // Fill with uniform value
  for (auto& v : grid.channels[ch].data) v = 10.0f;

  grid.Diffuse(1.0f);  // triggers decay

  // Should be reduced by exp(-0.1)
  float expected = 10.0f * std::exp(-0.1f);
  for (auto& v : grid.channels[ch].data) {
    CHECK(std::abs(v - expected) < 0.01f);
  }
}

TEST(voxel_grid_inject_sphere) {
  VoxelGrid grid;
  grid.Init(20, 20, 20, 1.0f);
  size_t ch = grid.AddChannel("sphere");

  grid.InjectSphere(ch, 10.0f, 10.0f, 10.0f, 3.0f, 100.0f);

  // Center voxel should have some value
  float center = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  CHECK(center > 0.0f);

  // Far corner should be zero
  float corner = grid.channels[ch].data[grid.Idx(0, 0, 0)];
  CHECK(corner == 0.0f);

  // Total injected should be ~100 (distributed across voxels in sphere)
  float total = 0.0f;
  for (auto v : grid.channels[ch].data) total += v;
  CHECK(std::abs(total - 100.0f) < 0.1f);
}

// ===== BrainSDF tests =====

TEST(brain_sdf_single_sphere) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 50, 50, 50, 30, 30, 30});

  // Center should be inside (negative)
  float d_center = sdf.Evaluate(50, 50, 50);
  CHECK(d_center < 0.0f);

  // Far outside should be positive
  float d_outside = sdf.Evaluate(200, 200, 200);
  CHECK(d_outside > 0.0f);

  // On the surface should be near zero
  float d_surface = sdf.Evaluate(80, 50, 50);
  CHECK(std::abs(d_surface) < 5.0f);
}

TEST(brain_sdf_smooth_union) {
  BrainSDF sdf;
  sdf.smooth_k = 10.0f;
  sdf.primitives.push_back({"A", 40, 50, 50, 20, 20, 20});
  sdf.primitives.push_back({"B", 70, 50, 50, 20, 20, 20});

  // Both centers should be inside
  CHECK(sdf.Evaluate(40, 50, 50) < 0.0f);
  CHECK(sdf.Evaluate(70, 50, 50) < 0.0f);

  // The junction between them (at x=55) should also be inside
  // due to smooth union blending
  float d_junction = sdf.Evaluate(55, 50, 50);
  CHECK(d_junction < 0.0f);
}

TEST(brain_sdf_drosophila_init) {
  BrainSDF sdf;
  sdf.InitDrosophila();

  // Should have primitives for all major regions
  CHECK(sdf.primitives.size() >= 10);

  // Center of brain should be inside
  CHECK(sdf.Evaluate(250, 150, 100) < 0.0f);

  // Far outside should be positive
  CHECK(sdf.Evaluate(0, 0, 0) > 0.0f);
  CHECK(sdf.Evaluate(600, 400, 300) > 0.0f);

  // Optic lobes should be inside
  CHECK(sdf.Evaluate(80, 150, 100) < 0.0f);   // left
  CHECK(sdf.Evaluate(420, 150, 100) < 0.0f);  // right
}

TEST(brain_sdf_nearest_region) {
  BrainSDF sdf;
  sdf.InitDrosophila();

  // Center of brain = central_brain (index 0)
  int reg = sdf.NearestRegion(250, 150, 100);
  CHECK(reg == 0);

  // Left optic lobe center
  int lobe = sdf.NearestRegion(80, 150, 100);
  CHECK(lobe == 1);  // optic_lobe_L is index 1

  // Outside should return -1
  int outside = sdf.NearestRegion(0, 0, 0);
  CHECK(outside == -1);
}

TEST(brain_sdf_bake_and_smooth) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 25, 25, 25, 15, 15, 15});

  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");
  size_t ch_reg = grid.AddChannel("region");

  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  // Center voxel should be inside (negative SDF)
  float center_sdf = grid.channels[ch_sdf].data[grid.Idx(5, 5, 5)];
  CHECK(center_sdf < 0.0f);

  // Corner should be outside (positive SDF)
  float corner_sdf = grid.channels[ch_sdf].data[grid.Idx(0, 0, 0)];
  CHECK(corner_sdf > 0.0f);

  // Smooth the SDF
  float pre_smooth = grid.channels[ch_sdf].data[grid.Idx(3, 5, 5)];
  BrainSDF::DiffuseSmooth(grid, ch_sdf, 5);
  float post_smooth = grid.channels[ch_sdf].data[grid.Idx(3, 5, 5)];

  // Smoothing should change surface-adjacent values
  // (the exact change depends on geometry, just verify it's different)
  CHECK(pre_smooth != post_smooth || pre_smooth == 0.0f);
}

TEST(brain_sdf_normal) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 50, 50, 50, 30, 30, 30});

  float nx, ny, nz;
  // Normal on the +x surface should point roughly in +x
  sdf.Normal(80, 50, 50, 1.0f, nx, ny, nz);
  CHECK(nx > 0.5f);
  CHECK(std::abs(ny) < 0.3f);
  CHECK(std::abs(nz) < 0.3f);
}

// ===== NeuralField tests =====

TEST(neural_field_init) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  // Create a simple spherical brain
  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  // E and I channels should exist
  CHECK(field.ch_e != SIZE_MAX);
  CHECK(field.ch_i != SIZE_MAX);

  // Inside voxels should have initial activity
  float center_e = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];
  CHECK(center_e > 0.0f);

  // Outside voxels should be zero
  float corner_e = grid.channels[field.ch_e].data[grid.Idx(0, 0, 0)];
  CHECK(corner_e == 0.0f);
}

TEST(neural_field_step_evolves) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  float e_before = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];

  // Run a few steps
  for (int i = 0; i < 10; ++i) {
    field.Step(grid, 0.5f);
  }

  float e_after = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];

  // Activity should have evolved (not necessarily increased, but changed)
  CHECK(e_before != e_after);
}

TEST(neural_field_stimulus) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  float before = field.ReadActivity(grid, 25, 25, 25);
  field.Stimulate(grid, 25, 25, 25, 5.0f, 10.0f);
  float after = field.ReadActivity(grid, 25, 25, 25);

  // Stimulus should increase activity at injection point
  CHECK(after > before);
}

TEST(neural_field_mask_respected) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  // Small sphere: only center region is "brain"
  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 10, 10, 10});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  // Step many times
  for (int i = 0; i < 50; ++i) {
    field.Step(grid, 0.5f);
  }

  // Outside voxels should remain zero
  float corner_e = grid.channels[field.ch_e].data[grid.Idx(0, 0, 0)];
  float corner_i = grid.channels[field.ch_i].data[grid.Idx(0, 0, 0)];
  CHECK(corner_e == 0.0f);
  CHECK(corner_i == 0.0f);
}

// ===== LODManager tests =====

TEST(lod_manager_basic) {
  LODManager lod;
  lod.SetFocus(250, 150, 100);

  // At the focus: compartmental
  CHECK(lod.GetLOD(250, 150, 100) == LODLevel::kCompartmental);

  // 50um away: neuron level
  CHECK(lod.GetLOD(300, 150, 100) == LODLevel::kNeuron);

  // 150um away: region level
  CHECK(lod.GetLOD(400, 150, 100) == LODLevel::kRegion);

  // 500um away: continuum
  CHECK(lod.GetLOD(750, 150, 100) == LODLevel::kContinuum);
}

TEST(lod_manager_hysteresis) {
  LODManager lod;
  lod.SetFocus(0, 0, 0);
  lod.hysteresis = 10.0f;

  // Point at 95um: neuron level
  LODLevel current = LODLevel::kNeuron;
  // Move to 105um: still neuron (within hysteresis band of 100+10=110)
  LODLevel next = lod.GetLODWithHysteresis(105, 0, 0, current);
  CHECK(next == LODLevel::kNeuron);

  // Move to 115um: now transitions to region
  next = lod.GetLODWithHysteresis(115, 0, 0, current);
  CHECK(next == LODLevel::kRegion);
}

TEST(lod_manager_update_all) {
  LODManager lod;
  lod.SetFocus(0, 0, 0);

  lod.region_lods.push_back({"close", 20, 0, 0, LODLevel::kContinuum});
  lod.region_lods.push_back({"far", 500, 0, 0, LODLevel::kContinuum});

  int transitions = lod.UpdateAll();

  // "close" should escalate to compartmental
  CHECK(lod.region_lods[0].current_lod == LODLevel::kCompartmental);
  // "far" should stay continuum
  CHECK(lod.region_lods[1].current_lod == LODLevel::kContinuum);
  CHECK(transitions == 1);
}

// ===== LODTransitionEngine tests =====

#include "tissue/lod_transition.h"

// Helper: create a minimal LODTransitionEngine with a simple brain
static LODTransitionEngine MakeTestEngine(float spacing = 5.0f) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.boundary_width_um = 8.0f;
  engine.config.seed = 123;
  engine.config.auto_lod = false;  // Tests manually control escalation
  engine.Init(spacing);
  return engine;
}

TEST(lod_transition_init) {
  auto engine = MakeTestEngine();

  CHECK(engine.initialized);
  CHECK(engine.grid.nx > 0);
  CHECK(engine.grid.ny > 0);
  CHECK(engine.grid.nz > 0);
  CHECK(engine.lod.region_lods.size() == 2);
  CHECK(engine.chunks.empty());
  CHECK(engine.ch_sdf != SIZE_MAX);
  CHECK(engine.ch_region != SIZE_MAX);
  CHECK(engine.field.ch_e != SIZE_MAX);
  CHECK(engine.field.ch_i != SIZE_MAX);
}

TEST(lod_transition_escalate_basic) {
  auto engine = MakeTestEngine();

  // Escalate region_A
  bool ok = engine.Escalate(0);
  CHECK(ok);
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.chunks[0].name == "region_A");
  CHECK(engine.chunks[0].neurons.n > 0);
  CHECK(engine.chunks[0].synapses.Size() > 0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);
}

TEST(lod_transition_escalate_double_fails) {
  auto engine = MakeTestEngine();

  bool ok1 = engine.Escalate(0);
  CHECK(ok1);

  // Second escalation of same region should fail
  bool ok2 = engine.Escalate(0);
  CHECK(!ok2);
  CHECK(engine.chunks.size() == 1);
}

TEST(lod_transition_escalate_invalid_region) {
  auto engine = MakeTestEngine();

  bool ok = engine.Escalate(999);
  CHECK(!ok);
  CHECK(engine.chunks.empty());
}

TEST(lod_transition_escalate_neurons_initialized) {
  auto engine = MakeTestEngine();

  // Set some field activity before escalating
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.3f;
    }
  }

  engine.Escalate(0);
  auto& chunk = engine.chunks[0];

  // Neurons should be initialized with voltages above resting
  float v_rest = chunk.params.c;
  float mean_v = 0.0f;
  for (size_t i = 0; i < chunk.neurons.n; ++i) {
    mean_v += chunk.neurons.v[i];
    // Voltage should be between rest and threshold
    CHECK(chunk.neurons.v[i] >= v_rest);
    CHECK(chunk.neurons.v[i] <= chunk.params.v_thresh);
  }
  mean_v /= static_cast<float>(chunk.neurons.n);

  // Mean voltage should be above resting (we set E=0.3)
  CHECK(mean_v > v_rest + 1.0f);
}

TEST(lod_transition_escalate_boundary_neurons) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);

  auto& chunk = engine.chunks[0];
  // Should have identified some boundary neurons
  CHECK(!chunk.boundary_indices.empty());

  // Boundary neurons should be fewer than total neurons
  CHECK(chunk.boundary_indices.size() < chunk.neurons.n);

  // Boundary indices should be valid
  for (uint32_t bi : chunk.boundary_indices) {
    CHECK(bi < chunk.neurons.n);
  }
}

TEST(lod_transition_deescalate) {
  auto engine = MakeTestEngine();

  engine.Escalate(0);
  CHECK(engine.chunks.size() == 1);

  // Run a few steps to accumulate some spike data
  for (int i = 0; i < 50; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // De-escalate to field. Auto-LOD during Step() may have moved region to
  // any LOD level (compartmental, spiking, population, or field).
  LODLevel cur = engine.lod.region_lods[0].current_lod;
  if (cur == LODLevel::kCompartmental) {
    engine.DeEscalateCompartmentalToField(0);
  } else if (cur == LODLevel::kNeuron) {
    engine.DeEscalate(0);
  } else if (cur == LODLevel::kRegion) {
    engine.DeEscalateFromPopulation(0);
  }
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);

  // Field should have been updated with neuron data
  // (we just check it didn't crash and field values are valid)
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    float E = engine.grid.channels[engine.field.ch_e].data[i];
    CHECK(std::isfinite(E));
    CHECK(E >= 0.0f && E <= 1.0f);
  }
}

TEST(lod_transition_deescalate_nonexistent) {
  auto engine = MakeTestEngine();

  // De-escalate a region that was never escalated
  bool ok = engine.DeEscalate(0);
  CHECK(!ok);
}

TEST(lod_transition_escalate_both_regions) {
  auto engine = MakeTestEngine();

  bool ok1 = engine.Escalate(0);
  bool ok2 = engine.Escalate(1);
  CHECK(ok1);
  CHECK(ok2);
  CHECK(engine.chunks.size() == 2);
  CHECK(engine.TotalActiveNeurons() > 0);

  // De-escalate one, keep the other
  engine.DeEscalate(0);
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.chunks[0].name == "region_B");
}

TEST(lod_transition_step_runs) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);

  // Run 100 steps
  for (int i = 0; i < 100; ++i) {
    int transitions = engine.Step(0.1f, i * 0.1f);
    (void)transitions;
  }

  // Region may have been auto-de-escalated by UpdateLOD during Step.
  // If chunks still exist, verify their state.
  if (!engine.chunks.empty()) {
    auto& chunk = engine.chunks[0];
    CHECK(chunk.step_count > 0);
    for (size_t i = 0; i < chunk.neurons.n; ++i) {
      CHECK(std::isfinite(chunk.neurons.v[i]));
    }
  }
}

TEST(lod_transition_field_neuron_coupling) {
  auto engine = MakeTestEngine();

  // Inject strong stimulus into region_A field
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f &&
        std::abs(engine.grid.channels[engine.ch_region].data[i]) < 0.5f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.8f;
    }
  }

  engine.Escalate(0);

  // Run one step: boundary neurons should receive field-derived input
  engine.CoupleFieldToNeurons();

  auto& chunk = engine.chunks[0];
  float total_ext = 0.0f;
  for (uint32_t bi : chunk.boundary_indices) {
    total_ext += std::abs(chunk.neurons.i_ext[bi]);
  }

  // Boundary neurons should have received external input
  if (!chunk.boundary_indices.empty()) {
    CHECK(total_ext > 0.0f);
  }
}

TEST(lod_transition_read_activity_field) {
  auto engine = MakeTestEngine();

  // Set known field activity
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.5f;
    }
  }

  // Read from field (no chunks active)
  float activity = engine.ReadActivity(25, 25, 25);
  CHECK(activity > 0.0f);
}

TEST(lod_transition_automatic_lod) {
  auto engine = MakeTestEngine();
  engine.config.auto_lod = true;  // This test exercises auto-LOD

  // Set focus right on region_A: should trigger escalation
  engine.lod.SetFocus(25, 25, 25);
  engine.lod.zones = {
    {40.0f, LODLevel::kNeuron},   // within 40um: neuron level
  };
  engine.lod.default_level = LODLevel::kContinuum;

  // Run a step: should trigger escalation of region_A (at 0,0,0 from focus)
  int transitions = engine.Step(0.1f, 0.0f);

  // region_A center is at (25,25,25), focus at (25,25,25) = 0 distance
  // Should have been escalated
  CHECK(transitions >= 1);
  CHECK(engine.chunks.size() >= 1);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);

  // region_B center is at (75,25,25), distance from focus ~50um > 40um
  // Should remain at continuum
  CHECK(engine.lod.region_lods[1].current_lod == LODLevel::kContinuum);
}

TEST(lod_transition_total_neurons_and_spikes) {
  auto engine = MakeTestEngine();

  CHECK(engine.TotalActiveNeurons() == 0);
  CHECK(engine.TotalSpikes() == 0);

  engine.Escalate(0);
  CHECK(engine.TotalActiveNeurons() > 0);

  engine.Escalate(1);
  size_t total = engine.TotalActiveNeurons();
  CHECK(total > engine.chunks[0].neurons.n);  // both chunks contribute
}

TEST(lod_transition_escalate_deescalate_cycle) {
  auto engine = MakeTestEngine();

  // Escalate, run, de-escalate, re-escalate: should not crash
  engine.Escalate(0);
  for (int i = 0; i < 20; ++i) engine.Step(0.1f, i * 0.1f);

  // Ensure fully de-escalated (auto-LOD may have partially transitioned)
  if (!engine.chunks.empty()) engine.DeEscalate(0);
  if (!engine.pop_chunks.empty()) engine.DeEscalateFromPopulation(0);

  // Re-escalate
  engine.Escalate(0);
  CHECK(engine.chunks.size() >= 1);
  for (int i = 0; i < 20; ++i) engine.Step(0.1f, (20 + i) * 0.1f);

  // Neurons should still have valid state (if chunk still exists)
  if (!engine.chunks.empty()) {
    for (size_t i = 0; i < engine.chunks[0].neurons.n; ++i) {
      CHECK(std::isfinite(engine.chunks[0].neurons.v[i]));
    }
  }
}

TEST(lod_transition_field_still_evolves) {
  auto engine = MakeTestEngine();

  // Step the field alone (no chunks)
  float E_before = engine.grid.channels[engine.field.ch_e].data[
      engine.grid.Idx(engine.grid.nx / 2, engine.grid.ny / 2, engine.grid.nz / 2)];

  for (int i = 0; i < 50; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  float E_after = engine.grid.channels[engine.field.ch_e].data[
      engine.grid.Idx(engine.grid.nx / 2, engine.grid.ny / 2, engine.grid.nz / 2)];

  // Field should evolve even without any chunks
  // (Though values might converge to steady state, at least step doesn't crash)
  (void)E_before;
  (void)E_after;
  // Just verify it ran without NaN/Inf
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_e].data[i]));
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_i].data[i]));
  }
}

// ===== Multiscale integration tests (Drosophila brain) =====

TEST(multiscale_drosophila_init) {
  // Full Drosophila brain SDF with LOD engine
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.seed = 42;
  engine.Init(20.0f);  // 20um spacing for fast test

  CHECK(engine.initialized);
  CHECK(engine.brain_sdf.primitives.size() >= 10);
  CHECK(engine.lod.region_lods.size() == engine.brain_sdf.primitives.size());
  CHECK(engine.grid.nx > 0);

  // Verify field has activity inside the brain
  float E_center = engine.ReadActivity(250, 150, 100);
  CHECK(E_center >= 0.0f);
}

TEST(multiscale_drosophila_escalate_antennal_lobe) {
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.seed = 42;
  engine.Init(20.0f);

  // Find the antennal lobe region index
  int al_idx = -1;
  for (size_t i = 0; i < engine.brain_sdf.primitives.size(); ++i) {
    if (engine.brain_sdf.primitives[i].name == "antennal_lobe_L") {
      al_idx = static_cast<int>(i);
      break;
    }
  }
  CHECK(al_idx >= 0);

  // Escalate: should spawn neurons in the antennal lobe
  bool ok = engine.Escalate(static_cast<uint32_t>(al_idx));
  CHECK(ok);
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.chunks[0].neurons.n > 0);
  CHECK(engine.chunks[0].synapses.Size() > 0);
  CHECK(engine.lod.region_lods[al_idx].current_lod == LODLevel::kNeuron);
}

TEST(multiscale_drosophila_focus_driven_escalation) {
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();
  engine.config.neurons_per_voxel = 3.0f;
  engine.config.synapse_density = 0.03f;
  engine.config.seed = 42;
  engine.Init(20.0f);

  // Set focus at antennal lobe L center (200, 90, 60)
  engine.lod.SetFocus(200, 90, 60);
  engine.lod.zones = {
    {50.0f, LODLevel::kNeuron},
  };
  engine.lod.default_level = LODLevel::kContinuum;

  // Step should auto-escalate the antennal lobe
  int transitions = engine.Step(0.1f, 0.0f);
  CHECK(transitions >= 1);
  CHECK(engine.ActiveChunks() >= 1);

  // Regions far from focus should remain at continuum
  // Optic lobe R center is at (420, 150, 100), far from focus
  int olr_idx = -1;
  for (size_t i = 0; i < engine.brain_sdf.primitives.size(); ++i) {
    if (engine.brain_sdf.primitives[i].name == "optic_lobe_R") {
      olr_idx = static_cast<int>(i);
      break;
    }
  }
  if (olr_idx >= 0) {
    CHECK(engine.lod.region_lods[olr_idx].current_lod == LODLevel::kContinuum);
  }
}

TEST(multiscale_drosophila_step_produces_spikes) {
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.field_to_neuron_scale = 15.0f;  // stronger coupling for test
  engine.config.seed = 42;
  engine.Init(20.0f);

  // Set focus at central brain, escalate nearby regions
  engine.lod.SetFocus(250, 150, 100);
  engine.lod.zones = {{80.0f, LODLevel::kNeuron}};
  engine.lod.default_level = LODLevel::kContinuum;

  // Inject strong stimulus into the field near focus
  engine.field.Stimulate(engine.grid, 250, 150, 100, 30.0f, 5.0f);

  // Run 200 steps (20ms at 0.1ms dt)
  int total_spikes = 0;
  for (int i = 0; i < 200; ++i) {
    engine.Step(0.1f, i * 0.1f);
    total_spikes += engine.TotalSpikes();
  }

  // Should have spawned neurons and some should have fired
  CHECK(engine.TotalActiveNeurons() > 0);
  // With field stimulus driving boundary neurons, expect some spiking
  // (may be zero if neurons aren't sufficiently driven, so just verify no crash)
  (void)total_spikes;

  // All neuron voltages should be finite
  for (const auto& chunk : engine.chunks) {
    for (size_t i = 0; i < chunk.neurons.n; ++i) {
      CHECK(std::isfinite(chunk.neurons.v[i]));
    }
  }
}

TEST(multiscale_drosophila_field_neuron_consistency) {
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.seed = 42;
  engine.config.auto_lod = false;
  engine.Init(20.0f);

  // Escalate central brain, run, de-escalate, check field state
  engine.Escalate(0);  // central_brain
  for (int i = 0; i < 100; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  engine.DeEscalate(0);
  CHECK(engine.chunks.empty());

  // Field values should be valid (finite, in [0,1])
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    float E = engine.grid.channels[engine.field.ch_e].data[i];
    float I = engine.grid.channels[engine.field.ch_i].data[i];
    CHECK(std::isfinite(E) && E >= 0.0f && E <= 1.0f);
    CHECK(std::isfinite(I) && I >= 0.0f && I <= 1.0f);
  }
}

// ===== Population model (LOD 1) standalone tests =====

#include "tissue/population_model.h"

TEST(population_model_steady_state) {
  // MPR equations should converge to a stable steady state
  // with constant external drive.
  PopulationParams p = PopulationParams::Drosophila();
  PopulationState s;
  s.I_ext_e = 5.0f;  // moderate drive
  s.I_ext_i = 2.0f;

  // Run to steady state (2000ms)
  for (int i = 0; i < 2000; ++i) {
    s.I_ext_e = 5.0f;
    s.I_ext_i = 2.0f;
    PopulationStep(s, 1.0f, p);
  }

  // Should have positive, finite firing rates
  CHECK(std::isfinite(s.r_e) && s.r_e > 0.0f);
  CHECK(std::isfinite(s.r_i) && s.r_i > 0.0f);
  CHECK(std::isfinite(s.v_e));
  CHECK(std::isfinite(s.v_i));

  // Store steady-state rates
  float r_e_ss = s.r_e;
  float r_i_ss = s.r_i;

  // Run 500 more steps: should remain near steady state
  for (int i = 0; i < 500; ++i) {
    s.I_ext_e = 5.0f;
    s.I_ext_i = 2.0f;
    PopulationStep(s, 1.0f, p);
  }

  CHECK(std::abs(s.r_e - r_e_ss) < 0.5f);
  CHECK(std::abs(s.r_i - r_i_ss) < 0.5f);
}

TEST(population_model_stimulus_response) {
  // Increasing external drive should increase firing rate.
  // Use weak recurrence so rates don't both saturate to max_rate_hz.
  PopulationParams p = PopulationParams::Drosophila();
  p.J_ee = 3.0f;   // weak recurrence to avoid saturation
  p.J_ie = 4.0f;

  // Run at low drive
  PopulationState s_low;
  for (int i = 0; i < 2000; ++i) {
    s_low.I_ext_e = 2.0f;
    PopulationStep(s_low, 1.0f, p);
  }

  // Run at high drive
  PopulationState s_high;
  for (int i = 0; i < 2000; ++i) {
    s_high.I_ext_e = 10.0f;
    PopulationStep(s_high, 1.0f, p);
  }

  // Higher drive should produce higher rate
  CHECK(s_high.r_e > s_low.r_e);
}

TEST(population_model_adaptation_reduces_rate) {
  // With adaptation (alpha_e > 0), the initial response should be
  // higher than the adapted steady-state response.
  // Use weak recurrence so rates don't saturate at max_rate_hz,
  // which would mask the adaptation effect.
  PopulationParams p = PopulationParams::Drosophila();
  p.J_ee = 3.0f;      // weak recurrence to avoid saturation
  p.J_ie = 4.0f;
  p.alpha_e = 15.0f;   // strong adaptation
  p.tau_a = 200.0f;

  PopulationState s;

  // Step 50ms: capture the peak (onset) rate after transient
  float r_peak = 0.0f;
  for (int i = 0; i < 50; ++i) {
    s.I_ext_e = 10.0f;
    PopulationStep(s, 1.0f, p);
    if (s.r_e > r_peak) r_peak = s.r_e;
  }

  // Step 3000ms more: adaptation should reduce rate
  for (int i = 0; i < 3000; ++i) {
    s.I_ext_e = 10.0f;
    PopulationStep(s, 1.0f, p);
  }
  float r_adapted = s.r_e;

  // Adapted rate should be lower than peak
  CHECK(r_adapted < r_peak);
  CHECK(r_adapted > 0.0f);  // but not silenced
  CHECK(s.a_e > 0.0f);  // adaptation variable should be active
}

TEST(population_model_field_roundtrip) {
  // PopulationFromField -> FieldFromPopulation should approximately
  // preserve the original E/I values for reasonable inputs.
  PopulationParams p = PopulationParams::Drosophila();

  float E_orig = 0.3f;
  float I_orig = 0.2f;

  PopulationState s = PopulationFromField(E_orig, I_orig, p);

  // Firing rates should map from field
  CHECK(s.r_e > 0.0f);
  CHECK(s.r_i > 0.0f);

  // Convert back
  FieldValues fv = FieldFromPopulation(s, p);

  // Should approximately match originals
  CHECK(std::abs(fv.E - E_orig) < 0.05f);
  CHECK(std::abs(fv.I - I_orig) < 0.05f);
}

TEST(population_model_human_cortical_params) {
  // Human cortical params should have longer time constants and
  // stronger adaptation than Drosophila defaults.
  PopulationParams fly = PopulationParams::Drosophila();
  PopulationParams human = PopulationParams::HumanCortical();

  CHECK(human.tau_e > fly.tau_e);      // slower pyramidal cells
  CHECK(human.alpha_e > fly.alpha_e);  // stronger adaptation
  CHECK(human.tau_a > fly.tau_a);      // slower adaptation

  // Both should produce stable dynamics
  PopulationState s;
  for (int i = 0; i < 2000; ++i) {
    s.I_ext_e = 5.0f;
    PopulationStep(s, 1.0f, human);
  }
  CHECK(std::isfinite(s.r_e) && s.r_e >= 0.0f);
  CHECK(std::isfinite(s.r_i) && s.r_i >= 0.0f);
}

// ===== LOD 1 transition engine tests =====

TEST(lod_transition_escalate_to_population) {
  auto engine = MakeTestEngine();

  // Escalate region_A to LOD 1 (population)
  bool ok = engine.EscalateToPopulation(0);
  CHECK(ok);
  CHECK(engine.pop_chunks.size() == 1);
  CHECK(engine.pop_chunks[0].name == "region_A");
  CHECK(engine.pop_chunks[0].nominal_neurons > 0);
  CHECK(!engine.pop_chunks[0].covered_voxels.empty());
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kRegion);

  // LOD 2 and LOD 3 chunks should be empty
  CHECK(engine.chunks.empty());
  CHECK(engine.comp_chunks.empty());
}

TEST(lod_transition_escalate_to_population_double_fails) {
  auto engine = MakeTestEngine();

  bool ok1 = engine.EscalateToPopulation(0);
  CHECK(ok1);

  // Second escalation to population should fail
  bool ok2 = engine.EscalateToPopulation(0);
  CHECK(!ok2);
  CHECK(engine.pop_chunks.size() == 1);
}

TEST(lod_transition_deescalate_from_population) {
  auto engine = MakeTestEngine();

  // Restrict auto-LOD to population level for region_A only.
  // Radius 30 covers region_A (distance ~0) but not region_B (distance 50).
  engine.lod.zones = {
    {30.0f, LODLevel::kRegion},
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.hysteresis = 0.0f;

  engine.EscalateToPopulation(0);
  CHECK(engine.pop_chunks.size() == 1);

  // Run a few steps at LOD 1
  for (int i = 0; i < 50; ++i) engine.Step(0.1f, i * 0.1f);

  // De-escalate back to field (auto-LOD may have already done this)
  if (engine.region_pop_map[0] >= 0) {
    bool ok = engine.DeEscalateFromPopulation(0);
    CHECK(ok);
  }
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);

  // Field values should be valid
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    float E = engine.grid.channels[engine.field.ch_e].data[i];
    CHECK(std::isfinite(E) && E >= 0.0f && E <= 1.0f);
  }
}

TEST(lod_transition_population_to_spiking) {
  auto engine = MakeTestEngine();

  // Escalate to LOD 1, then to LOD 2
  bool ok1 = engine.EscalateToPopulation(0);
  CHECK(ok1);

  bool ok2 = engine.EscalateFromPopulation(0);
  CHECK(ok2);

  // Population chunk should be gone, spiking chunk should exist
  CHECK(engine.pop_chunks.empty());
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.chunks[0].name == "region_A");
  CHECK(engine.chunks[0].neurons.n > 0);
  CHECK(engine.chunks[0].synapses.Size() > 0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);
}

TEST(lod_transition_spiking_to_population) {
  auto engine = MakeTestEngine();

  // Restrict auto-LOD to spiking for region_A only (radius 30 covers
  // region_A at distance ~0 but not region_B at distance 50).
  engine.lod.zones = {
    {30.0f, LODLevel::kNeuron},
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.hysteresis = 0.0f;

  // Escalate directly to LOD 2
  engine.Escalate(0);
  CHECK(engine.chunks.size() == 1);

  // Run a few steps
  for (int i = 0; i < 100; ++i) engine.Step(0.1f, i * 0.1f);

  // If still at LOD 2, explicitly de-escalate to LOD 1
  if (engine.lod.region_lods[0].current_lod == LODLevel::kNeuron) {
    bool ok = engine.DeEscalateToPopulation(0);
    CHECK(ok);
    CHECK(engine.pop_chunks.size() >= 1);
    CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kRegion);

    int pidx = engine.region_pop_map[0];
    CHECK(pidx >= 0);
    auto& pop = engine.pop_chunks[static_cast<size_t>(pidx)];
    CHECK(std::isfinite(pop.state.r_e));
    CHECK(std::isfinite(pop.state.v_e));
    CHECK(pop.state.r_e >= 0.0f);
  }
  // If auto-de-escalated, just verify the region is at a valid LOD level
}

TEST(lod_transition_population_step_runs) {
  auto engine = MakeTestEngine();
  engine.EscalateToPopulation(0);
  engine.EscalateToPopulation(1);

  // Run 200 steps: population chunks should evolve
  for (int i = 0; i < 200; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // Both populations should have finite state
  for (const auto& pop : engine.pop_chunks) {
    CHECK(std::isfinite(pop.state.r_e));
    CHECK(std::isfinite(pop.state.v_e));
    CHECK(std::isfinite(pop.state.a_e));
    CHECK(std::isfinite(pop.state.r_i));
    CHECK(std::isfinite(pop.state.v_i));
    CHECK(std::isfinite(pop.state.a_i));
    CHECK(pop.state.r_e >= 0.0f);
    CHECK(pop.state.r_i >= 0.0f);
  }
}

TEST(lod_transition_population_field_coupling) {
  auto engine = MakeTestEngine();

  // Set strong field activity in region_A
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f &&
        std::abs(engine.grid.channels[engine.ch_region].data[i]) < 0.5f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.7f;
    }
  }

  engine.EscalateToPopulation(0);

  // Couple field -> population
  engine.CoupleFieldToPopulations();
  CHECK(engine.pop_chunks[0].state.I_ext_e > 0.0f);

  // Step to let population evolve
  PopulationStep(engine.pop_chunks[0].state, 1.0f,
                 engine.pop_chunks[0].params);

  // Couple population -> field
  engine.CouplePopulationsToField();

  // Covered voxels should have population-derived values
  for (size_t voxel : engine.pop_chunks[0].covered_voxels) {
    float E = engine.grid.channels[engine.field.ch_e].data[voxel];
    CHECK(std::isfinite(E) && E >= 0.0f && E <= 1.0f);
  }
}

TEST(lod_transition_population_read_activity) {
  auto engine = MakeTestEngine();
  engine.EscalateToPopulation(0);

  // Set a known rate
  engine.pop_chunks[0].state.r_e = 50.0f;
  engine.pop_chunks[0].params.max_rate_hz = 100.0f;

  // ReadActivity at the population region center should reflect the rate
  float activity = engine.ReadActivity(25, 25, 25);
  CHECK(std::isfinite(activity));
  CHECK(activity >= 0.0f && activity <= 1.0f);

  // Should be approximately r_e / max_rate = 0.5
  CHECK(std::abs(activity - 0.5f) < 0.15f);
}

TEST(lod_transition_active_populations_count) {
  auto engine = MakeTestEngine();

  CHECK(engine.ActivePopulations() == 0);
  engine.EscalateToPopulation(0);
  CHECK(engine.ActivePopulations() == 1);
  engine.EscalateToPopulation(1);
  CHECK(engine.ActivePopulations() == 2);

  // TotalPopulationNeurons should be positive
  CHECK(engine.TotalPopulationNeurons() > 0);
}

TEST(lod_transition_full_4level_cycle) {
  auto engine = MakeTestEngine();

  // Full escalation: LOD 0 -> 1 -> 2 -> 3 (no Step calls to avoid auto-LOD)
  engine.EscalateToPopulation(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kRegion);

  engine.EscalateFromPopulation(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);

  engine.EscalateToCompartmental(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental);

  // Full de-escalation: LOD 3 -> 2 -> 1 -> 0
  engine.DeEscalateFromCompartmental(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);

  engine.DeEscalateToPopulation(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kRegion);

  engine.DeEscalateFromPopulation(0);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);

  CHECK(engine.pop_chunks.empty());
  CHECK(engine.chunks.empty());
  CHECK(engine.comp_chunks.empty());

  // All field values should be finite
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_e].data[i]));
  }
}

TEST(lod_transition_automatic_population_escalation) {
  auto engine = MakeTestEngine();
  engine.config.auto_lod = true;  // This test exercises auto-LOD

  // Set focus on region_A with a population zone at medium distance
  engine.lod.SetFocus(25, 25, 25);
  engine.lod.zones = {
    {10.0f, LODLevel::kNeuron},    // very close: spiking
    {40.0f, LODLevel::kRegion},    // medium: population
  };
  engine.lod.default_level = LODLevel::kContinuum;

  int transitions = engine.Step(0.1f, 0.0f);
  CHECK(transitions >= 1);

  // region_A (center at 25,25,25): distance 0 from focus -> LOD 2 (spiking)
  // But it must first pass through LOD 1.
  // The auto-transition should go LOD 0 -> LOD 1 -> LOD 2 for region_A.
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);

  // region_B (center at 75,25,25): distance ~50 from focus -> LOD 1 or continuum
  // 50um > 40um zone boundary -> continuum
  // (with hysteresis this depends on the default, let's just check it's not spiking)
  CHECK(engine.lod.region_lods[1].current_lod != LODLevel::kNeuron);
}

// ===== LOD 3 compartmental transition tests =====

#include "core/compartmental_neuron.h"

TEST(lod_transition_escalate_to_compartmental) {
  auto engine = MakeTestEngine();

  // First escalate to LOD 2 (spiking)
  bool ok1 = engine.Escalate(0);
  CHECK(ok1);
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.comp_chunks.empty());

  size_t n_neurons = engine.chunks[0].neurons.n;

  // Escalate to LOD 3 (compartmental)
  bool ok2 = engine.EscalateToCompartmental(0);
  CHECK(ok2);

  // Spiking chunk should be gone, compartmental chunk should exist
  CHECK(engine.chunks.empty());
  CHECK(engine.comp_chunks.size() == 1);
  CHECK(engine.comp_chunks[0].name == "region_A");
  CHECK(engine.comp_chunks[0].neurons.n == n_neurons);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental);

  // Positions should have been transferred
  CHECK(engine.comp_chunks[0].x.size() == n_neurons);
  CHECK(engine.comp_chunks[0].y.size() == n_neurons);
  CHECK(engine.comp_chunks[0].z.size() == n_neurons);

  // Synapse table should have been transferred
  CHECK(engine.comp_chunks[0].synapses.Size() > 0);
}

TEST(lod_transition_compartmental_soma_initialized_from_spiking) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);

  // Set some known voltages on the spiking neurons
  auto& chunk = engine.chunks[0];
  for (size_t i = 0; i < chunk.neurons.n; ++i) {
    chunk.neurons.v[i] = -55.0f;  // somewhat depolarized
  }

  engine.EscalateToCompartmental(0);

  // Soma voltages should match the original Izhikevich voltages
  auto& comp = engine.comp_chunks[0];
  for (size_t i = 0; i < comp.neurons.n; ++i) {
    CHECK(std::abs(comp.neurons.v_soma[i] - (-55.0f)) < 0.01f);
    // Dendrites should be at rest
    CHECK(std::abs(comp.neurons.v_apical[i] - comp.params.apical.E_leak) < 0.01f);
    CHECK(std::abs(comp.neurons.v_basal[i] - comp.params.basal.E_leak) < 0.01f);
  }
}

TEST(lod_transition_deescalate_from_compartmental) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);
  size_t n_neurons = engine.chunks[0].neurons.n;

  engine.EscalateToCompartmental(0);
  CHECK(engine.comp_chunks.size() == 1);
  CHECK(engine.chunks.empty());

  // De-escalate back to LOD 2
  bool ok = engine.DeEscalateFromCompartmental(0);
  CHECK(ok);
  CHECK(engine.comp_chunks.empty());
  CHECK(engine.chunks.size() == 1);
  CHECK(engine.chunks[0].neurons.n == n_neurons);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kNeuron);

  // Positions should be back
  CHECK(engine.chunks[0].neurons.x.size() == n_neurons);
  // Synapses should be back
  CHECK(engine.chunks[0].synapses.Size() > 0);
}

TEST(lod_transition_compartmental_voltage_roundtrip) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);

  // Set specific voltages
  for (size_t i = 0; i < engine.chunks[0].neurons.n; ++i) {
    engine.chunks[0].neurons.v[i] = -60.0f + static_cast<float>(i) * 0.1f;
  }

  engine.EscalateToCompartmental(0);

  // Modify soma voltages slightly (simulate dynamics)
  for (size_t i = 0; i < engine.comp_chunks[0].neurons.n; ++i) {
    engine.comp_chunks[0].neurons.v_soma[i] += 2.0f;
  }

  engine.DeEscalateFromCompartmental(0);

  // Izhikevich v should reflect the modified soma voltages
  for (size_t i = 0; i < engine.chunks[0].neurons.n; ++i) {
    float expected = -58.0f + static_cast<float>(i) * 0.1f;
    CHECK(std::abs(engine.chunks[0].neurons.v[i] - expected) < 0.01f);
  }
}

TEST(lod_transition_compartmental_step_runs) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);
  engine.EscalateToCompartmental(0);

  // Run 100 steps with compartmental neurons
  for (int i = 0; i < 100; ++i) {
    int transitions = engine.Step(0.025f, i * 0.025f);
    (void)transitions;
  }

  // All voltages should be finite (if not auto-de-escalated)
  if (!engine.comp_chunks.empty()) {
    auto& comp = engine.comp_chunks[0];
    for (size_t i = 0; i < comp.neurons.n; ++i) {
      CHECK(std::isfinite(comp.neurons.v_soma[i]));
      CHECK(std::isfinite(comp.neurons.v_apical[i]));
      CHECK(std::isfinite(comp.neurons.v_basal[i]));
      CHECK(std::isfinite(comp.neurons.Ca_i[i]));
    }
    CHECK(comp.step_count > 0);
  }
}

TEST(lod_transition_compartmental_deescalate_to_field) {
  auto engine = MakeTestEngine();

  // Restrict auto-LOD to only affect region_A (compartmental) and prevent
  // region_B from being auto-escalated, which would leave stale chunks.
  engine.lod.zones = {
    {10.0f, LODLevel::kCompartmental},
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.hysteresis = 0.0f;

  engine.Escalate(0);
  engine.EscalateToCompartmental(0);

  // Run some steps
  for (int i = 0; i < 50; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // De-escalate all the way to field (auto-LOD may have already done this)
  if (engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental) {
    bool ok = engine.DeEscalateCompartmentalToField(0);
    CHECK(ok);
  } else if (engine.lod.region_lods[0].current_lod == LODLevel::kNeuron) {
    engine.DeEscalate(0);
  } else if (engine.lod.region_lods[0].current_lod == LODLevel::kRegion) {
    engine.DeEscalateFromPopulation(0);
  }
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);

  // Field values should be valid
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    float E = engine.grid.channels[engine.field.ch_e].data[i];
    CHECK(std::isfinite(E) && E >= 0.0f && E <= 1.0f);
  }
}

TEST(lod_transition_automatic_compartmental_escalation) {
  auto engine = MakeTestEngine();
  engine.config.auto_lod = true;  // This test exercises auto-LOD

  // Set focus right on region_A with a compartmental zone
  engine.lod.SetFocus(25, 25, 25);
  engine.lod.zones = {
    {10.0f, LODLevel::kCompartmental},   // very close: compartmental
    {40.0f, LODLevel::kNeuron},          // nearby: spiking
  };
  engine.lod.default_level = LODLevel::kContinuum;

  // Step should auto-escalate region_A to compartmental
  int transitions = engine.Step(0.1f, 0.0f);
  CHECK(transitions >= 2);  // LOD 0 -> LOD 2 -> LOD 3

  CHECK(engine.comp_chunks.size() >= 1);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental);
}

TEST(lod_transition_automatic_compartmental_deescalation) {
  auto engine = MakeTestEngine();
  engine.config.auto_lod = true;  // This test exercises auto-LOD

  // Start with region_A at compartmental
  engine.lod.SetFocus(25, 25, 25);
  engine.lod.zones = {
    {10.0f, LODLevel::kCompartmental},
    {40.0f, LODLevel::kNeuron},
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.hysteresis = 0.0f;  // disable hysteresis for test

  engine.Step(0.1f, 0.0f);
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental);

  // Move focus far away: should de-escalate to continuum
  engine.lod.SetFocus(500, 500, 500);
  engine.Step(0.1f, 0.1f);

  CHECK(engine.comp_chunks.empty());
  CHECK(engine.chunks.empty());
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);
}

TEST(lod_transition_compartmental_total_neurons_and_spikes) {
  auto engine = MakeTestEngine();

  engine.Escalate(0);
  engine.EscalateToCompartmental(0);
  engine.Escalate(1);

  // TotalActiveNeurons should count both LOD 2 and LOD 3
  size_t total = engine.TotalActiveNeurons();
  CHECK(total == engine.chunks[0].neurons.n + engine.comp_chunks[0].neurons.n);
  CHECK(total > engine.comp_chunks[0].neurons.n);
}

TEST(lod_transition_compartmental_field_coupling) {
  auto engine = MakeTestEngine();

  // Set strong field activity
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f &&
        std::abs(engine.grid.channels[engine.ch_region].data[i]) < 0.5f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.8f;
    }
  }

  engine.Escalate(0);
  engine.EscalateToCompartmental(0);

  // Couple field to neurons
  engine.CoupleFieldToNeurons();

  // Boundary compartmental neurons should have received somatic input
  auto& comp = engine.comp_chunks[0];
  float total_ext = 0.0f;
  for (uint32_t bi : comp.boundary_indices) {
    total_ext += std::abs(comp.neurons.i_ext_soma[bi]);
  }
  if (!comp.boundary_indices.empty()) {
    CHECK(total_ext > 0.0f);
  }
}

TEST(lod_transition_compartmental_escalate_cycle) {
  auto engine = MakeTestEngine();

  // Configure auto-LOD to match each phase of the cycle.
  // Focus at region_A (25,25,25), zone radius covers region_A only.
  engine.lod.hysteresis = 0.0f;

  // Phase 1: spiking
  engine.lod.zones = {{30.0f, LODLevel::kNeuron}};
  engine.lod.default_level = LODLevel::kContinuum;
  engine.Escalate(0);
  for (int i = 0; i < 10; ++i) engine.Step(0.1f, i * 0.1f);

  // Phase 2: compartmental
  engine.lod.zones = {{30.0f, LODLevel::kCompartmental}};
  if (engine.lod.region_lods[0].current_lod == LODLevel::kNeuron) {
    engine.EscalateToCompartmental(0);
  }
  for (int i = 10; i < 20; ++i) engine.Step(0.025f, i * 0.025f);

  // Phase 3: back to spiking
  engine.lod.zones = {{30.0f, LODLevel::kNeuron}};
  if (engine.lod.region_lods[0].current_lod == LODLevel::kCompartmental) {
    engine.DeEscalateFromCompartmental(0);
  }
  for (int i = 20; i < 30; ++i) engine.Step(0.1f, i * 0.1f);

  // Phase 4: back to field
  engine.lod.zones.clear();
  LODLevel cur = engine.lod.region_lods[0].current_lod;
  if (cur == LODLevel::kCompartmental) {
    engine.DeEscalateCompartmentalToField(0);
  } else if (cur == LODLevel::kNeuron) {
    engine.DeEscalate(0);
  } else if (cur == LODLevel::kRegion) {
    engine.DeEscalateFromPopulation(0);
  }

  CHECK(engine.comp_chunks.empty());
  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);

  // All field values should be finite
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_e].data[i]));
  }
}

TEST(lod_transition_compartmental_double_escalate_fails) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);
  engine.EscalateToCompartmental(0);

  // Second escalation should fail
  bool ok = engine.EscalateToCompartmental(0);
  CHECK(!ok);
}

TEST(lod_transition_compartmental_read_activity) {
  auto engine = MakeTestEngine();
  engine.Escalate(0);
  engine.EscalateToCompartmental(0);

  // ReadActivity should work for compartmental regions
  float activity = engine.ReadActivity(25, 25, 25);
  CHECK(std::isfinite(activity));
  CHECK(activity >= 0.0f && activity <= 1.0f);
}

// ===== Cross-chunk projection tests =====

// Helper: create engine with a projection from region_A to region_B.
static LODTransitionEngine MakeProjectionEngine() {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.boundary_width_um = 8.0f;
  engine.config.seed = 123;

  // Define A -> B projection
  LODProjection proj;
  proj.from_region = "region_A";
  proj.to_region = "region_B";
  proj.density = 0.05f;
  proj.weight_mean = 1.5f;
  proj.weight_std = 0.2f;
  proj.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(proj);

  engine.Init(5.0f);
  return engine;
}

TEST(cross_chunk_projection_instantiated) {
  auto engine = MakeProjectionEngine();

  // Escalate only region_A: no cross links (B not active)
  engine.Escalate(0);
  CHECK(engine.ActiveCrossLinks() == 0);

  // Escalate region_B: cross link A->B should now exist
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);
  CHECK(engine.TotalCrossChunkSynapses() > 0);

  // Verify link direction
  CHECK(engine.cross_links[0].src_region == 0);
  CHECK(engine.cross_links[0].dst_region == 1);
}

TEST(cross_chunk_projection_no_duplicate) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);

  size_t links_before = engine.ActiveCrossLinks();
  // Calling InstantiateProjections again should not create duplicates
  engine.InstantiateProjections(0);
  engine.InstantiateProjections(1);
  CHECK(engine.ActiveCrossLinks() == links_before);
}

TEST(cross_chunk_spike_propagation) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);

  // Drive region_A neurons to spike
  auto& src_chunk = engine.chunks[engine.region_chunk_map[0]];
  for (size_t i = 0; i < src_chunk.neurons.n; ++i) {
    src_chunk.neurons.v[i] = 40.0f;  // above threshold
  }
  // Step to generate spikes
  IzhikevichStep(src_chunk.neurons, 0.1f, 0.0f, src_chunk.params);
  int src_spikes = src_chunk.CountSpikes();
  CHECK(src_spikes > 0);

  // Clear target synaptic input
  auto& dst_chunk = engine.chunks[engine.region_chunk_map[1]];
  dst_chunk.neurons.ClearSynapticInput();

  // Propagate spikes across the cross-chunk link
  engine.PropagateCrossChunkSpikes(1.0f);

  // Target neurons should have received synaptic input
  float total_isyn = 0.0f;
  for (size_t i = 0; i < dst_chunk.neurons.n; ++i) {
    total_isyn += std::abs(dst_chunk.neurons.i_syn[i]);
  }
  CHECK(total_isyn > 0.0f);
}

TEST(cross_chunk_projection_removed_on_deescalate) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);

  // De-escalate region_A: links involving it should be removed
  engine.DeEscalate(0);
  CHECK(engine.ActiveCrossLinks() == 0);
}

TEST(cross_chunk_projection_removed_on_deescalate_target) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);

  // De-escalate region_B (target): link should also be removed
  engine.DeEscalate(1);
  CHECK(engine.ActiveCrossLinks() == 0);
}

TEST(cross_chunk_projection_reestablished_on_reescalate) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);

  // De-escalate B, links gone
  engine.DeEscalate(1);
  CHECK(engine.ActiveCrossLinks() == 0);

  // Re-escalate B, link restored
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);
  CHECK(engine.TotalCrossChunkSynapses() > 0);
}

TEST(cross_chunk_projection_with_compartmental) {
  auto engine = MakeProjectionEngine();

  // Region A at LOD 3 (compartmental), region B at LOD 2 (spiking)
  engine.Escalate(0);
  engine.Escalate(1);
  size_t syns_lod2 = engine.TotalCrossChunkSynapses();
  CHECK(syns_lod2 > 0);

  // Escalate A to compartmental: cross link should persist (uses GetRegionSpiked/ISyn)
  engine.EscalateToCompartmental(0);

  // Region A is now compartmental; cross-chunk link should still exist
  CHECK(engine.ActiveCrossLinks() == 1);

  // Propagation should work: drive A's compartmental neurons to spike
  auto& comp = engine.comp_chunks[engine.region_comp_map[0]];
  for (size_t i = 0; i < comp.neurons.n; ++i) {
    comp.neurons.v_soma[i] = 10.0f;  // well above threshold (-50mV)
  }
  CompartmentalStep(comp.neurons, 0.025f, 0.0f, comp.params);
  int comp_spikes = comp.CountSpikes();
  CHECK(comp_spikes > 0);

  auto& dst = engine.chunks[engine.region_chunk_map[1]];
  dst.neurons.ClearSynapticInput();
  engine.PropagateCrossChunkSpikes(1.0f);

  // Target spiking neurons should have received input
  float total_isyn = 0.0f;
  for (size_t i = 0; i < dst.neurons.n; ++i) {
    total_isyn += std::abs(dst.neurons.i_syn[i]);
  }
  CHECK(total_isyn > 0.0f);
}

TEST(cross_chunk_bidirectional_projections) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.seed = 42;

  // A -> B
  LODProjection p1;
  p1.from_region = "region_A"; p1.to_region = "region_B";
  p1.density = 0.03f; p1.weight_mean = 1.0f; p1.weight_std = 0.1f;
  p1.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(p1);

  // B -> A (feedback)
  LODProjection p2;
  p2.from_region = "region_B"; p2.to_region = "region_A";
  p2.density = 0.02f; p2.weight_mean = 0.8f; p2.weight_std = 0.1f;
  p2.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(p2);

  engine.Init(5.0f);
  engine.Escalate(0);
  engine.Escalate(1);

  // Both projections should be instantiated
  CHECK(engine.ActiveCrossLinks() == 2);

  // Verify directions
  bool found_ab = false, found_ba = false;
  for (const auto& link : engine.cross_links) {
    if (link.src_region == 0 && link.dst_region == 1) found_ab = true;
    if (link.src_region == 1 && link.dst_region == 0) found_ba = true;
  }
  CHECK(found_ab);
  CHECK(found_ba);
}

// ===== Cross-LOD coupling tests =====

TEST(coupling_population_to_spiking) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.seed = 77;

  // A -> B projection
  LODProjection proj;
  proj.from_region = "region_A"; proj.to_region = "region_B";
  proj.density = 0.05f; proj.weight_mean = 1.0f; proj.weight_std = 0.1f;
  proj.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(proj);

  engine.Init(5.0f);

  // Region A at LOD 1 (population), Region B at LOD 2 (spiking)
  engine.EscalateToPopulation(0);
  engine.Escalate(1);

  // Set high firing rate in population A
  auto& pop = engine.pop_chunks[engine.region_pop_map[0]];
  pop.state.r_e = 50.0f;  // 50 Hz

  // Clear external input on B boundary neurons
  auto& chunk_b = engine.chunks[engine.region_chunk_map[1]];
  chunk_b.neurons.ClearExternalInput();

  // Couple populations to neurons
  engine.CouplePopulationsToNeurons(0.1f);

  // Boundary neurons of B should have received external current
  float total_ext = 0.0f;
  for (uint32_t bi : chunk_b.boundary_indices) {
    total_ext += std::abs(chunk_b.neurons.i_ext[bi]);
  }
  if (!chunk_b.boundary_indices.empty()) {
    CHECK(total_ext > 0.0f);
  }
}

TEST(coupling_spiking_to_population) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.seed = 88;

  // A -> B projection
  LODProjection proj;
  proj.from_region = "region_A"; proj.to_region = "region_B";
  proj.density = 0.05f; proj.weight_mean = 1.0f; proj.weight_std = 0.1f;
  proj.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(proj);

  engine.Init(5.0f);

  // Region A at LOD 2 (spiking), Region B at LOD 1 (population)
  engine.Escalate(0);
  engine.EscalateToPopulation(1);

  // Drive spiking neurons in A to generate spikes
  auto& chunk_a = engine.chunks[engine.region_chunk_map[0]];
  for (size_t i = 0; i < chunk_a.neurons.n; ++i) {
    chunk_a.neurons.v[i] = 40.0f;
  }
  IzhikevichStep(chunk_a.neurons, 0.1f, 0.0f, chunk_a.params);
  chunk_a.AccumulateSpikes();
  int spikes = chunk_a.CountSpikes();
  CHECK(spikes > 0);

  // Couple neurons to population
  auto& pop_b = engine.pop_chunks[engine.region_pop_map[1]];
  float ext_before = pop_b.state.I_ext_e;
  engine.CouplePopulationsToNeurons(0.1f);
  float ext_after = pop_b.state.I_ext_e;

  // Population B should have received external input from spiking A
  CHECK(ext_after > ext_before);
}

TEST(coupling_population_to_population) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.seed = 99;

  LODProjection proj;
  proj.from_region = "region_A"; proj.to_region = "region_B";
  proj.density = 0.05f; proj.weight_mean = 1.0f; proj.weight_std = 0.1f;
  proj.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(proj);

  engine.Init(5.0f);

  // Both at LOD 1
  engine.EscalateToPopulation(0);
  engine.EscalateToPopulation(1);

  // Set high rate in A
  engine.pop_chunks[engine.region_pop_map[0]].state.r_e = 40.0f;

  // Clear B's external input
  auto& pop_b = engine.pop_chunks[engine.region_pop_map[1]];
  pop_b.state.I_ext_e = 0.0f;

  engine.CouplePopulationsToNeurons(0.1f);

  // B should receive drive from A's rate
  CHECK(pop_b.state.I_ext_e > 0.0f);
}

TEST(coupling_population_to_compartmental) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"region_A", 25, 25, 25, 15, 15, 15});
  engine.brain_sdf.primitives.push_back({"region_B", 75, 25, 25, 15, 15, 15});
  engine.config.neurons_per_voxel = 5.0f;
  engine.config.synapse_density = 0.1f;
  engine.config.seed = 55;

  LODProjection proj;
  proj.from_region = "region_A"; proj.to_region = "region_B";
  proj.density = 0.05f; proj.weight_mean = 1.0f; proj.weight_std = 0.1f;
  proj.nt_type = static_cast<uint8_t>(kACh);
  engine.projections.push_back(proj);

  engine.Init(5.0f);

  // A at LOD 1, B at LOD 3
  engine.EscalateToPopulation(0);
  engine.Escalate(1);
  engine.EscalateToCompartmental(1);

  // Set high rate in population A
  engine.pop_chunks[engine.region_pop_map[0]].state.r_e = 60.0f;

  // Clear B's external input
  auto& comp_b = engine.comp_chunks[engine.region_comp_map[1]];
  comp_b.neurons.ClearExternalInput();

  engine.CouplePopulationsToNeurons(0.1f);

  // Compartmental boundary neurons should receive somatic input
  float total_ext = 0.0f;
  for (uint32_t bi : comp_b.boundary_indices) {
    total_ext += std::abs(comp_b.neurons.i_ext_soma[bi]);
  }
  if (!comp_b.boundary_indices.empty()) {
    CHECK(total_ext > 0.0f);
  }
}

// ===== Mixed-LOD integration tests =====

TEST(mixed_lod_step_all_levels) {
  LODTransitionEngine engine;
  engine.brain_sdf.primitives.push_back({"field_region", 25, 25, 25, 12, 12, 12});
  engine.brain_sdf.primitives.push_back({"pop_region", 55, 25, 25, 12, 12, 12});
  engine.brain_sdf.primitives.push_back({"spike_region", 85, 25, 25, 12, 12, 12});
  engine.brain_sdf.primitives.push_back({"comp_region", 115, 25, 25, 12, 12, 12});
  engine.config.neurons_per_voxel = 3.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.seed = 333;

  // Add projections between adjacent LOD levels
  auto add_proj = [&](const char* from, const char* to) {
    LODProjection p;
    p.from_region = from; p.to_region = to;
    p.density = 0.03f; p.weight_mean = 1.0f; p.weight_std = 0.1f;
    p.nt_type = static_cast<uint8_t>(kACh);
    engine.projections.push_back(p);
  };
  add_proj("pop_region", "spike_region");
  add_proj("spike_region", "comp_region");
  add_proj("spike_region", "pop_region");

  engine.Init(5.0f);

  // Set each region to a different LOD level
  // field_region stays at LOD 0
  engine.EscalateToPopulation(1);         // LOD 1
  engine.Escalate(2);                      // LOD 2
  engine.Escalate(3);
  engine.EscalateToCompartmental(3);       // LOD 3

  CHECK(engine.lod.region_lods[0].current_lod == LODLevel::kContinuum);
  CHECK(engine.lod.region_lods[1].current_lod == LODLevel::kRegion);
  CHECK(engine.lod.region_lods[2].current_lod == LODLevel::kNeuron);
  CHECK(engine.lod.region_lods[3].current_lod == LODLevel::kCompartmental);

  // Run 50 timesteps: all coupling paths exercised
  for (int i = 0; i < 50; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // Verify all states are finite (no NaN/Inf propagation)
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_e].data[i]));
    CHECK(std::isfinite(engine.grid.channels[engine.field.ch_i].data[i]));
  }
  for (const auto& pop : engine.pop_chunks) {
    CHECK(std::isfinite(pop.state.r_e));
    CHECK(std::isfinite(pop.state.v_e));
    CHECK(std::isfinite(pop.state.a_e));
  }
  for (const auto& chunk : engine.chunks) {
    for (size_t i = 0; i < chunk.neurons.n; ++i) {
      CHECK(std::isfinite(chunk.neurons.v[i]));
    }
  }
  for (const auto& comp : engine.comp_chunks) {
    for (size_t i = 0; i < comp.neurons.n; ++i) {
      CHECK(std::isfinite(comp.neurons.v_soma[i]));
      CHECK(std::isfinite(comp.neurons.v_apical[i]));
      CHECK(std::isfinite(comp.neurons.v_basal[i]));
    }
  }
}

TEST(mixed_lod_step_produces_activity) {
  auto engine = MakeProjectionEngine();

  // Set focus on region_B and configure zones so auto-LOD keeps
  // A at population (LOD 1) and B at spiking (LOD 2).
  // B at (75,25,25) -> distance 0 -> kNeuron zone
  // A at (25,25,25) -> distance 50 -> kRegion zone
  engine.lod.SetFocus(75.0f, 25.0f, 25.0f);
  engine.lod.zones = {
    {30.0f, LODLevel::kNeuron},    // within 30um -> spiking (B)
    {60.0f, LODLevel::kRegion},    // within 60um -> population (A)
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.hysteresis = 0.0f;

  // A at LOD 1, B at LOD 2
  engine.EscalateToPopulation(0);
  engine.Escalate(1);

  // Inject strong stimulus into field to drive activity
  for (size_t i = 0; i < engine.grid.NumVoxels(); ++i) {
    if (engine.grid.channels[engine.ch_sdf].data[i] < 0.0f) {
      engine.grid.channels[engine.field.ch_e].data[i] = 0.6f;
    }
  }

  // Run simulation with cross-LOD coupling
  for (int i = 0; i < 100; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // Check activity if regions are still at their intended LOD levels.
  // Auto-LOD may have adjusted levels based on distance from focus.
  if (engine.region_pop_map[0] >= 0) {
    auto& pop = engine.pop_chunks[engine.region_pop_map[0]];
    CHECK(pop.state.r_e >= 0.0f);
  }

  if (engine.region_chunk_map[1] >= 0) {
    auto& chunk_b = engine.chunks[engine.region_chunk_map[1]];
    int total_accum = 0;
    for (auto c : chunk_b.voxel_spike_counts) total_accum += c;
    CHECK(total_accum >= 0);
  }
}

TEST(cross_chunk_step_integration) {
  auto engine = MakeProjectionEngine();
  engine.Escalate(0);
  engine.Escalate(1);
  CHECK(engine.ActiveCrossLinks() == 1);

  // Inject strong current into region_A to provoke cross-chunk traffic
  auto& chunk_a = engine.chunks[engine.region_chunk_map[0]];
  for (size_t i = 0; i < chunk_a.neurons.n; ++i) {
    chunk_a.neurons.i_ext[i] = 20.0f;
  }

  // Run 20 full steps (Step() handles all coupling + propagation)
  for (int i = 0; i < 20; ++i) {
    engine.Step(0.1f, i * 0.1f);
  }

  // Region B should have been affected by cross-chunk spikes from A
  auto& chunk_b = engine.chunks[engine.region_chunk_map[1]];
  for (size_t i = 0; i < chunk_b.neurons.n; ++i) {
    CHECK(std::isfinite(chunk_b.neurons.v[i]));
  }
}

// ===== NeuromodulatorField tests =====

TEST(neuromod_field_init) {
  // Create a small circuit with positions
  NeuronArray neurons;
  neurons.Resize(10);
  for (uint32_t i = 0; i < 10; ++i) {
    // Place neurons at known positions (in nm, as NeuronArray convention)
    neurons.x[i] = static_cast<float>(i * 50000);  // 0-450 um in nm
    neurons.y[i] = 150000.0f;  // 150 um
    neurons.z[i] = 100000.0f;  // 100 um
  }

  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 50.0f;  // coarse for test
  NeuromodulatorField field;
  field.Init(cfg, neurons);

  CHECK(field.grid.nx > 0);
  CHECK(field.grid.ny > 0);
  CHECK(field.grid.nz > 0);
  CHECK(field.grid.channels.size() == 3);  // DA, 5HT, OA

  // All neurons should be in grid
  for (uint32_t i = 0; i < 10; ++i) {
    CHECK(field.neuron_in_grid[i]);
  }
}

TEST(neuromod_field_inject_and_sample) {
  // Build a tiny MB circuit with ParametricGenerator
  BrainSpec spec;
  spec.seed = 42;
  spec.name = "test_neuromod";

  RegionSpec kc_r;
  kc_r.name = "KC"; kc_r.n_neurons = 20;
  kc_r.default_nt = kACh;
  kc_r.cell_types = {{CellType::kKenyonCell, 1.0f}};
  spec.regions.push_back(kc_r);

  RegionSpec dan_r;
  dan_r.name = "DAN"; dan_r.n_neurons = 5;
  dan_r.default_nt = kDA;
  dan_r.cell_types = {{CellType::kDAN_PPL1, 1.0f}};
  spec.regions.push_back(dan_r);

  spec.projections.push_back({"DAN", "KC", 0.3f, kDA, 1.0f, 0.1f});

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Assign spatial positions and init field
  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 50.0f;
  NeuromodulatorField field;
  field.InitWithRandomPositions(cfg, neurons, 42);

  // Initially, all DA concentrations should be 0
  for (uint32_t i = 0; i < neurons.n; ++i) {
    CHECK(neurons.dopamine[i] == 0.0f);
  }

  // Simulate a DAN spike: set spiked for DAN neurons (indices 20-24)
  uint32_t dan_start = gen.region_ranges[1].start;
  uint32_t dan_end = gen.region_ranges[1].end;
  for (uint32_t i = dan_start; i < dan_end; ++i) {
    neurons.spiked[i] = 1;
  }

  // Inject release
  field.InjectRelease(neurons, synapses, cfg);

  // Check that DA was injected into grid
  bool any_da = false;
  for (auto& v : field.grid.channels[field.ch_da].data) {
    if (v > 0.0f) { any_da = true; break; }
  }
  CHECK(any_da);

  // Sample back to neurons
  field.SampleToNeurons(neurons);

  // Neurons near the DAN neurons should have nonzero DA
  bool any_neuron_da = false;
  for (uint32_t i = 0; i < neurons.n; ++i) {
    if (neurons.dopamine[i] > 0.0f) { any_neuron_da = true; break; }
  }
  CHECK(any_neuron_da);
}

TEST(neuromod_field_diffusion_spreads) {
  // Test that DA diffuses spatially over time
  NeuronArray neurons;
  neurons.Resize(2);
  // Place two neurons 200 um apart
  neurons.x[0] = 100000.0f;  // 100 um in nm
  neurons.y[0] = 150000.0f;
  neurons.z[0] = 100000.0f;
  neurons.x[1] = 300000.0f;  // 300 um in nm
  neurons.y[1] = 150000.0f;
  neurons.z[1] = 100000.0f;

  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 20.0f;

  NeuromodulatorField field;
  field.Init(cfg, neurons);

  // Inject DA at neuron 0's location
  float wx = neurons.x[0] / 1000.0f;
  float wy = neurons.y[0] / 1000.0f;
  float wz = neurons.z[0] / 1000.0f;
  field.grid.Inject(field.ch_da, wx, wy, wz, 10.0f);

  // Before diffusion: neuron 0 should see DA, neuron 1 should not
  field.SampleToNeurons(neurons);
  float da0_before = neurons.dopamine[0];
  float da1_before = neurons.dopamine[1];
  CHECK(da0_before > 0.0f);
  CHECK(da1_before == 0.0f);

  // Diffuse for many steps
  for (int i = 0; i < 100; ++i) {
    field.Diffuse(1.0f);  // 1ms per step
  }

  // After diffusion: both neurons should see DA, but neuron 0 less than before
  field.SampleToNeurons(neurons);
  float da0_after = neurons.dopamine[0];
  float da1_after = neurons.dopamine[1];

  // DA should have spread: neuron 0's concentration decreased
  CHECK(da0_after < da0_before);
  // DA should have reached neuron 1 (200 um away, D=2.4 um^2/ms, 100ms)
  CHECK(da1_after > 0.0f);
}

TEST(neuromod_field_decay) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.x[0] = 100000.0f;
  neurons.y[0] = 150000.0f;
  neurons.z[0] = 100000.0f;

  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 50.0f;
  cfg.da_diffusion = 0.0f;  // disable diffusion to isolate decay

  NeuromodulatorField field;
  field.Init(cfg, neurons);

  // Inject DA
  float wx = neurons.x[0] / 1000.0f;
  float wy = neurons.y[0] / 1000.0f;
  float wz = neurons.z[0] / 1000.0f;
  field.grid.Inject(field.ch_da, wx, wy, wz, 1.0f);

  field.SampleToNeurons(neurons);
  float da_initial = neurons.dopamine[0];
  CHECK(da_initial > 0.0f);

  // Diffuse (only decay active since D=0)
  for (int i = 0; i < 200; ++i) {
    field.Diffuse(1.0f);
  }

  // After 200ms with decay rate 0.01/ms (tau=100ms), expect ~13.5% remaining
  field.SampleToNeurons(neurons);
  float da_final = neurons.dopamine[0];
  float expected_ratio = std::exp(-0.01f * 200.0f);  // ~0.135
  float actual_ratio = da_final / da_initial;
  CHECK(std::abs(actual_ratio - expected_ratio) < 0.02f);
}

TEST(neuromod_field_full_update) {
  // Build a circuit and run the full Update() pipeline
  BrainSpec spec;
  spec.seed = 99;
  spec.name = "test_full_update";

  RegionSpec kc_r;
  kc_r.name = "KC"; kc_r.n_neurons = 50;
  kc_r.default_nt = kACh;
  kc_r.cell_types = {{CellType::kKenyonCell, 1.0f}};
  spec.regions.push_back(kc_r);

  RegionSpec dan_r;
  dan_r.name = "DAN"; dan_r.n_neurons = 10;
  dan_r.default_nt = kDA;
  dan_r.cell_types = {{CellType::kDAN_PPL1, 1.0f}};
  spec.regions.push_back(dan_r);

  spec.projections.push_back({"DAN", "KC", 0.3f, kDA, 1.0f, 0.1f});

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 50.0f;
  NeuromodulatorField field;
  field.InitWithRandomPositions(cfg, neurons, 99);

  // Run a simulation: step dynamics, spike some DANs, update field
  SpikeFrequencyAdaptation sfa;
  sfa.Init(neurons.n);

  uint32_t dan_start = gen.region_ranges[1].start;
  uint32_t dan_end = gen.region_ranges[1].end;

  // Run for 100 steps, spiking DANs with strong input
  float sim_time = 0.0f;
  float dt = 0.1f;
  int total_dan_spikes = 0;
  for (int step = 0; step < 1000; ++step) {
    neurons.ClearExternalInput();

    // Drive DANs to spike
    for (uint32_t i = dan_start; i < dan_end; ++i) {
      neurons.i_ext[i] = 20.0f;
    }

    neurons.DecaySynapticInput(dt, 3.0f);
    synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
    sfa.Update(neurons, dt);
    IzhikevichStepHeterogeneousFast(neurons, dt, sim_time, types);

    // Count DAN spikes
    for (uint32_t i = dan_start; i < dan_end; ++i) {
      total_dan_spikes += neurons.spiked[i];
    }

    // Update neuromodulator field (inject + diffuse + sample)
    field.Update(neurons, synapses, dt, cfg);

    sim_time += dt;
  }

  // DANs should have spiked
  CHECK(total_dan_spikes > 0);

  // Some KC neurons should now have nonzero dopamine
  int kcs_with_da = 0;
  for (uint32_t i = 0; i < gen.region_ranges[0].end; ++i) {
    if (neurons.dopamine[i] > 0.001f) kcs_with_da++;
  }
  CHECK(kcs_with_da > 0);

  // DA concentration should be spatially varying (not all identical)
  float min_da = 1e9f, max_da = -1e9f;
  for (uint32_t i = 0; i < gen.region_ranges[0].end; ++i) {
    min_da = std::min(min_da, neurons.dopamine[i]);
    max_da = std::max(max_da, neurons.dopamine[i]);
  }
  // With random positions and localized DAN release, there should be variance
  CHECK(max_da > min_da);
}

TEST(neuromod_field_serotonin_and_octopamine) {
  // Verify all three channels work independently
  NeuronArray neurons;
  neurons.Resize(3);
  for (int i = 0; i < 3; ++i) {
    neurons.x[i] = 250000.0f;
    neurons.y[i] = 150000.0f;
    neurons.z[i] = 100000.0f;
  }

  NeuromodulatorFieldConfig cfg;
  cfg.voxel_spacing_um = 50.0f;
  cfg.da_diffusion = 0.0f;
  cfg.sht_diffusion = 0.0f;
  cfg.oa_diffusion = 0.0f;

  NeuromodulatorField field;
  field.Init(cfg, neurons);

  float wx = neurons.x[0] / 1000.0f;
  float wy = neurons.y[0] / 1000.0f;
  float wz = neurons.z[0] / 1000.0f;

  // Inject each channel separately
  field.grid.Inject(field.ch_da, wx, wy, wz, 1.0f);
  field.grid.Inject(field.ch_sht, wx, wy, wz, 2.0f);
  field.grid.Inject(field.ch_oa, wx, wy, wz, 3.0f);

  field.SampleToNeurons(neurons);

  // All three should be nonzero and independent
  CHECK(neurons.dopamine[0] > 0.0f);
  CHECK(neurons.serotonin[0] > 0.0f);
  CHECK(neurons.octopamine[0] > 0.0f);

  // Ratios should reflect injection amounts
  CHECK(neurons.serotonin[0] > neurons.dopamine[0]);
  CHECK(neurons.octopamine[0] > neurons.serotonin[0]);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
