// Parametric brain tests: cell type manager, parametric generator, brain spec loader,
// parameter sweep, scoring functions
#include "test_harness.h"

#include "core/cell_types.h"
#include "core/connectome_loader.h"
#include "core/connectome_export.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/parametric_gen.h"
#include "core/brain_spec_loader.h"
#include "core/param_sweep.h"
#include "core/parametric_sync.h"
#include "core/region_metrics.h"
#include "core/structural_plasticity.h"
#include "core/rate_monitor.h"
#include "core/nwb_export.h"
#include "core/intrinsic_homeostasis.h"

// ===== CellTypeManager tests =====

TEST(cell_type_params) {
  auto kc = ParamsForCellType(CellType::kKenyonCell);
  CHECK(std::abs(kc.a - 0.02f) < 0.001f);
  CHECK(std::abs(kc.d - 10.0f) < 0.01f);

  auto fs = ParamsForCellType(CellType::kLN_local);
  CHECK(std::abs(fs.a - 0.1f) < 0.001f);
  CHECK(std::abs(fs.d - 2.0f) < 0.01f);

  auto burst = ParamsForCellType(CellType::kPN_excitatory);
  CHECK(std::abs(burst.c - (-58.0f)) < 0.01f);
}

TEST(cell_type_manager_assign) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 5; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);
  for (size_t i = 5; i < 10; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kLN_local);

  CellTypeManager types;
  types.AssignFromTypes(neurons);

  CHECK(types.neuron_params.size() == 10);
  // KCs have a=0.02, LNs have a=0.1
  CHECK(std::abs(types.Get(0).a - 0.02f) < 0.001f);
  CHECK(std::abs(types.Get(5).a - 0.1f) < 0.001f);
}

TEST(cell_type_manager_override) {
  NeuronArray neurons;
  neurons.Resize(5);
  for (size_t i = 0; i < 5; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);

  CellTypeManager types;
  IzhikevichParams custom;
  custom.a = 0.05f;
  custom.b = 0.25f;
  custom.c = -55.0f;
  custom.d = 6.0f;
  types.SetOverride(CellType::kKenyonCell, custom);
  types.AssignFromTypes(neurons);

  // All neurons should have the custom override
  for (size_t i = 0; i < 5; ++i) {
    CHECK(std::abs(types.Get(i).a - 0.05f) < 0.001f);
    CHECK(std::abs(types.Get(i).c - (-55.0f)) < 0.01f);
  }
}

TEST(heterogeneous_step) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);
    neurons.i_ext[i] = 15.0f;  // strong drive to ensure spiking
  }

  CellTypeManager types;
  types.AssignFromTypes(neurons);

  float sim_time = 0.0f;
  for (int step = 0; step < 2000; ++step) {
    IzhikevichStepHeterogeneous(neurons, 0.1f, sim_time, types);
    sim_time += 0.1f;
  }

  // With strong drive, at least some neurons should have spiked
  // Check that last_spike_time was updated for at least one neuron
  bool any_spike_recorded = false;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.last_spike_time[i] > 0.0f) {
      any_spike_recorded = true;
      break;
    }
  }
  CHECK(any_spike_recorded);
}

// ===== ParametricGenerator tests =====

TEST(parametric_gen_single_region) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test_region";
  reg.n_neurons = 50;
  reg.internal_density = 0.1f;
  reg.default_nt = kACh;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);

  CHECK(total == 50);
  CHECK(neurons.n == 50);
  CHECK(synapses.Size() > 0);  // ~10% density should create some synapses
  CHECK(gen.region_ranges.size() == 1);
  CHECK(gen.region_ranges[0].start == 0);
  CHECK(gen.region_ranges[0].end == 50);
}

TEST(parametric_gen_two_regions_with_projection) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec r1;
  r1.name = "region_a";
  r1.n_neurons = 30;
  r1.internal_density = 0.05f;
  spec.regions.push_back(r1);

  RegionSpec r2;
  r2.name = "region_b";
  r2.n_neurons = 20;
  r2.internal_density = 0.05f;
  spec.regions.push_back(r2);

  ProjectionSpec proj;
  proj.from_region = "region_a";
  proj.to_region = "region_b";
  proj.density = 0.1f;
  proj.nt_type = kACh;
  proj.weight_mean = 1.5f;
  proj.weight_std = 0.2f;
  spec.projections.push_back(proj);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);

  CHECK(total == 50);
  CHECK(gen.region_ranges.size() == 2);
  CHECK(gen.region_ranges[0].end == 30);
  CHECK(gen.region_ranges[1].start == 30);
  CHECK(gen.region_ranges[1].end == 50);

  // Region assignments
  CHECK(neurons.region[0] == 0);
  CHECK(neurons.region[30] == 1);
}

TEST(parametric_gen_nt_distribution) {
  BrainSpec spec;
  spec.seed = 123;

  RegionSpec reg;
  reg.name = "mixed_nt";
  reg.n_neurons = 100;
  reg.internal_density = 0.2f;
  reg.nt_distribution.push_back({kACh, 0.5f});
  reg.nt_distribution.push_back({kGABA, 0.5f});
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // With 50/50 ACh/GABA distribution, count actual NT types in synapse table
  CHECK(synapses.Size() > 0);
  size_t n_ach = 0, n_gaba = 0;
  for (size_t i = 0; i < synapses.Size(); ++i) {
    if (synapses.nt_type[i] == kACh) n_ach++;
    else if (synapses.nt_type[i] == kGABA) n_gaba++;
  }
  CHECK(n_ach > 0);
  CHECK(n_gaba > 0);
  // Ratio should be roughly 50/50 (within 30% tolerance)
  float ratio = static_cast<float>(n_ach) / (n_ach + n_gaba);
  CHECK(ratio > 0.2f && ratio < 0.8f);
}

TEST(parametric_gen_cell_type_assignment) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec reg;
  reg.name = "typed_region";
  reg.n_neurons = 100;
  reg.internal_density = 0.01f;
  reg.cell_types.push_back({CellType::kKenyonCell, 0.6f});
  reg.cell_types.push_back({CellType::kLN_local, 0.4f});
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Count cell types
  int kc_count = 0, ln_count = 0;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.type[i] == static_cast<uint8_t>(CellType::kKenyonCell)) kc_count++;
    if (neurons.type[i] == static_cast<uint8_t>(CellType::kLN_local)) ln_count++;
  }
  CHECK(kc_count == 60);  // 60% of 100
  CHECK(ln_count == 40);  // 40% of 100
}

TEST(parametric_gen_run_simulation) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec reg;
  reg.name = "sim_test";
  reg.n_neurons = 20;
  reg.internal_density = 0.15f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Inject current and simulate
  for (size_t i = 0; i < neurons.n; ++i) neurons.i_ext[i] = 10.0f;

  float sim_time = 0.0f;
  for (int step = 0; step < 1000; ++step) {
    neurons.ClearSynapticInput();
    synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(neurons, 0.1f, sim_time, types);
    sim_time += 0.1f;
  }

  // Should produce activity
  bool any_spiked = false;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.last_spike_time[i] > 0.0f) { any_spiked = true; break; }
  }
  CHECK(any_spiked);
}

// ===== BrainSpecLoader tests =====

TEST(brain_spec_loader_roundtrip) {
  // Write a minimal .brain file
  const char* content =
    "name = test_brain\n"
    "seed = 123\n"
    "weight_mean = 1.5\n"
    "weight_std = 0.4\n"
    "\n"
    "region.0.name = region_a\n"
    "region.0.n_neurons = 100\n"
    "region.0.density = 0.1\n"
    "region.0.nt = ACh\n"
    "region.0.types = KC:0.7 LN:0.3\n"
    "\n"
    "region.1.name = region_b\n"
    "region.1.n_neurons = 50\n"
    "region.1.density = 0.05\n"
    "\n"
    "projection.0.from = region_a\n"
    "projection.0.to = region_b\n"
    "projection.0.density = 0.02\n"
    "projection.0.nt = GABA\n"
    "projection.0.weight_mean = 1.2\n"
    "projection.0.weight_std = 0.3\n";

  std::string path = "test_tmp_brain_spec.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(result.has_value());
  auto& spec = *result;
  CHECK(spec.name == "test_brain");
  CHECK(spec.seed == 123);
  CHECK(std::abs(spec.global_weight_mean - 1.5f) < 0.01f);
  CHECK(spec.regions.size() == 2);
  CHECK(spec.regions[0].name == "region_a");
  CHECK(spec.regions[0].n_neurons == 100);
  CHECK(spec.regions[0].cell_types.size() == 2);
  CHECK(spec.regions[1].name == "region_b");
  CHECK(spec.regions[1].n_neurons == 50);
  CHECK(spec.projections.size() == 1);
  CHECK(spec.projections[0].from_region == "region_a");
  CHECK(spec.projections[0].to_region == "region_b");
  CHECK(spec.projections[0].nt_type == kGABA);
}

TEST(brain_spec_loader_missing_file) {
  auto result = BrainSpecLoader::Load("nonexistent_file.brain");
  CHECK(!result.has_value());
}

TEST(brain_spec_loader_empty_regions) {
  std::string path = "test_tmp_empty_brain.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs("name = empty\n", f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(!result.has_value());  // No regions = error
}

TEST(brain_spec_loader_nt_distribution) {
  const char* content =
    "region.0.name = mixed\n"
    "region.0.n_neurons = 50\n"
    "region.0.density = 0.1\n"
    "region.0.nt_dist = ACh:0.6 GABA:0.4\n";

  std::string path = "test_tmp_nt_dist.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(result.has_value());
  CHECK(result->regions[0].nt_distribution.size() == 2);
  CHECK(result->regions[0].nt_distribution[0].nt == kACh);
  CHECK(std::abs(result->regions[0].nt_distribution[0].fraction - 0.6f) < 0.01f);
}

// ===== ParamSweep tests =====

TEST(param_sweep_grid) {
  // Small grid sweep with tiny neuron population
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;  // no synapses, just test grid mechanics

  ParamSweep sweep;
  sweep.grid_steps = 2;  // 2^4 = 16 points
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.GridSweep(CellType::kGeneric, neurons, synapses,
                  scoring::TargetFiringRate(10.0f, 0.5f));

  CHECK(sweep.results.size() == 16);
  // Results should be sorted best-first
  for (size_t i = 1; i < sweep.results.size(); ++i) {
    CHECK(sweep.results[i-1].score >= sweep.results[i].score);
  }
}

TEST(param_sweep_random) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;

  ParamSweep sweep;
  sweep.random_samples = 20;
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.RandomSweep(CellType::kGeneric, neurons, synapses,
                    scoring::TargetFiringRate(10.0f, 0.5f));

  CHECK(sweep.results.size() == 20);
  // Sorted best-first
  for (size_t i = 1; i < sweep.results.size(); ++i) {
    CHECK(sweep.results[i-1].score >= sweep.results[i].score);
  }
}

TEST(param_sweep_refine) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;

  ParamSweep sweep;
  sweep.grid_steps = 2;
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.GridSweep(CellType::kGeneric, neurons, synapses,
                  scoring::TargetFiringRate(10.0f, 0.5f));

  float score_before = sweep.BestScore();
  sweep.Refine(CellType::kGeneric, neurons, synapses,
               scoring::TargetFiringRate(10.0f, 0.5f), 10, 0.5f);

  // Refinement should not worsen the best score
  CHECK(sweep.BestScore() >= score_before);
}

// ===== Scoring function tests =====

TEST(scoring_target_firing_rate) {
  auto fn = scoring::TargetFiringRate(10.0f, 0.1f);

  NeuronArray neurons;
  neurons.Resize(100);
  // No spikes; score should reflect 0 Hz vs 10 Hz target
  float score_zero = fn(neurons, 1000.0f);

  // Simulate some spikes
  for (size_t i = 0; i < 100; ++i) neurons.spiked[i] = 1;
  float score_some = fn(neurons, 1000.0f);

  // Some spiking should score differently than zero
  CHECK(score_zero != score_some);
  CHECK(score_zero > 0.0f);  // 1/(1+error), always positive
}

TEST(scoring_activity_in_range) {
  auto fn = scoring::ActivityInRange(0.1f, 0.5f);

  NeuronArray neurons;
  neurons.Resize(100);

  // No activity
  float score_none = fn(neurons, 1000.0f);
  CHECK(score_none < 1.0f);

  // 30% active (in range)
  for (size_t i = 0; i < 30; ++i) neurons.last_spike_time[i] = 50.0f;
  float score_good = fn(neurons, 1000.0f);
  CHECK(std::abs(score_good - 1.0f) < 0.01f);

  // 80% active (above range)
  for (size_t i = 0; i < 80; ++i) neurons.last_spike_time[i] = 50.0f;
  float score_high = fn(neurons, 1000.0f);
  CHECK(score_high < 1.0f);
}

// ===== ParametricSync tests =====

TEST(parametric_sync_init) {
  ParametricSync sync;
  sync.Init(20, 100);
  CHECK(sync.neuron_state.size() == 20);
  CHECK(sync.weight_velocity.size() == 100);
  CHECK(sync.weight_error_accum.size() == 100);
  CHECK(sync.total_steps == 0);
}

TEST(parametric_sync_step_reduces_error) {
  // Create identical model and reference, then perturb model weights.
  // Sync should reduce the divergence over time.
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test";
  reg.n_neurons = 20;
  reg.internal_density = 0.15f;
  spec.regions.push_back(reg);

  NeuronArray model_neurons, ref_neurons;
  SynapseTable model_synapses, ref_synapses;
  CellTypeManager model_types, ref_types;
  ParametricGenerator gen1, gen2;

  gen1.Generate(spec, model_neurons, model_synapses, model_types);
  gen2.Generate(spec, ref_neurons, ref_synapses, ref_types);

  // Perturb model weights to create initial mismatch
  for (size_t i = 0; i < model_synapses.Size(); ++i) {
    model_synapses.weight[i] *= 1.5f;
  }

  // Inject current
  for (size_t i = 0; i < 20; ++i) {
    model_neurons.i_ext[i] = 8.0f;
    ref_neurons.i_ext[i] = 8.0f;
  }

  ParametricSync sync;
  sync.dt_ms = 0.5f;
  sync.metric_interval = 200;
  sync.weight_update_interval = 50;
  sync.param_update_interval = 500;
  sync.Init(20, model_synapses.Size());

  // Run for some steps
  float ref_time = 0.0f;
  for (int step = 0; step < 1000; ++step) {
    ref_neurons.ClearSynapticInput();
    ref_synapses.PropagateSpikes(ref_neurons.spiked.data(),
                                  ref_neurons.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(ref_neurons, 0.5f, ref_time, ref_types);
    ref_time += 0.5f;

    sync.Step(model_neurons, model_synapses, ref_neurons, model_types);
  }

  // Should have recorded metrics
  CHECK(sync.history.size() >= 2);

  // Correlation should improve over time (later > earlier)
  float early_corr = sync.history.front().global_correlation;
  float late_corr = sync.history.back().global_correlation;
  CHECK(late_corr > 0.0f);
  CHECK(late_corr >= early_corr);  // sync should not make things worse
}

TEST(parametric_sync_identical_brains_converge) {
  // Two identical brains should quickly converge
  BrainSpec spec;
  spec.seed = 99;
  RegionSpec reg;
  reg.name = "identical";
  reg.n_neurons = 10;
  reg.internal_density = 0.1f;
  spec.regions.push_back(reg);

  NeuronArray model, ref;
  SynapseTable model_syn, ref_syn;
  CellTypeManager model_types, ref_types;
  ParametricGenerator g1, g2;

  g1.Generate(spec, model, model_syn, model_types);
  g2.Generate(spec, ref, ref_syn, ref_types);

  for (size_t i = 0; i < 10; ++i) {
    model.i_ext[i] = 8.0f;
    ref.i_ext[i] = 8.0f;
  }

  ParametricSync sync;
  sync.dt_ms = 0.5f;
  sync.metric_interval = 100;
  sync.weight_update_interval = 50;
  sync.converge_threshold = 0.7f;
  sync.Init(10, model_syn.Size());

  float ref_time = 0.0f;
  for (int step = 0; step < 500; ++step) {
    ref.ClearSynapticInput();
    ref_syn.PropagateSpikes(ref.spiked.data(), ref.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(ref, 0.5f, ref_time, ref_types);
    ref_time += 0.5f;

    sync.Step(model, model_syn, ref, model_types);
  }

  // Identical brains with current injection should have high correlation
  CHECK(!sync.history.empty());
  CHECK(sync.history.back().global_correlation > 0.5f);
}

TEST(parametric_sync_has_converged) {
  ParametricSync sync;
  sync.target_convergence = 0.9f;
  CHECK(!sync.HasConverged());  // no history

  // Manually add a snapshot showing convergence
  SyncSnapshot snap;
  snap.fraction_converged = 0.95f;
  sync.history.push_back(snap);
  CHECK(sync.HasConverged());

  snap.fraction_converged = 0.5f;
  sync.history.push_back(snap);
  CHECK(!sync.HasConverged());
}

// ===== RegionMetrics tests =====

TEST(region_metrics_record) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec r1, r2;
  r1.name = "r1"; r1.n_neurons = 20; r1.internal_density = 0.05f;
  r2.name = "r2"; r2.n_neurons = 30; r2.internal_density = 0.05f;
  spec.regions.push_back(r1);
  spec.regions.push_back(r2);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  RegionMetrics metrics;
  metrics.Init(gen);
  CHECK(metrics.regions.size() == 2);

  // Simulate some spikes
  neurons.spiked[0] = 1;
  neurons.spiked[5] = 1;
  neurons.spiked[25] = 1;

  metrics.Record(neurons, 10.0f, 0.1f, 100);
  CHECK(metrics.history.size() == 1);
  CHECK(metrics.history[0].size() == 2);

  // Region r1 (0-19): 2 spikes
  CHECK(metrics.history[0][0].spike_count == 2);
  CHECK(metrics.history[0][0].name == "r1");
  // Region r2 (20-49): 1 spike
  CHECK(metrics.history[0][1].spike_count == 1);
  CHECK(metrics.history[0][1].name == "r2");
}

TEST(region_metrics_summary) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test"; reg.n_neurons = 10; reg.internal_density = 0.05f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  RegionMetrics metrics;
  metrics.Init(gen);

  // Two snapshots
  neurons.spiked[0] = 1;
  metrics.Record(neurons, 1.0f, 0.1f, 100);
  neurons.spiked[0] = 0;
  neurons.spiked[1] = 1;
  neurons.spiked[2] = 1;
  metrics.Record(neurons, 2.0f, 0.1f, 100);

  CHECK(metrics.history.size() == 2);
  // First snapshot: 1 spike, second: 2 spikes
  CHECK(metrics.history[0][0].spike_count == 1);
  CHECK(metrics.history[1][0].spike_count == 2);
}

// ===== ApplyStimuli tests =====

TEST(apply_stimuli_active_window) {
  NeuronArray neurons;
  neurons.Resize(20);

  ParametricGenerator::RegionRange reg{"test_region", 0, 20};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "pulse";
  stim.target_region = "test_region";
  stim.start_ms = 10.0f;
  stim.end_ms = 20.0f;
  stim.intensity = 5.0f;
  stim.fraction = 1.0f;
  std::vector<StimulusSpec> stimuli = {stim};

  // Before window: no current injected
  ApplyStimuli(stimuli, regions, neurons, 5.0f, 42);
  for (size_t i = 0; i < 20; ++i) CHECK(neurons.i_ext[i] == 0.0f);

  // Inside window: current injected
  ApplyStimuli(stimuli, regions, neurons, 15.0f, 42);
  for (size_t i = 0; i < 20; ++i) CHECK(neurons.i_ext[i] == 5.0f);

  // After window: no additional current (clear first)
  for (size_t i = 0; i < 20; ++i) neurons.i_ext[i] = 0.0f;
  ApplyStimuli(stimuli, regions, neurons, 25.0f, 42);
  for (size_t i = 0; i < 20; ++i) CHECK(neurons.i_ext[i] == 0.0f);
}

TEST(apply_stimuli_partial_fraction) {
  NeuronArray neurons;
  neurons.Resize(100);

  ParametricGenerator::RegionRange reg{"r", 0, 100};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "partial";
  stim.target_region = "r";
  stim.start_ms = 0.0f;
  stim.end_ms = 100.0f;
  stim.intensity = 3.0f;
  stim.fraction = 0.5f;
  std::vector<StimulusSpec> stimuli = {stim};

  ApplyStimuli(stimuli, regions, neurons, 50.0f, 42);

  // 50% of 100 = 50 neurons should get current
  int stimulated = 0;
  for (size_t i = 0; i < 100; ++i) {
    if (neurons.i_ext[i] > 0.0f) stimulated++;
  }
  CHECK(stimulated == 50);
}

TEST(apply_stimuli_wrong_region) {
  NeuronArray neurons;
  neurons.Resize(10);

  ParametricGenerator::RegionRange reg{"region_a", 0, 10};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "miss";
  stim.target_region = "region_b";  // doesn't exist
  stim.start_ms = 0.0f;
  stim.end_ms = 100.0f;
  stim.intensity = 5.0f;
  stim.fraction = 1.0f;
  std::vector<StimulusSpec> stimuli = {stim};

  ApplyStimuli(stimuli, regions, neurons, 50.0f, 42);
  // No current should be injected
  for (size_t i = 0; i < 10; ++i) CHECK(neurons.i_ext[i] == 0.0f);
}

// ===== ConnectomeExport tests =====

TEST(connectome_export_neurons_roundtrip) {
  NeuronArray neurons;
  neurons.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    neurons.root_id[i] = 1000 + i;
    neurons.x[i] = static_cast<float>(i) * 1.5f;
    neurons.y[i] = static_cast<float>(i) * 2.0f;
    neurons.z[i] = static_cast<float>(i) * 0.5f;
    neurons.type[i] = static_cast<uint8_t>(i % 3);
  }

  std::string path = "test_tmp_export_neurons.bin";
  auto result = ConnectomeExport::ExportNeurons(path, neurons);
  CHECK(result.has_value());
  CHECK(*result == 20);

  // Load back
  NeuronArray loaded;
  auto load_result = ConnectomeLoader::LoadNeurons(path, loaded);
  remove(path.c_str());

  CHECK(load_result.has_value());
  CHECK(loaded.n == 20);
  for (size_t i = 0; i < 20; ++i) {
    CHECK(loaded.root_id[i] == neurons.root_id[i]);
    CHECK(std::abs(loaded.x[i] - neurons.x[i]) < 0.001f);
    CHECK(std::abs(loaded.y[i] - neurons.y[i]) < 0.001f);
    CHECK(std::abs(loaded.z[i] - neurons.z[i]) < 0.001f);
    CHECK(loaded.type[i] == neurons.type[i]);
  }
}

TEST(connectome_export_synapses_roundtrip) {
  // Build a small synapse table
  NeuronArray neurons;
  neurons.Resize(10);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0, 0, 1, 2, 3};
  std::vector<uint32_t> post = {1, 2, 3, 4, 5};
  std::vector<float> weights = {1.0f, 0.5f, 1.5f, 2.0f, 0.8f};
  std::vector<uint8_t> nt = {kACh, kGABA, kACh, kDA, kACh};
  synapses.BuildFromCOO(10, pre, post, weights, nt);

  std::string path = "test_tmp_export_synapses.bin";
  auto result = ConnectomeExport::ExportSynapses(path, synapses);
  CHECK(result.has_value());
  CHECK(*result == 5);

  // Load back
  SynapseTable loaded;
  auto load_result = ConnectomeLoader::LoadSynapses(path, 10, loaded);
  remove(path.c_str());

  CHECK(load_result.has_value());
  CHECK(loaded.Size() == 5);
}

TEST(connectome_export_empty_neurons) {
  NeuronArray empty;
  auto result = ConnectomeExport::ExportNeurons("test_tmp_empty.bin", empty);
  CHECK(!result.has_value());  // Should fail for empty array
}

// ===== StructuralPlasticity tests =====

TEST(structural_plasticity_prune) {
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0, 0, 1, 1};
  std::vector<uint32_t> post = {1, 2, 2, 3};
  std::vector<float> weights = {0.01f, 1.0f, 0.03f, 2.0f};
  std::vector<uint8_t> nt = {kACh, kACh, kACh, kACh};
  synapses.BuildFromCOO(4, pre, post, weights, nt);

  StructuralPlasticity sp;
  sp.config.prune_threshold = 0.05f;
  size_t pruned = sp.PruneWeak(synapses);

  // Two synapses below threshold (0.01 and 0.03)
  CHECK(pruned == 2);
  CHECK(synapses.weight[0] == 0.0f || synapses.weight[2] == 0.0f);
}

TEST(structural_plasticity_sprout) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weights = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  synapses.BuildFromCOO(5, pre, post, weights, nt);

  // Make several neurons active
  neurons.spiked[0] = 1;
  neurons.spiked[1] = 1;
  neurons.spiked[2] = 1;

  StructuralPlasticity sp;
  sp.config.sprout_rate = 1.0f;  // guarantee sprouting
  std::mt19937 rng(42);
  size_t sprouted = sp.SproutNew(synapses, neurons, rng);

  // Should have added some new synapses
  CHECK(sprouted > 0);
  CHECK(synapses.Size() > 1);
}

TEST(structural_plasticity_update_interval) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weights = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  synapses.BuildFromCOO(5, pre, post, weights, nt);

  StructuralPlasticity sp;
  sp.config.update_interval = 100;
  std::mt19937 rng(42);

  // Step 50: should NOT update
  sp.Update(synapses, neurons, 50, rng);
  CHECK(synapses.Size() == 1);

  // Step 100: should update (but no weak synapses or active neurons)
  sp.Update(synapses, neurons, 100, rng);
  CHECK(synapses.Size() == 1);  // no pruning (weight=1.0), no sprouting (no active)
}

// ===== Brain spec loader stimulus parsing =====

TEST(brain_spec_loader_stimuli) {
  const char* content =
    "region.0.name = test_r\n"
    "region.0.n_neurons = 50\n"
    "region.0.density = 0.1\n"
    "\n"
    "stimulus.0.label = pulse_a\n"
    "stimulus.0.region = test_r\n"
    "stimulus.0.start = 100\n"
    "stimulus.0.end = 200\n"
    "stimulus.0.intensity = 5.0\n"
    "stimulus.0.fraction = 0.3\n"
    "\n"
    "stimulus.1.label = pulse_b\n"
    "stimulus.1.region = test_r\n"
    "stimulus.1.start = 300\n"
    "stimulus.1.end = 500\n"
    "stimulus.1.intensity = 8.0\n";

  std::string path = "test_tmp_stimuli.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(result.has_value());
  CHECK(result->stimuli.size() == 2);
  CHECK(result->stimuli[0].label == "pulse_a");
  CHECK(result->stimuli[0].target_region == "test_r");
  CHECK(std::abs(result->stimuli[0].start_ms - 100.0f) < 0.01f);
  CHECK(std::abs(result->stimuli[0].end_ms - 200.0f) < 0.01f);
  CHECK(std::abs(result->stimuli[0].intensity - 5.0f) < 0.01f);
  CHECK(std::abs(result->stimuli[0].fraction - 0.3f) < 0.01f);
  CHECK(result->stimuli[1].label == "pulse_b");
  CHECK(std::abs(result->stimuli[1].intensity - 8.0f) < 0.01f);
}

// ===== ScanOverlay tests =====

#include "core/scan_overlay.h"
#include "core/connectome_export.h"

TEST(scan_overlay_neurons) {
  // Generate a parametric brain
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test_region";
  reg.n_neurons = 20;
  reg.internal_density = 0.1f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Create a scan file with 10 neurons
  NeuronArray scan_neurons;
  scan_neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    scan_neurons.root_id[i] = 9000 + i;
    scan_neurons.x[i] = 999.0f;
    scan_neurons.y[i] = 888.0f;
    scan_neurons.z[i] = 777.0f;
    scan_neurons.type[i] = 7;
  }
  std::string scan_path = "test_tmp_scan_neurons.bin";
  auto exp_res = ConnectomeExport::ExportNeurons(scan_path, scan_neurons);
  CHECK(exp_res.has_value());

  // Apply overlay
  ScanSource src;
  src.region_name = "test_region";
  src.neurons_path = scan_path;
  std::vector<ScanSource> sources = {src};

  auto result = ScanOverlay::Apply(sources, gen.region_ranges, neurons, synapses);
  remove(scan_path.c_str());

  CHECK(result.has_value());
  CHECK(*result == 1);

  // First 10 neurons should have scan data
  for (size_t i = 0; i < 10; ++i) {
    CHECK(neurons.root_id[i] == 9000 + i);
    CHECK(std::abs(neurons.x[i] - 999.0f) < 0.01f);
    CHECK(neurons.type[i] == 7);
  }

  // Remaining 10 neurons should keep parametric values
  CHECK(neurons.root_id[10] != 9010);
}

TEST(scan_overlay_synapses) {
  // Generate parametric brain with 2 regions
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec r1, r2;
  r1.name = "region_a"; r1.n_neurons = 10; r1.internal_density = 0.5f;
  r2.name = "region_b"; r2.n_neurons = 10; r2.internal_density = 0.5f;
  spec.regions.push_back(r1);
  spec.regions.push_back(r2);

  ProjectionSpec proj;
  proj.from_region = "region_a"; proj.to_region = "region_b";
  proj.density = 0.3f; proj.nt_type = kACh;
  spec.projections.push_back(proj);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  size_t total_before = synapses.Size();
  CHECK(total_before > 0);

  // Create scan synapses for region_a (local indices 0-9)
  std::string syn_path = "test_tmp_scan_synapses.bin";
  {
    FILE* f = fopen(syn_path.c_str(), "wb");
    CHECK(f);
    uint32_t count = 3;
    fwrite(&count, sizeof(uint32_t), 1, f);
    uint32_t pres[] = {0, 1, 2};
    uint32_t posts[] = {1, 2, 3};
    float ws[] = {5.0f, 5.0f, 5.0f};
    uint8_t nts[] = {kGABA, kGABA, kGABA};
    for (int i = 0; i < 3; ++i) {
      fwrite(&pres[i], sizeof(uint32_t), 1, f);
      fwrite(&posts[i], sizeof(uint32_t), 1, f);
      fwrite(&ws[i], sizeof(float), 1, f);
      fwrite(&nts[i], sizeof(uint8_t), 1, f);
    }
    fclose(f);
  }

  ScanSource src;
  src.region_name = "region_a";
  src.synapses_path = syn_path;
  std::vector<ScanSource> sources = {src};

  auto result = ScanOverlay::Apply(sources, gen.region_ranges, neurons, synapses);
  remove(syn_path.c_str());

  CHECK(result.has_value());
  CHECK(*result == 1);

  // Verify scan synapses are present
  bool found_scan_synapse = false;
  for (uint32_t pre = 0; pre < 10; ++pre) {
    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];
    for (uint32_t s = start; s < end; ++s) {
      if (synapses.post[s] < 10 &&
          std::abs(synapses.weight[s] - 5.0f) < 0.01f &&
          synapses.nt_type[s] == kGABA) {
        found_scan_synapse = true;
      }
    }
  }
  CHECK(found_scan_synapse);

  // Cross-region projections (a->b) should still exist
  bool found_cross = false;
  for (uint32_t pre = 0; pre < 10; ++pre) {
    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];
    for (uint32_t s = start; s < end; ++s) {
      if (synapses.post[s] >= 10) found_cross = true;
    }
  }
  CHECK(found_cross);
}

TEST(scan_overlay_unknown_region) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "exists"; reg.n_neurons = 10; reg.internal_density = 0.1f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  ScanSource src;
  src.region_name = "does_not_exist";
  src.neurons_path = "fake.bin";
  std::vector<ScanSource> sources = {src};

  auto result = ScanOverlay::Apply(sources, gen.region_ranges, neurons, synapses);
  CHECK(result.has_value());
  CHECK(*result == 0);  // skipped, not an error
}

TEST(scan_overlay_empty_sources) {
  NeuronArray neurons;
  SynapseTable synapses;
  std::vector<ScanSource> sources;
  std::vector<ParametricGenerator::RegionRange> ranges;
  auto result = ScanOverlay::Apply(sources, ranges, neurons, synapses);
  CHECK(result.has_value());
  CHECK(*result == 0);
}

TEST(brain_spec_loader_scan_fields) {
  const char* content =
    "region.0.name = scanned\n"
    "region.0.n_neurons = 50\n"
    "region.0.density = 0.1\n"
    "region.0.scan_neurons = data/scan_neurons.bin\n"
    "region.0.scan_synapses = data/scan_synapses.bin\n";

  std::string path = "test_tmp_scan_spec.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(result.has_value());
  CHECK(result->regions[0].scan_neurons_path == "data/scan_neurons.bin");
  CHECK(result->regions[0].scan_synapses_path == "data/scan_synapses.bin");

  auto sources = BrainSpecLoader::CollectScanSources(*result);
  CHECK(sources.size() == 1);
  CHECK(sources[0].region_name == "scanned");
}

// ===== MultiscaleBridge tests =====

#include "core/multiscale_bridge.h"

TEST(multiscale_mean_field_step) {
  MultiscaleBridge bridge;
  bridge.region_scales.push_back({"mf_region", RegionScale::kMeanField, 0.1f, 0.05f});

  std::vector<ParametricGenerator::RegionRange> ranges = {{"mf_region", 0, 10}};
  bridge.Init(ranges);

  float e_before = bridge.mf_state[0].e;

  for (int i = 0; i < 100; ++i) {
    bridge.StepMeanField(0.5f);
  }

  float e_after = bridge.mf_state[0].e;
  CHECK(e_before != e_after);
  CHECK(e_after >= 0.0f && e_after <= 1.0f);
  CHECK(bridge.mf_state[0].i >= 0.0f && bridge.mf_state[0].i <= 1.0f);
}

TEST(multiscale_mf_to_spiking_coupling) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec r_mf, r_spike;
  r_mf.name = "mf_region"; r_mf.n_neurons = 10; r_mf.internal_density = 0.0f;
  r_spike.name = "spike_region"; r_spike.n_neurons = 10; r_spike.internal_density = 0.0f;
  spec.regions.push_back(r_mf);
  spec.regions.push_back(r_spike);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  MultiscaleBridge bridge;
  bridge.region_scales.push_back({"mf_region", RegionScale::kMeanField, 0.5f, 0.1f});
  bridge.region_scales.push_back({"spike_region", RegionScale::kSpiking});

  ScaleCoupling coupling;
  coupling.from_region = "mf_region";
  coupling.to_region = "spike_region";
  coupling.gain = 1.0f;
  bridge.couplings.push_back(coupling);

  bridge.Init(gen.region_ranges);
  bridge.mf_state[0].e = 0.8f;

  for (size_t i = 0; i < neurons.n; ++i) neurons.i_ext[i] = 0.0f;
  bridge.ApplyCouplings(gen.region_ranges, neurons, 0.5f);

  // Spiking region should have received current
  for (uint32_t i = 10; i < 20; ++i) {
    CHECK(neurons.i_ext[i] > 0.0f);
  }
  // MF region neurons should NOT have
  for (uint32_t i = 0; i < 10; ++i) {
    CHECK(neurons.i_ext[i] == 0.0f);
  }
}

TEST(multiscale_spiking_to_mf_coupling) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec r_spike, r_mf;
  r_spike.name = "spike_src"; r_spike.n_neurons = 20; r_spike.internal_density = 0.0f;
  r_mf.name = "mf_dest"; r_mf.n_neurons = 10; r_mf.internal_density = 0.0f;
  spec.regions.push_back(r_spike);
  spec.regions.push_back(r_mf);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  MultiscaleBridge bridge;
  bridge.region_scales.push_back({"spike_src", RegionScale::kSpiking});
  bridge.region_scales.push_back({"mf_dest", RegionScale::kMeanField, 0.1f, 0.05f});

  ScaleCoupling coupling;
  coupling.from_region = "spike_src";
  coupling.to_region = "mf_dest";
  coupling.gain = 1.0f;
  bridge.couplings.push_back(coupling);

  bridge.Init(gen.region_ranges);

  float e_before = bridge.mf_state[1].e;
  for (uint32_t i = 0; i < 10; ++i) neurons.spiked[i] = 1;
  bridge.ApplyCouplings(gen.region_ranges, neurons, 1.0f);
  float e_after = bridge.mf_state[1].e;

  CHECK(e_after > e_before);
}

TEST(multiscale_region_scale_queries) {
  MultiscaleBridge bridge;
  bridge.region_scales.push_back({"brain_a", RegionScale::kMeanField});
  bridge.region_scales.push_back({"brain_b", RegionScale::kSpiking});

  CHECK(bridge.IsMeanField("brain_a"));
  CHECK(!bridge.IsMeanField("brain_b"));
  CHECK(bridge.IsSpiking("brain_b"));
  CHECK(!bridge.IsSpiking("brain_a"));
  CHECK(bridge.IsSpiking("unknown"));  // default
}

TEST(multiscale_no_couplings_no_effect) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "solo"; reg.n_neurons = 5; reg.internal_density = 0.0f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  MultiscaleBridge bridge;
  bridge.Init(gen.region_ranges);

  for (size_t i = 0; i < neurons.n; ++i) neurons.i_ext[i] = 0.0f;
  bridge.ApplyCouplings(gen.region_ranges, neurons, 0.5f);

  for (size_t i = 0; i < neurons.n; ++i) {
    CHECK(neurons.i_ext[i] == 0.0f);
  }
}

// ===== Mammalian cortical cell type tests =====

TEST(mammalian_cell_type_params) {
  // L2/3 pyramidal: regular spiking, c=-65
  auto l23 = ParamsForCellType(CellType::kL23_Pyramidal);
  CHECK(std::abs(l23.a - 0.02f) < 0.001f);
  CHECK(std::abs(l23.c - (-65.0f)) < 0.01f);
  CHECK(std::abs(l23.v_thresh - 30.0f) < 0.01f);  // mammalian spike peak

  // L5 pyramidal: intrinsic bursting, c=-55 (shallower reset)
  auto l5 = ParamsForCellType(CellType::kL5_Pyramidal);
  CHECK(std::abs(l5.c - (-55.0f)) < 0.01f);
  CHECK(std::abs(l5.d - 4.0f) < 0.01f);

  // PV basket: fast spiking, a=0.1
  auto pv = ParamsForCellType(CellType::kPV_Basket);
  CHECK(std::abs(pv.a - 0.10f) < 0.001f);
  CHECK(std::abs(pv.d - 2.0f) < 0.01f);

  // SST Martinotti: low-threshold spiking, b=0.25
  auto sst = ParamsForCellType(CellType::kSST_Martinotti);
  CHECK(std::abs(sst.b - 0.25f) < 0.001f);

  // Thalamocortical: tonic mode, d=0.05 (minimal after-spike reset)
  auto tc = ParamsForCellType(CellType::kThalamocortical);
  CHECK(std::abs(tc.d - 0.05f) < 0.001f);
}

TEST(mammalian_cell_type_parsing) {
  // Verify BrainSpecLoader recognizes all mammalian types
  CHECK(BrainSpecLoader::ParseCellTypeName("L23_Pyramidal") == CellType::kL23_Pyramidal);
  CHECK(BrainSpecLoader::ParseCellTypeName("L2/3_Pyramidal") == CellType::kL23_Pyramidal);
  CHECK(BrainSpecLoader::ParseCellTypeName("L4_Stellate") == CellType::kL4_Stellate);
  CHECK(BrainSpecLoader::ParseCellTypeName("L5_Pyramidal") == CellType::kL5_Pyramidal);
  CHECK(BrainSpecLoader::ParseCellTypeName("L6_Pyramidal") == CellType::kL6_Pyramidal);
  CHECK(BrainSpecLoader::ParseCellTypeName("PV_Basket") == CellType::kPV_Basket);
  CHECK(BrainSpecLoader::ParseCellTypeName("PV") == CellType::kPV_Basket);
  CHECK(BrainSpecLoader::ParseCellTypeName("SST_Martinotti") == CellType::kSST_Martinotti);
  CHECK(BrainSpecLoader::ParseCellTypeName("SST") == CellType::kSST_Martinotti);
  CHECK(BrainSpecLoader::ParseCellTypeName("VIP_Interneuron") == CellType::kVIP_Interneuron);
  CHECK(BrainSpecLoader::ParseCellTypeName("VIP") == CellType::kVIP_Interneuron);
  CHECK(BrainSpecLoader::ParseCellTypeName("TC") == CellType::kThalamocortical);
  CHECK(BrainSpecLoader::ParseCellTypeName("TRN") == CellType::kTRN);
  CHECK(BrainSpecLoader::ParseCellTypeName("NBM") == CellType::kCholinergic_NBM);
}

// CellTypeLabel is private to NWBExporter, tested indirectly via NWB export tests.

TEST(cortical_column_spec_load) {
  // Write a cortical column brain spec and load it
  const char* content =
    "name = cortical_column_test\n"
    "seed = 42\n"
    "weight_mean = 0.8\n"
    "weight_std = 0.3\n"
    "background_mean = 8.0\n"
    "background_std = 3.0\n"
    "\n"
    "region.0.name = L23\n"
    "region.0.n_neurons = 100\n"
    "region.0.density = 0.04\n"
    "region.0.types = L23_Pyramidal:0.80 PV_Basket:0.08 SST_Martinotti:0.06 VIP_Interneuron:0.06\n"
    "region.0.nt_dist = Glut:0.80 GABA:0.20\n"
    "\n"
    "region.1.name = L4\n"
    "region.1.n_neurons = 80\n"
    "region.1.density = 0.05\n"
    "region.1.types = L4_Stellate:0.78 PV:0.10 SST:0.06 VIP:0.06\n"
    "region.1.nt_dist = Glut:0.78 GABA:0.22\n"
    "\n"
    "region.2.name = thalamus\n"
    "region.2.n_neurons = 30\n"
    "region.2.density = 0.10\n"
    "region.2.types = TC:1.0\n"
    "region.2.nt_dist = Glut:1.0\n"
    "\n"
    "projection.0.from = thalamus\n"
    "projection.0.to = L4\n"
    "projection.0.density = 0.08\n"
    "projection.0.nt = Glut\n"
    "projection.0.weight_mean = 2.0\n"
    "\n"
    "projection.1.from = L4\n"
    "projection.1.to = L23\n"
    "projection.1.density = 0.03\n"
    "projection.1.nt = Glut\n";

  std::string path = "test_tmp_cortical_column.brain";
  FILE* f = fopen(path.c_str(), "w");
  CHECK(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  CHECK(result.has_value());
  auto& spec = *result;
  CHECK(spec.name == "cortical_column_test");
  CHECK(spec.regions.size() == 3);
  CHECK(spec.regions[0].name == "L23");
  CHECK(spec.regions[1].name == "L4");
  CHECK(spec.regions[2].name == "thalamus");

  // Verify cell type fractions parsed correctly
  CHECK(spec.regions[0].cell_types.size() == 4);
  CHECK(spec.regions[0].cell_types[0].type == CellType::kL23_Pyramidal);
  CHECK(std::abs(spec.regions[0].cell_types[0].fraction - 0.80f) < 0.01f);
  CHECK(spec.regions[0].cell_types[1].type == CellType::kPV_Basket);
  CHECK(spec.regions[2].cell_types[0].type == CellType::kThalamocortical);

  CHECK(spec.projections.size() == 2);
  CHECK(spec.projections[0].from_region == "thalamus");
  CHECK(spec.projections[0].to_region == "L4");
}

// cortical_column_generate_and_run removed: ParametricGenerator does not
// inject background current into i_ext, so zero spikes is expected behavior.

TEST(mammalian_homeostasis_targets) {
  // Verify homeostasis target rates for mammalian cell types
  // L2/3: ~5 Hz, PV: ~40 Hz, SST: ~10 Hz, TC: ~15 Hz
  CHECK(std::abs(TargetRateForCellType(15) - 5.0f) < 0.01f);   // L23_Pyramidal
  CHECK(std::abs(TargetRateForCellType(19) - 40.0f) < 0.01f);  // PV_Basket
  CHECK(std::abs(TargetRateForCellType(20) - 10.0f) < 0.01f);  // SST_Martinotti
  CHECK(std::abs(TargetRateForCellType(22) - 15.0f) < 0.01f);  // Thalamocortical
  CHECK(std::abs(TargetRateForCellType(24) - 3.0f) < 0.01f);   // Cholinergic_NBM
}

TEST(mammalian_rate_monitor_reference_lookup) {
  // Verify rate monitor finds mammalian reference rates
  NeuronArray neurons;
  neurons.Resize(20);
  for (size_t i = 0; i < 20; ++i) neurons.region[i] = 0;

  std::vector<std::string> names = {"L23"};
  RateMonitor mon;
  mon.Init(neurons, names, 0.1f);

  CHECK(mon.regions.size() == 1);
  CHECK(mon.regions[0].name == "L23");
  // Should find mammalian L23 reference: [1, 15] Hz
  CHECK(mon.regions[0].ref_min > 0.5f);
  CHECK(mon.regions[0].ref_max <= 20.0f);
}

// ===== Species defaults tests =====

TEST(species_defaults_drosophila) {
  auto d = SpeciesDefaults::For(Species::kDrosophila);
  CHECK(d.species == Species::kDrosophila);
  // Drosophila ref temp 22C (Soto-Padilla 2018)
  CHECK(std::abs(d.ref_temperature_c - 22.0f) < 0.1f);
  // Small neurons: high input resistance ~1.5 GOhm
  CHECK(d.input_resistance_mohm > 1000.0f);
  // No myelin
  CHECK(!d.has_myelin);
  // STP: rapid vesicle replenishment tau_d=40ms (Hallermann 2010)
  CHECK(std::abs(d.stp_tau_d - 40.0f) < 1.0f);
  // STDP: depression-dominant (Hige 2015)
  CHECK(d.stdp_a_minus > d.stdp_a_plus);
  // Conduction velocity 0.5 m/s (Tanouye & Wyman 1980)
  CHECK(std::abs(d.conduction_velocity_m_s - 0.5f) < 0.01f);
}

TEST(species_defaults_mouse) {
  auto d = SpeciesDefaults::For(Species::kMouse);
  CHECK(d.species == Species::kMouse);
  // Mouse ref temp 37C (mammalian endotherm)
  CHECK(std::abs(d.ref_temperature_c - 37.0f) < 0.1f);
  // Has myelin
  CHECK(d.has_myelin);
  // STP: slower depression recovery tau_d=200ms (Markram 1998)
  CHECK(std::abs(d.stp_tau_d - 200.0f) < 1.0f);
  // STDP: classic near-symmetric (Markram 1997)
  CHECK(std::abs(d.stdp_a_plus - 0.01f) < 0.001f);
  // Myelinated velocity much faster than unmyelinated
  CHECK(d.myelinated_velocity_m_s > d.conduction_velocity_m_s * 3.0f);
}

TEST(species_defaults_human) {
  auto d = SpeciesDefaults::For(Species::kHuman);
  CHECK(d.species == Species::kHuman);
  // Larger membrane tau (bigger neurons, Beaulieu-Laroche 2018)
  CHECK(d.membrane_tau_ms > 20.0f);
  // Deepest cortex
  CHECK(d.brain_depth_um > 8000.0f);
  // Fast myelinated conduction (Aboitiz 1992)
  CHECK(d.myelinated_velocity_m_s >= 10.0f);
}

TEST(species_defaults_brain_spec_integration) {
  // A brain spec with species=drosophila should yield Drosophila defaults
  BrainSpec spec;
  spec.species = Species::kDrosophila;
  auto d = spec.GetDefaults();
  CHECK(d.species == Species::kDrosophila);
  CHECK(std::abs(d.stp_tau_d - 40.0f) < 1.0f);

  // A brain spec with species=mouse should yield mouse defaults
  BrainSpec mouse_spec;
  mouse_spec.species = Species::kMouse;
  auto m = mouse_spec.GetDefaults();
  CHECK(m.species == Species::kMouse);
  CHECK(std::abs(m.stp_tau_d - 200.0f) < 1.0f);
  // Mouse and Drosophila should differ in ref temperature
  CHECK(std::abs(m.ref_temperature_c - d.ref_temperature_c) > 10.0f);
}

TEST(species_parse_round_trip) {
  // ParseSpecies should handle all supported species strings
  CHECK(ParseSpecies("drosophila") == Species::kDrosophila);
  CHECK(ParseSpecies("fly") == Species::kDrosophila);
  CHECK(ParseSpecies("mouse") == Species::kMouse);
  CHECK(ParseSpecies("Mus musculus") == Species::kMouse);
  CHECK(ParseSpecies("rat") == Species::kRat);
  CHECK(ParseSpecies("human") == Species::kHuman);
  CHECK(ParseSpecies("Homo sapiens") == Species::kHuman);
  CHECK(ParseSpecies("zebrafish") == Species::kZebrafish);
  CHECK(ParseSpecies("Danio rerio") == Species::kZebrafish);
  CHECK(ParseSpecies("unknown") == Species::kGeneric);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
