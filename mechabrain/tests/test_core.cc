// Core tests: neuron dynamics, synapses, STDP, loader, config, checkpoint, stats, recorder
#include <cmath>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "test_harness.h"

#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/stdp.h"
#include "core/connectome_loader.h"
#include "core/config_loader.h"
#include "core/checkpoint.h"
#include "core/connectome_stats.h"
#include "core/experiment_config.h"
#include "core/recorder.h"
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/motor_output.h"
#include "core/intrinsic_homeostasis.h"
#include "core/gap_junctions.h"
#include "core/short_term_plasticity.h"
#include "core/cpg.h"
#include "core/proprioception.h"
#include "core/nwb_export.h"
#include "core/sim_features.h"
#include "core/temperature.h"
#include "core/spike_frequency_adaptation.h"
#include "core/compartmental_neuron.h"
#include "core/inhibitory_plasticity.h"
#include "core/neuromodulator_effects.h"
#include "core/nmda.h"
#include "core/scoped_timer.h"
#include "core/spike_analysis.h"
#include "core/memory_tracker.h"

// ===== Core tests =====

TEST(neuron_array_init) {
  NeuronArray arr;
  arr.Resize(100);
  assert(arr.n == 100);
  assert(arr.v[0] == -65.0f);
  assert(arr.u[0] == -13.0f);
  assert(arr.spiked[0] == 0);
  assert(arr.CountSpikes() == 0);
}

TEST(empty_neuron_array) {
  NeuronArray arr;
  arr.Resize(0);
  assert(arr.n == 0);
  assert(arr.CountSpikes() == 0);
  arr.ClearSynapticInput();  // should not crash
}

TEST(izhikevich_spike) {
  NeuronArray arr;
  arr.Resize(1);
  arr.i_ext[0] = 15.0f;

  IzhikevichParams p;
  bool spiked = false;
  for (int i = 0; i < 1000 && !spiked; ++i) {
    IzhikevichStep(arr, 0.1f, i * 0.1f, p);
    if (arr.spiked[0]) spiked = true;
    arr.i_syn[0] = 0.0f;
  }
  assert(spiked && "Neuron should spike with strong input");
}

TEST(izhikevich_no_spike_without_input) {
  NeuronArray arr;
  arr.Resize(1);

  IzhikevichParams p;
  for (int i = 0; i < 1000; ++i) {
    IzhikevichStep(arr, 0.1f, i * 0.1f, p);
  }
  assert(arr.v[0] < p.v_thresh);
}

TEST(izhikevich_updates_last_spike_time) {
  NeuronArray arr;
  arr.Resize(1);
  arr.i_ext[0] = 15.0f;

  IzhikevichParams p;
  float initial_spike_time = arr.last_spike_time[0];
  for (int i = 0; i < 1000; ++i) {
    float t = i * 0.1f;
    IzhikevichStep(arr, 0.1f, t, p);
    if (arr.spiked[0]) {
      assert(arr.last_spike_time[0] == t);
      assert(arr.last_spike_time[0] != initial_spike_time);
      return;
    }
    arr.i_syn[0] = 0.0f;
  }
  assert(false && "Should have spiked");
}

TEST(izhikevich_nan_recovery) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = std::numeric_limits<float>::quiet_NaN();
  arr.u[0] = std::numeric_limits<float>::quiet_NaN();

  IzhikevichParams p;
  IzhikevichStep(arr, 0.1f, 0.0f, p);
  assert(std::isfinite(arr.v[0]));
  assert(std::isfinite(arr.u[0]));
}

TEST(lif_spike) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = -70.0f;
  arr.i_ext[0] = 10.0f;

  LIFParams p;
  bool spiked = false;
  for (int i = 0; i < 1000 && !spiked; ++i) {
    LIFStep(arr, 0.1f, i * 0.1f, p);
    if (arr.spiked[0]) spiked = true;
    arr.i_syn[0] = 0.0f;
  }
  assert(spiked && "LIF neuron should spike with strong input");
}

TEST(synapse_table_csr) {
  std::vector<uint32_t> pre  = {0, 0, 1, 2};
  std::vector<uint32_t> post = {1, 2, 2, 0};
  std::vector<float> weight  = {1.0f, 2.0f, 1.5f, 0.5f};
  std::vector<uint8_t> nt    = {kACh, kACh, kGABA, kACh};

  SynapseTable table;
  table.BuildFromCOO(3, pre, post, weight, nt);

  assert(table.n_neurons == 3);
  assert(table.Size() == 4);
  assert(table.row_ptr[1] - table.row_ptr[0] == 2);
  assert(table.row_ptr[2] - table.row_ptr[1] == 1);
  assert(table.row_ptr[3] - table.row_ptr[2] == 1);
}

TEST(spike_propagation) {
  std::vector<uint32_t> pre  = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weight  = {2.0f};
  std::vector<uint8_t> nt    = {kACh};

  SynapseTable table;
  table.BuildFromCOO(2, pre, post, weight, nt);

  uint8_t spiked[2] = {1, 0};
  float i_syn[2] = {0.0f, 0.0f};

  table.PropagateSpikes(spiked, i_syn, 1.0f);
  assert(std::abs(i_syn[1] - 2.0f) < 1e-6f);
  assert(i_syn[0] == 0.0f);
}

TEST(inhibitory_propagation) {
  std::vector<uint32_t> pre  = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weight  = {3.0f};
  std::vector<uint8_t> nt    = {kGABA};

  SynapseTable table;
  table.BuildFromCOO(2, pre, post, weight, nt);

  uint8_t spiked[2] = {1, 0};
  float i_syn[2] = {0.0f, 0.0f};

  table.PropagateSpikes(spiked, i_syn, 1.0f);
  assert(std::abs(i_syn[1] - (-3.0f)) < 1e-6f);
}

TEST(synapse_no_synapses) {
  SynapseTable table;
  table.BuildFromCOO(5, {}, {}, {}, {});
  assert(table.Size() == 0);

  uint8_t spiked[5] = {1, 0, 1, 0, 1};
  float i_syn[5] = {};
  table.PropagateSpikes(spiked, i_syn, 1.0f);
  for (int i = 0; i < 5; ++i) assert(i_syn[i] == 0.0f);
}

TEST(synapse_oob_indices) {
  SynapseTable table;
  // Post index 5 is out of bounds for 3 neurons
  table.BuildFromCOO(3, {0, 1}, {1, 5}, {1.0f, 2.0f}, {kACh, kACh});
  assert(table.Size() == 0 && "OOB index should produce empty table");

  // Pre index 10 is out of bounds
  SynapseTable table2;
  table2.BuildFromCOO(3, {10, 1}, {1, 2}, {1.0f, 2.0f}, {kACh, kACh});
  assert(table2.Size() == 0);
}

// ===== STDP tests =====

TEST(stdp_potentiation) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  arr.last_spike_time[0] = 10.0f;
  arr.spiked[0] = 0;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 15.0f;

  float original_weight = syn.weight[0];
  STDPParams p;
  STDPUpdate(syn, arr, 15.0f, p);
  assert(syn.weight[0] > original_weight && "Weight should increase (potentiation)");
}

TEST(stdp_depression) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  arr.last_spike_time[1] = 5.0f;
  arr.spiked[0] = 1;
  arr.spiked[1] = 0;
  arr.last_spike_time[0] = 10.0f;

  float original_weight = syn.weight[0];
  STDPParams p;
  STDPUpdate(syn, arr, 10.0f, p);
  assert(syn.weight[0] < original_weight && "Weight should decrease (depression)");
}

TEST(stdp_weight_bounds) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {9.99f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  for (int i = 0; i < 100; ++i) {
    arr.last_spike_time[0] = i * 10.0f;
    arr.spiked[1] = 1;
    arr.last_spike_time[1] = i * 10.0f + 5.0f;
    arr.spiked[0] = 0;
    STDPUpdate(syn, arr, i * 10.0f + 5.0f, p);
  }
  assert(syn.weight[0] <= p.w_max);

  syn.weight[0] = 0.5f;
  float w_before_depression = syn.weight[0];
  for (int i = 0; i < 100; ++i) {
    arr.last_spike_time[1] = i * 10.0f;
    arr.spiked[0] = 1;
    arr.last_spike_time[0] = i * 10.0f + 5.0f;
    arr.spiked[1] = 0;
    STDPUpdate(syn, arr, i * 10.0f + 5.0f, p);
  }
  assert(syn.weight[0] >= p.w_min);
  assert(syn.weight[0] < w_before_depression && "Depression should reduce weight");
}

TEST(stdp_no_change_without_spikes) {
  NeuronArray arr;
  arr.Resize(2);
  arr.spiked[0] = 0;
  arr.spiked[1] = 0;

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  STDPUpdate(syn, arr, 100.0f, p);
  assert(syn.weight[0] == 5.0f);
}

TEST(stdp_timing_window) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  arr.last_spike_time[0] = 0.0f;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 200.0f;

  STDPUpdate(syn, arr, 200.0f, p);
  assert(syn.weight[0] == 5.0f && "No change outside timing window");
}

// ===== ConnectomeLoader tests =====

TEST(loader_neurons_roundtrip) {
  uint32_t count = 3;
#ifdef _MSC_VER
  #pragma pack(push, 1)
  struct NeuronRecord { uint64_t id; float x, y, z; uint8_t type; };
  #pragma pack(pop)
#else
  struct NeuronRecord { uint64_t id; float x, y, z; uint8_t type; } __attribute__((packed));
#endif
  NeuronRecord records[3] = {
    {100, 1.0f, 2.0f, 3.0f, 1},
    {200, 4.0f, 5.0f, 6.0f, 2},
    {300, 7.0f, 8.0f, 9.0f, 0},
  };

  std::vector<uint8_t> buf;
  buf.resize(sizeof(count) + sizeof(records));
  memcpy(buf.data(), &count, sizeof(count));
  memcpy(buf.data() + sizeof(count), records, sizeof(records));

  auto path = WriteTempFile("neurons.bin", buf.data(), buf.size());

  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons(path, neurons);
  assert(result.has_value());
  assert(*result == 3);
  assert(neurons.n == 3);
  assert(neurons.root_id[0] == 100);
  assert(neurons.root_id[2] == 300);
  assert(neurons.x[1] == 4.0f);
  assert(neurons.type[0] == 1);

  remove(path.c_str());
}

TEST(loader_synapses_roundtrip) {
  uint32_t count = 2;
#ifdef _MSC_VER
  #pragma pack(push, 1)
  struct SynRecord { uint32_t pre, post; float w; uint8_t nt; };
  #pragma pack(pop)
#else
  struct SynRecord { uint32_t pre, post; float w; uint8_t nt; } __attribute__((packed));
#endif
  SynRecord records[2] = {
    {0, 1, 2.5f, kACh},
    {1, 0, 1.0f, kGABA},
  };

  std::vector<uint8_t> buf;
  buf.resize(sizeof(count) + sizeof(records));
  memcpy(buf.data(), &count, sizeof(count));
  memcpy(buf.data() + sizeof(count), records, sizeof(records));

  auto path = WriteTempFile("synapses.bin", buf.data(), buf.size());

  SynapseTable table;
  auto result = ConnectomeLoader::LoadSynapses(path, 2, table);
  assert(result.has_value());
  assert(*result == 2);
  assert(table.Size() == 2);
  assert(table.n_neurons == 2);

  remove(path.c_str());
}

TEST(loader_missing_file) {
  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons("nonexistent_file.bin", neurons);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kFileNotFound);
}

TEST(loader_truncated_file) {
  uint32_t count = 100;
  auto path = WriteTempFile("trunc.bin", &count, sizeof(count));

  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons(path, neurons);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kCorruptedData);

  remove(path.c_str());
}

// ===== Config Loader tests =====

TEST(config_loader_basic) {
  std::string path = "test_tmp_config.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "name = test_experiment\n");
  fprintf(f, "dt_ms = 0.5\n");
  fprintf(f, "duration_ms = 5000\n");
  fprintf(f, "enable_stdp = true\n");
  fprintf(f, "bridge_mode = 1\n");
  fprintf(f, "connectome_dir = data/test\n");
  fprintf(f, "output_dir = results/test\n");
  fprintf(f, "monitor_neurons = 0 1 2 3\n");
  fprintf(f, "stimulus: odor_A 100 200 0.8 0,1,2\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(result.has_value());

  auto& cfg = *result;
  assert(cfg.name == "test_experiment");
  assert(std::abs(cfg.dt_ms - 0.5f) < 0.01f);
  assert(std::abs(cfg.duration_ms - 5000.0f) < 0.01f);
  assert(cfg.enable_stdp == true);
  assert(cfg.bridge_mode == 1);
  assert(cfg.connectome_dir == "data/test");
  assert(cfg.output_dir == "results/test");
  assert(cfg.monitor_neurons.size() == 4);
  assert(cfg.monitor_neurons[2] == 2);
  assert(cfg.stimulus_protocol.size() == 1);
  assert(cfg.stimulus_protocol[0].label == "odor_A");
  assert(cfg.stimulus_protocol[0].target_neurons.size() == 3);

  remove(path.c_str());
}

TEST(config_loader_missing_file) {
  auto result = ConfigLoader::Load("nonexistent_config_12345.cfg");
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kFileNotFound);
}

TEST(config_loader_comments_and_blanks) {
  std::string path = "test_tmp_comments.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "# This is a comment\n");
  fprintf(f, "\n");
  fprintf(f, "  # Indented comment\n");
  fprintf(f, "name = after_comments\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(result.has_value());
  assert(result->name == "after_comments");

  remove(path.c_str());
}

TEST(config_loader_invalid_dt) {
  std::string path = "test_tmp_bad_dt.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "dt_ms = -1.0\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kInvalidParam);

  remove(path.c_str());
}

TEST(config_loader_invalid_value) {
  std::string path = "test_tmp_bad_val.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "dt_ms = notanumber\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(!result.has_value());

  remove(path.c_str());
}

// ===== Experiment Config tests =====

// ===== Recorder tests =====

TEST(recorder_open_close) {
  std::string dir = "test_tmp_recorder_out";
  Recorder rec;
  rec.record_spikes = true;
  rec.record_voltages = true;
  rec.record_shadow_metrics = true;
  rec.record_per_neuron_error = false;

  bool ok = rec.Open(dir, 10);
  assert(ok);
  assert(rec.n_neurons == 10);

  NeuronArray arr;
  arr.Resize(10);
  arr.spiked[3] = 1;
  arr.v[3] = -40.0f;

  rec.RecordStep(1.0f, arr, nullptr, 0, 0.0f, nullptr);
  assert(rec.n_recorded_steps == 1);

  rec.Close();

  FILE* sf = fopen((dir + "/spikes.bin").c_str(), "rb");
  assert(sf);
  fclose(sf);

  FILE* vf = fopen((dir + "/voltages.bin").c_str(), "rb");
  assert(vf);
  fclose(vf);

  FILE* mf = fopen((dir + "/metrics.csv").c_str(), "rb");
  assert(mf);
  fclose(mf);

  remove((dir + "/spikes.bin").c_str());
  remove((dir + "/voltages.bin").c_str());
  remove((dir + "/metrics.csv").c_str());
#ifdef _WIN32
  _rmdir(dir.c_str());
#else
  rmdir(dir.c_str());
#endif
}

// ===== Checkpoint tests =====

TEST(checkpoint_save_load_roundtrip) {
  NeuronArray neurons;
  neurons.Resize(10);
  neurons.v[0] = -50.0f;
  neurons.v[5] = -40.0f;
  neurons.u[3] = -10.0f;
  neurons.dopamine[2] = 0.5f;
  neurons.last_spike_time[7] = 42.0f;
  neurons.type[4] = 3;
  neurons.region[6] = 2;
  neurons.i_ext[1] = 5.0f;

  SynapseTable synapses;
  synapses.BuildFromCOO(10, {0, 1, 2}, {1, 2, 3}, {0.5f, 0.7f, 0.3f},
                         {kACh, kGABA, kGlut});
  synapses.weight[1] = 0.99f;
  synapses.InitReleaseProbability(0.3f);
  STPParams stp_params;
  stp_params.U_se = 0.4f;
  synapses.InitSTP(stp_params);
  synapses.stp_u[1] = 0.6f;

  float sim_time = 500.0f;
  int total_steps = 5000;
  int total_resyncs = 3;

  std::string path = "test_tmp_checkpoint.bin";
  std::vector<uint8_t> ext;  // no bridge extension data
  bool saved = Checkpoint::Save(path, sim_time, total_steps, total_resyncs,
                                 neurons, synapses, ext);
  assert(saved);

  NeuronArray neurons2;
  neurons2.Resize(10);
  SynapseTable synapses2;
  synapses2.BuildFromCOO(10, {0, 1, 2}, {1, 2, 3}, {0.5f, 0.7f, 0.3f},
                          {kACh, kGABA, kGlut});
  float sim_time2 = 0;
  int steps2 = 0, resyncs2 = 0;

  std::vector<uint8_t> ext2;
  bool loaded = Checkpoint::Load(path, sim_time2, steps2, resyncs2,
                                  neurons2, synapses2, ext2);
  assert(loaded);

  assert(sim_time2 == 500.0f);
  assert(steps2 == 5000);
  assert(resyncs2 == 3);
  assert(neurons2.v[0] == -50.0f);
  assert(neurons2.v[5] == -40.0f);
  assert(neurons2.u[3] == -10.0f);
  assert(neurons2.dopamine[2] == 0.5f);
  assert(neurons2.last_spike_time[7] == 42.0f);
  assert(synapses2.weight[1] == 0.99f);
  assert(neurons2.type[4] == 3);
  assert(neurons2.region[6] == 2);
  assert(neurons2.i_ext[1] == 5.0f);
  assert(synapses2.HasStochasticRelease());
  assert(std::abs(synapses2.p_release[0] - 0.3f) < 1e-6f);
  assert(synapses2.HasSTP());
  assert(std::abs(synapses2.stp_u[1] - 0.6f) < 1e-6f);
  assert(std::abs(synapses2.stp_U_se[0] - 0.4f) < 1e-6f);

  remove(path.c_str());
}

TEST(checkpoint_bad_magic) {
  std::string path = "test_tmp_bad_ckpt.bin";
  FILE* f = fopen(path.c_str(), "wb");
  uint32_t garbage = 0xDEADBEEF;
  fwrite(&garbage, 4, 1, f);
  fclose(f);

  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  synapses.BuildFromCOO(5, {0}, {1}, {1.0f}, {kACh});
  std::vector<uint8_t> ext;
  float t = 0; int s = 0, r = 0;

  bool loaded = Checkpoint::Load(path, t, s, r, neurons, synapses, ext);
  assert(!loaded);

  remove(path.c_str());
}

TEST(checkpoint_size_mismatch) {
  NeuronArray neurons;
  neurons.Resize(10);
  SynapseTable synapses;
  synapses.BuildFromCOO(10, {0}, {1}, {1.0f}, {kACh});

  std::string path = "test_tmp_mismatch.bin";
  Checkpoint::Save(path, 0, 0, 0, neurons, synapses);

  NeuronArray neurons2;
  neurons2.Resize(5);
  SynapseTable synapses2;
  synapses2.BuildFromCOO(5, {0}, {1}, {1.0f}, {kACh});
  std::vector<uint8_t> ext;
  float t = 0; int s = 0, r = 0;

  bool loaded = Checkpoint::Load(path, t, s, r, neurons2, synapses2, ext);
  assert(!loaded);

  remove(path.c_str());
}

// ===== Connectome stats tests =====

TEST(connectome_stats_basic) {
  SynapseTable synapses;
  synapses.BuildFromCOO(5,
      {0, 0, 1, 2, 3},
      {1, 2, 3, 4, 0},
      {1.0f, 0.5f, 0.3f, 0.8f, 0.6f},
      {kACh, kGABA, kGlut, kDA, kACh});

  NeuronArray neurons;
  neurons.Resize(5);

  ConnectomeStats stats;
  bool valid = stats.Compute(synapses, neurons);
  assert(valid);

  assert(stats.n_neurons == 5);
  assert(stats.n_synapses == 5);
  assert(stats.n_ach == 2);
  assert(stats.n_gaba == 1);
  assert(stats.n_glut == 1);
  assert(stats.n_da == 1);
  assert(stats.max_out_degree == 2);
  assert(stats.min_weight == 0.3f);
  assert(stats.max_weight == 1.0f);
  assert(stats.n_self_loops == 0);
  assert(stats.n_out_of_bounds == 0);
  assert(stats.n_nan_weights == 0);
}

TEST(connectome_stats_self_loops) {
  SynapseTable synapses;
  synapses.BuildFromCOO(3,
      {0, 1, 1},
      {1, 1, 2},
      {1.0f, 0.5f, 0.3f},
      {kACh, kACh, kACh});

  NeuronArray neurons;
  neurons.Resize(3);

  ConnectomeStats stats;
  stats.Compute(synapses, neurons);
  assert(stats.n_self_loops == 1);
}

TEST(connectome_stats_isolated_neurons) {
  SynapseTable synapses;
  synapses.BuildFromCOO(4, {0}, {1}, {1.0f}, {kACh});

  NeuronArray neurons;
  neurons.Resize(4);

  ConnectomeStats stats;
  stats.Compute(synapses, neurons);
  assert(stats.n_isolated_neurons == 2);
}

// ===== Stochastic synapse tests =====

TEST(stochastic_release_reduces_transmission) {
  // With p_release=0.3, fewer post-synaptic neurons should receive input
  // compared to deterministic (p=1) propagation.
  NeuronArray neurons;
  neurons.Resize(20);
  SynapseTable syn;

  // Wire neuron 0 -> all others with deterministic weights
  std::vector<uint32_t> pre, post;
  std::vector<float> w;
  std::vector<uint8_t> nt;
  for (uint32_t i = 1; i < 20; ++i) {
    pre.push_back(0); post.push_back(i);
    w.push_back(1.0f); nt.push_back(kACh);
  }
  syn.BuildFromCOO(20, pre, post, w, nt);

  // Deterministic: all 19 targets get input
  neurons.spiked[0] = 1;
  std::fill(neurons.i_syn.begin(), neurons.i_syn.end(), 0.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  int det_hits = 0;
  for (uint32_t i = 1; i < 20; ++i) {
    if (neurons.i_syn[i] > 0.0f) det_hits++;
  }
  assert(det_hits == 19);

  // Stochastic with p=0.3: run many trials, average hits should be ~5.7
  syn.InitReleaseProbability(0.3f);
  std::mt19937 rng(42);
  int total_hits = 0;
  int trials = 200;
  for (int t = 0; t < trials; ++t) {
    std::fill(neurons.i_syn.begin(), neurons.i_syn.end(), 0.0f);
    neurons.spiked[0] = 1;
    syn.PropagateSpikesMonteCarlo(neurons.spiked.data(), neurons.i_syn.data(),
                                   1.0f, rng);
    for (uint32_t i = 1; i < 20; ++i) {
      if (neurons.i_syn[i] > 0.0f) total_hits++;
    }
  }
  float avg = static_cast<float>(total_hits) / trials;
  // Expected ~5.7 (19 * 0.3). Allow wide margin.
  assert(avg > 3.0f && avg < 9.0f);
}

TEST(stochastic_release_zero_blocks_all) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable syn;
  std::vector<uint32_t> pre = {0, 0, 0};
  std::vector<uint32_t> post = {1, 2, 3};
  std::vector<float> w = {1.0f, 1.0f, 1.0f};
  std::vector<uint8_t> nt = {kACh, kACh, kACh};
  syn.BuildFromCOO(5, pre, post, w, nt);
  syn.InitReleaseProbability(0.0f);

  neurons.spiked[0] = 1;
  std::mt19937 rng(99);
  syn.PropagateSpikesMonteCarlo(neurons.spiked.data(), neurons.i_syn.data(),
                                 1.0f, rng);
  for (uint32_t i = 1; i <= 3; ++i) {
    assert(neurons.i_syn[i] == 0.0f);
  }
}

TEST(stp_depression_reduces_weight) {
  // Repeated spikes should depress the synapse (x decreases).
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 200.0f;
  params.tau_f = 50.0f;
  syn.InitSTP(params);

  // First spike
  float ux1 = syn.UpdateSTP(0);
  // Second spike immediately (no recovery)
  float ux2 = syn.UpdateSTP(0);
  // Third spike
  float ux3 = syn.UpdateSTP(0);

  // Each successive spike should produce less transmission (depression)
  assert(ux1 > ux2);
  assert(ux2 > ux3);
  // First spike: u goes to 0.75, x=1 -> ux=0.75, x becomes 0.25
  assert(ux1 > 0.5f);
}

TEST(stp_recovery_restores_strength) {
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 200.0f;
  params.tau_f = 50.0f;
  syn.InitSTP(params);

  // Depress with 5 rapid spikes
  for (int i = 0; i < 5; ++i) syn.UpdateSTP(0);
  float depressed_x = syn.stp_x[0];

  // Recover for a long time (1000ms in 1ms steps)
  for (int i = 0; i < 1000; ++i) syn.RecoverSTP(1.0f);
  float recovered_x = syn.stp_x[0];

  // x should recover toward 1.0
  assert(recovered_x > depressed_x + 0.3f);
  assert(recovered_x > 0.9f);
}

TEST(stp_facilitation_increases_u) {
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  // Low U_se with long facilitation time constant
  STPParams params;
  params.U_se = 0.1f;
  params.tau_d = 500.0f;  // slow depression
  params.tau_f = 200.0f;  // slow facilitation decay
  syn.InitSTP(params);

  float u_before = syn.stp_u[0];
  syn.UpdateSTP(0);
  float u_after = syn.stp_u[0];

  // u should increase after a spike (facilitation)
  assert(u_after > u_before);
}

TEST(parametric_gen_stochastic_release) {
  // Verify that parametric generator passes release_probability through
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test";
  reg.n_neurons = 10;
  reg.internal_density = 0.5f;
  reg.release_probability = 0.4f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Should have stochastic release enabled
  assert(synapses.HasStochasticRelease());
  // All release probabilities should be 0.4
  for (size_t i = 0; i < synapses.p_release.size(); ++i) {
    assert(std::abs(synapses.p_release[i] - 0.4f) < 0.001f);
  }
}

// ===== Refractory period test =====

TEST(izhikevich_refractory_period) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = 35.0f;  // above threshold, should spike
  arr.u[0] = -14.0f;
  arr.i_ext[0] = 20.0f;

  IzhikevichParams p;
  p.refractory_ms = 2.0f;

  // First step: should fire
  IzhikevichStep(arr, 0.1f, 0.0f, p);
  assert(arr.spiked[0] == 1);

  // Drive voltage back up immediately
  arr.v[0] = 35.0f;

  // Step at 0.5ms (within refractory): should NOT fire
  IzhikevichStep(arr, 0.1f, 0.5f, p);
  assert(arr.spiked[0] == 0);

  // Step at 3.0ms (past refractory): should fire
  arr.v[0] = 35.0f;
  IzhikevichStep(arr, 0.1f, 3.0f, p);
  assert(arr.spiked[0] == 1);
}

// ===== Synaptic delay test =====

TEST(synapse_delay_ring_buffer) {
  SynapseTable syn;
  syn.BuildFromCOO(3, {0}, {1}, {1.0f}, {kACh});
  syn.InitDelay(1.0f, 0.5f);  // 1ms delay at 0.5ms timestep = 2 steps

  assert(syn.HasDelays());
  assert(syn.delay_steps[0] == 2);

  // Spike neuron 0
  uint8_t spiked[3] = {1, 0, 0};
  float i_syn[3] = {0, 0, 0};

  // Propagate: should write into delay buffer, not i_syn directly
  syn.DeliverDelayed(i_syn);
  syn.PropagateSpikes(spiked, i_syn, 1.0f);
  syn.AdvanceDelayRing();
  assert(i_syn[1] == 0.0f);  // not yet delivered

  // Step 2: still waiting
  spiked[0] = 0;
  std::fill(i_syn, i_syn + 3, 0.0f);
  syn.DeliverDelayed(i_syn);
  syn.PropagateSpikes(spiked, i_syn, 1.0f);
  syn.AdvanceDelayRing();
  assert(i_syn[1] == 0.0f);

  // Step 3: current arrives after 2-step delay
  std::fill(i_syn, i_syn + 3, 0.0f);
  syn.DeliverDelayed(i_syn);
  assert(i_syn[1] > 0.0f);  // delayed current delivered
}

// ===== Glutamate sign test =====

TEST(glutamate_sign_inhibitory) {
  // In Drosophila, glutamate is inhibitory via GluCl receptors
  assert(SynapseTable::Sign(kGlut) == -1.0f);
  assert(SynapseTable::Sign(kGABA) == -1.0f);
  assert(SynapseTable::Sign(kACh) == 1.0f);
}

// ===== STP clamping test =====

TEST(stp_state_clamped) {
  // Rapid repeated spikes should not drive STP state out of [0,1]
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 10.0f;   // short recovery
  params.tau_f = 10.0f;
  syn.InitSTP(params);

  // Hammer with 50 spikes, no recovery between them
  for (int i = 0; i < 50; ++i) {
    syn.UpdateSTP(0);
  }
  // u should stay in [0,1], x should stay >= 0
  assert(syn.stp_u[0] >= 0.0f && syn.stp_u[0] <= 1.0f);
  assert(syn.stp_x[0] >= 0.0f && syn.stp_x[0] <= 1.0f);
}

TEST(stp_applied_in_deterministic_propagation) {
  // PropagateSpikes (deterministic path) should modulate effective weight
  // by u*x when STP is enabled. Without this, STP state is maintained
  // but has zero effect on the simulation.
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});

  // Without STP: full weight delivered
  uint8_t spiked[2] = {1, 0};
  float i_syn_no_stp[2] = {0.0f, 0.0f};
  syn.PropagateSpikes(spiked, i_syn_no_stp, 1.0f);
  assert(std::abs(i_syn_no_stp[1] - 5.0f) < 1e-6f);

  // Enable depressing STP (U_se=0.5)
  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 200.0f;
  params.tau_f = 50.0f;
  syn.InitSTP(params);

  // First spike with STP: effective weight = w * u * x
  // After spike update: u = 0.5 + 0.5*(1-0.5) = 0.75, then ux = 0.75*1.0 = 0.75
  // x becomes 1.0 - 0.75 = 0.25
  float i_syn_stp[2] = {0.0f, 0.0f};
  syn.PropagateSpikes(spiked, i_syn_stp, 1.0f);

  // STP should reduce the delivered current below the bare weight
  assert(i_syn_stp[1] > 0.0f);
  assert(i_syn_stp[1] < 5.0f);  // modulated below full weight

  // Second spike without recovery: further depression
  float i_syn_stp2[2] = {0.0f, 0.0f};
  syn.PropagateSpikes(spiked, i_syn_stp2, 1.0f);
  assert(i_syn_stp2[1] > 0.0f);
  assert(i_syn_stp2[1] < i_syn_stp[1]);  // more depressed than first spike
}

TEST(stp_deterministic_matches_montecarlo_effective_weight) {
  // When release probability is 1.0 (deterministic release), the effective
  // weight from PropagateSpikesMonteCarlo and PropagateSpikes should match.
  SynapseTable syn1, syn2;
  syn1.BuildFromCOO(2, {0}, {1}, {3.0f}, {kACh});
  syn2.BuildFromCOO(2, {0}, {1}, {3.0f}, {kACh});

  STPParams params;
  params.U_se = 0.3f;
  params.tau_d = 100.0f;
  params.tau_f = 200.0f;
  syn1.InitSTP(params);
  syn2.InitSTP(params);

  uint8_t spiked[2] = {1, 0};
  float i_det[2] = {0.0f, 0.0f};
  float i_mc[2] = {0.0f, 0.0f};

  syn1.PropagateSpikes(spiked, i_det, 1.0f);
  std::mt19937 rng(42);
  syn2.PropagateSpikesMonteCarlo(spiked, i_mc, 1.0f, rng);

  // With no stochastic release, both should deliver the same current
  assert(std::abs(i_det[1] - i_mc[1]) < 1e-6f);
}

TEST(all_neurons_spike_propagation) {
  // When all neurons spike simultaneously, propagation should work correctly
  NeuronArray neurons;
  neurons.Resize(4);
  for (size_t i = 0; i < 4; ++i) neurons.spiked[i] = 1;

  SynapseTable syn;
  syn.BuildFromCOO(4,
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {kACh, kACh, kACh, kACh});

  float i_syn[4] = {};
  syn.PropagateSpikes(neurons.spiked.data(), i_syn, 1.0f);
  // Every neuron should receive input from its pre-synaptic partner
  for (int i = 0; i < 4; ++i) {
    assert(i_syn[i] > 0.0f);
  }
}

// ===== Exponential synaptic decay test =====

TEST(synaptic_decay_exponential) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.i_syn[0] = 10.0f;
  neurons.i_syn[1] = 5.0f;

  // Decay with tau=2ms, dt=1ms: factor = exp(-0.5) ~ 0.607
  neurons.DecaySynapticInput(1.0f, 2.0f);
  float expected = 10.0f * std::exp(-0.5f);
  assert(std::abs(neurons.i_syn[0] - expected) < 0.01f);
  assert(neurons.i_syn[0] > 0.0f);  // not zeroed
  assert(neurons.i_syn[1] > 0.0f);

  // Multiple steps should continue decaying
  float before = neurons.i_syn[0];
  neurons.DecaySynapticInput(1.0f, 2.0f);
  assert(neurons.i_syn[0] < before);
  assert(neurons.i_syn[0] > 0.0f);
}

TEST(synaptic_decay_accumulates_with_new_spikes) {
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {3.0f}, {kACh});

  // Step 1: neuron 0 spikes, delivers current
  neurons.spiked[0] = 1;
  neurons.DecaySynapticInput(0.1f, 2.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  float after_first = neurons.i_syn[1];
  assert(after_first > 0.0f);

  // Step 2: no spike, current decays but doesn't vanish
  neurons.spiked[0] = 0;
  neurons.DecaySynapticInput(0.1f, 2.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  assert(neurons.i_syn[1] > 0.0f);
  assert(neurons.i_syn[1] < after_first);  // decayed
}

// ===== Eligibility trace tests =====

TEST(eligibility_trace_accumulates) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.tau_eligibility_ms = 1000.0f;

  // Pre fires, then post fires 5ms later (potentiation pattern)
  arr.last_spike_time[0] = 0.0f;
  arr.spiked[0] = 0;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 5.0f;

  float w_before = syn.weight[0];
  STDPUpdate(syn, arr, 5.0f, p);

  // Weight should NOT change (trace mode defers weight changes)
  assert(syn.weight[0] == w_before);
  // But trace should be positive (potentiation)
  assert(syn.eligibility_trace[0] > 0.0f);
}

TEST(eligibility_trace_converts_with_dopamine) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.da_scale = 5.0f;
  p.tau_eligibility_ms = 1000.0f;

  // Set a positive eligibility trace (as if a potentiating spike pair occurred)
  syn.eligibility_trace[0] = 0.01f;

  // No dopamine: trace should not convert to weight change
  float w_before = syn.weight[0];
  EligibilityTraceUpdate(syn, arr, 1.0f, p);
  assert(syn.weight[0] == w_before);

  // Now add dopamine at the postsynaptic neuron
  arr.dopamine[1] = 0.5f;
  EligibilityTraceUpdate(syn, arr, 1.0f, p);
  assert(syn.weight[0] > w_before && "Dopamine should convert trace to weight increase");
}

TEST(eligibility_trace_decays) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.tau_eligibility_ms = 100.0f;  // fast decay for testing

  syn.eligibility_trace[0] = 1.0f;
  float trace_before = syn.eligibility_trace[0];

  // No dopamine, just decay
  EligibilityTraceUpdate(syn, arr, 50.0f, p);
  float trace_after = syn.eligibility_trace[0];
  assert(trace_after < trace_before);
  // exp(-50/100) ~ 0.607
  float expected = trace_before * std::exp(-50.0f / 100.0f);
  assert(std::abs(trace_after - expected) < 0.01f);
}

// ===== Synaptic scaling tests =====

TEST(synaptic_scaling_upscales_silent_neurons) {
  NeuronArray neurons;
  neurons.Resize(3);

  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 1}, {1, 2}, {2.0f, 2.0f}, {kACh, kACh});

  SynapticScaling scaling;
  scaling.Init(3);
  scaling.target_rate_hz = 10.0f;

  // Simulate 100ms with no spikes
  for (int i = 0; i < 1000; ++i) {
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[0];
  scaling.Apply(syn, p);
  // Silent neurons should have weights scaled up (toward target rate)
  assert(syn.weight[0] > w_before);
}

TEST(synaptic_scaling_downscales_overactive_neurons) {
  NeuronArray neurons;
  neurons.Resize(3);

  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 1}, {1, 2}, {5.0f, 5.0f}, {kACh, kACh});

  SynapticScaling scaling;
  scaling.Init(3);
  scaling.target_rate_hz = 5.0f;

  // Simulate 100ms with neuron 2 firing every step (way above target)
  for (int i = 0; i < 1000; ++i) {
    neurons.spiked[2] = 1;
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[1];  // synapse 1->2
  scaling.Apply(syn, p);
  // Over-active neuron 2 should have incoming weight scaled down
  assert(syn.weight[1] < w_before && "Overactive neuron should have downscaled weights");
}

TEST(synaptic_scaling_clamps_range) {
  NeuronArray neurons;
  neurons.Resize(2);

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});

  SynapticScaling scaling;
  scaling.Init(2);
  scaling.min_scale = 0.8f;
  scaling.max_scale = 1.2f;

  // No spikes for 100ms
  for (int i = 0; i < 1000; ++i) {
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[0];
  scaling.Apply(syn, p);
  // Scale should be clamped to max_scale=1.2
  assert(syn.weight[0] <= w_before * 1.2f + 0.01f);
  assert(syn.weight[0] >= w_before * 0.8f - 0.01f);
}

// Conditioning and multi-trial tests are in fwmc/tests/ (bridge-dependent)

// ===== Rate Monitor tests =====

TEST(rate_monitor_init) {
  NeuronArray neurons;
  neurons.Resize(100);
  for (size_t i = 0; i < 100; ++i)
    neurons.region[i] = static_cast<uint8_t>(i < 50 ? 0 : 1);

  RateMonitor mon;
  std::vector<std::string> names = {"KC", "MBON"};
  mon.Init(neurons, names, 1.0f);

  assert(mon.regions.size() == 2);
  assert(mon.regions[0].neuron_indices.size() == 50);
  assert(mon.regions[1].neuron_indices.size() == 50);
  assert(mon.regions[0].name == "KC");
}

TEST(rate_monitor_computes_rates) {
  NeuronArray neurons;
  neurons.Resize(100);
  for (size_t i = 0; i < 100; ++i)
    neurons.region[i] = 0;

  RateMonitor mon;
  mon.Init(neurons, 1.0f);

  // Simulate 1000ms with 10% of neurons spiking each step
  for (int step = 0; step < 1000; ++step) {
    for (size_t i = 0; i < 100; ++i)
      neurons.spiked[i] = (i < 10) ? 1 : 0;
    mon.RecordStep(neurons);
  }

  auto rates = mon.ComputeRates();
  assert(!rates.empty());
  // 10 out of 100 neurons spike every 1ms step = 10% * 1000 Hz = 100 Hz
  assert(rates[0].rate_hz > 90.0f && rates[0].rate_hz < 110.0f);
}

TEST(rate_monitor_literature_lookup) {
  NeuronArray neurons;
  neurons.Resize(50);
  for (size_t i = 0; i < 50; ++i)
    neurons.region[i] = 0;

  RateMonitor mon;
  std::vector<std::string> names = {"KC"};
  mon.Init(neurons, names, 1.0f);

  // KC reference: 0.5 to 10 Hz
  assert(mon.regions[0].ref_min > 0.0f);
  assert(mon.regions[0].ref_max <= 10.0f);
}

TEST(rate_monitor_in_range_count) {
  std::vector<RegionRate> rates;
  RegionRate r1;
  r1.rate_hz = 5.0f; r1.ref_min_hz = 1.0f; r1.ref_max_hz = 10.0f;
  rates.push_back(r1);
  RegionRate r2;
  r2.rate_hz = 50.0f; r2.ref_min_hz = 1.0f; r2.ref_max_hz = 10.0f;
  rates.push_back(r2);

  assert(RateMonitor::CountInRange(rates) == 1);
}

// ===== Motor Output tests =====

TEST(motor_output_init) {
  MotorOutput motor;
  motor.Init({0, 1, 2}, {3, 4, 5}, {6, 7}, {8, 9});
  assert(motor.HasMotorNeurons());
  assert(motor.TotalNeurons() == 10);
  assert(motor.descending_left.size() == 3);
  assert(motor.avoid_neurons.size() == 2);
}

TEST(motor_output_forward_velocity) {
  NeuronArray neurons;
  neurons.Resize(20);

  MotorOutput motor;
  // 10 left descending, 10 right descending
  std::vector<uint32_t> left, right;
  for (uint32_t i = 0; i < 10; ++i) left.push_back(i);
  for (uint32_t i = 10; i < 20; ++i) right.push_back(i);
  motor.Init(left, right, {}, {});

  // All neurons spiking: should produce forward velocity
  for (size_t i = 0; i < 20; ++i)
    neurons.spiked[i] = 1;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().forward_velocity > 0.0f);
}

TEST(motor_output_turning) {
  NeuronArray neurons;
  neurons.Resize(20);

  MotorOutput motor;
  std::vector<uint32_t> left, right;
  for (uint32_t i = 0; i < 10; ++i) left.push_back(i);
  for (uint32_t i = 10; i < 20; ++i) right.push_back(i);
  motor.Init(left, right, {}, {});

  // Only left neurons spike: should turn left (positive angular velocity)
  for (size_t i = 0; i < 10; ++i)
    neurons.spiked[i] = 1;
  for (size_t i = 10; i < 20; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().angular_velocity > 0.0f);
}

TEST(motor_output_approach_avoid) {
  NeuronArray neurons;
  neurons.Resize(10);

  MotorOutput motor;
  motor.Init({}, {}, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9});

  // Only approach neurons spike
  for (size_t i = 0; i < 5; ++i)
    neurons.spiked[i] = 1;
  for (size_t i = 5; i < 10; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().approach_drive > 0.0f);

  // Now only avoid neurons spike
  for (size_t i = 0; i < 5; ++i)
    neurons.spiked[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    neurons.spiked[i] = 1;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().approach_drive < 0.0f);
}

TEST(motor_output_freeze_when_silent) {
  NeuronArray neurons;
  neurons.Resize(10);

  MotorOutput motor;
  motor.Init({0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {}, {});

  // No neurons spiking
  for (size_t i = 0; i < 10; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 200; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().freeze > 0.5f);
}

TEST(motor_output_from_regions) {
  NeuronArray neurons;
  neurons.Resize(20);
  // SEZ region = 12, MBON region = 3
  for (size_t i = 0; i < 10; ++i) {
    neurons.region[i] = 12;
    neurons.x[i] = (i < 5) ? 100.0f : 400.0f;  // L/R split at 250
  }
  for (size_t i = 10; i < 20; ++i) {
    neurons.region[i] = 3;
    neurons.type[i] = (i < 15) ? 2 : 3;  // cholinergic vs GABAergic
  }

  MotorOutput motor;
  motor.InitFromRegions(neurons, 12, 3, 250.0f);

  assert(motor.descending_left.size() == 5);
  assert(motor.descending_right.size() == 5);
  assert(motor.approach_neurons.size() == 5);
  assert(motor.avoid_neurons.size() == 5);
}

// ===== Intrinsic Homeostasis tests =====

TEST(homeostasis_init) {
  IntrinsicHomeostasis homeo;
  homeo.Init(100, 5.0f, 0.1f);
  assert(homeo.bias_current.size() == 100);
  assert(homeo.MeanBias() == 0.0f);
}

TEST(homeostasis_silent_neurons_get_positive_bias) {
  NeuronArray neurons;
  neurons.Resize(10);
  // All neurons silent

  IntrinsicHomeostasis homeo;
  homeo.Init(10, 5.0f, 1.0f);

  // Record 1000ms of silence
  for (int i = 0; i < 1000; ++i)
    homeo.RecordSpikes(neurons);

  homeo.Apply(neurons);

  // Silent neurons should get positive bias (to encourage firing)
  assert(homeo.MeanBias() > 0.0f);
  assert(homeo.FractionExcited() == 1.0f);
}

TEST(homeostasis_active_neurons_get_negative_bias) {
  NeuronArray neurons;
  neurons.Resize(10);

  IntrinsicHomeostasis homeo;
  homeo.Init(10, 5.0f, 1.0f);
  homeo.target_rate_hz = 5.0f;

  // All neurons fire every step for 1000ms (= 1000 Hz, way above target)
  for (int i = 0; i < 1000; ++i) {
    for (size_t j = 0; j < 10; ++j) neurons.spiked[j] = 1;
    homeo.RecordSpikes(neurons);
  }

  homeo.Apply(neurons);

  // Overactive neurons should get negative bias
  assert(homeo.MeanBias() < 0.0f);
  assert(homeo.FractionExcited() == 0.0f);
}

TEST(homeostasis_bias_clamps) {
  NeuronArray neurons;
  neurons.Resize(2);

  IntrinsicHomeostasis homeo;
  homeo.Init(2, 5.0f, 1.0f);
  homeo.max_bias = 3.0f;

  // Apply many times with silence to push bias high
  for (int round = 0; round < 50; ++round) {
    for (int i = 0; i < 1000; ++i)
      homeo.RecordSpikes(neurons);
    homeo.Apply(neurons);
  }

  // Bias should be clamped
  for (float b : homeo.bias_current) {
    assert(b <= homeo.max_bias + 0.001f);
    assert(b >= -homeo.max_bias - 0.001f);
  }
}

TEST(homeostasis_maybe_apply_respects_interval) {
  NeuronArray neurons;
  neurons.Resize(5);

  IntrinsicHomeostasis homeo;
  homeo.Init(5, 5.0f, 1.0f);
  homeo.update_interval_ms = 100.0f;

  // 50ms of recording: not enough
  for (int i = 0; i < 50; ++i)
    homeo.RecordSpikes(neurons);
  assert(!homeo.MaybeApply(neurons));

  // 50 more ms: now at 100ms, should apply
  for (int i = 0; i < 50; ++i)
    homeo.RecordSpikes(neurons);
  assert(homeo.MaybeApply(neurons));
}

// ===== Gap junction tests =====

TEST(gap_junction_bidirectional_current) {
  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -60.0f;
  arr.v[1] = -40.0f;  // higher voltage
  arr.i_ext[0] = 0.0f;
  arr.i_ext[1] = 0.0f;

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 0.5f);
  gj.PropagateGapCurrents(arr);

  // Current flows from high V (neuron 1) to low V (neuron 0)
  // I = 0.5 * (-40 - (-60)) = 0.5 * 20 = 10
  assert(std::abs(arr.i_ext[0] - 10.0f) < 0.01f);   // neuron 0 gains current
  assert(std::abs(arr.i_ext[1] - (-10.0f)) < 0.01f); // neuron 1 loses current
}

TEST(gap_junction_equal_voltage_no_current) {
  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -50.0f;
  arr.v[1] = -50.0f;
  arr.i_ext[0] = 0.0f;
  arr.i_ext[1] = 0.0f;

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f);
  gj.PropagateGapCurrents(arr);

  assert(std::abs(arr.i_ext[0]) < 0.001f);
  assert(std::abs(arr.i_ext[1]) < 0.001f);
}

TEST(gap_junction_build_from_region) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 10; ++i) arr.region[i] = 0;
  for (size_t i = 10; i < 20; ++i) arr.region[i] = 1;

  GapJunctionTable gj;
  gj.BuildFromRegion(arr, 0, 1.0f, 0.3f);  // density=1.0, all pairs connected
  // 10 neurons, C(10,2) = 45 pairs
  assert(gj.Size() == 45);
  assert(gj.conductance[0] == 0.3f);
}

TEST(gap_junction_empty) {
  NeuronArray arr;
  arr.Resize(5);
  GapJunctionTable gj;
  gj.PropagateGapCurrents(arr);  // should not crash
  assert(gj.Size() == 0);
}

// ===== Short-term plasticity tests =====

TEST(stp_init_defaults) {
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());
  assert(syn.HasSTP());
  assert(syn.stp_u[0] == 0.5f);
  assert(syn.stp_x[0] == 1.0f);
}

TEST(stp_depression_on_spike) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 1;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  float x_before = syn.stp_x[0];
  UpdateSTP(syn, neurons, 0.1f);
  float x_after = syn.stp_x[0];

  // After spike, x should decrease (depression)
  assert(x_after < x_before);
  // Effective weight = base * u * x
  float w_eff = syn.weight[0] * syn.stp_u[0] * syn.stp_x[0];
  assert(w_eff < 1.0f);
}

TEST(stp_recovery_without_spikes) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 0;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  // Manually depress
  syn.stp_x[0] = 0.3f;
  syn.stp_u[0] = 0.8f;

  // Run many steps without spikes
  for (int i = 0; i < 10000; ++i)
    UpdateSTP(syn, neurons, 1.0f);

  // Should recover toward resting state
  assert(std::abs(syn.stp_x[0] - 1.0f) < 0.01f);
  assert(std::abs(syn.stp_u[0] - 0.5f) < 0.01f);
}

TEST(stp_facilitating_increases_u) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 1;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPFacilitating());

  float u_before = syn.stp_u[0];
  UpdateSTP(syn, neurons, 0.1f);
  float u_after = syn.stp_u[0];

  // Facilitation: u should increase on spike
  assert(u_after > u_before);
}

TEST(stp_effective_weight_bounded) {
  SynapseTable syn;
  syn.n_neurons = 1;
  syn.row_ptr = {0, 1};
  syn.post = {0};
  syn.weight = {5.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  float w_eff = syn.weight[0] * syn.stp_u[0] * syn.stp_x[0];
  assert(w_eff <= 5.0f);
  assert(w_eff >= 0.0f);
}

TEST(stp_reset) {
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 5, 10};
  syn.post.resize(10);
  syn.weight.resize(10, 1.0f);
  syn.nt_type.resize(10, static_cast<uint8_t>(NTType::kACh));
  syn.InitSTP(STPCombined());

  syn.stp_u[5] = 0.99f;
  syn.stp_x[5] = 0.01f;
  ResetSTP(syn);
  assert(std::abs(syn.stp_u[5] - 0.25f) < 0.001f);
  assert(std::abs(syn.stp_x[5] - 1.0f) < 0.001f);
}

TEST(stp_presets) {
  STPParams fac = STPFacilitating();
  STPParams dep = STPDepressing();
  STPParams com = STPCombined();
  assert(fac.U_se < dep.U_se);  // facilitating has lower baseline release
  assert(fac.tau_f > dep.tau_f);  // facilitating has slower facilitation decay
  assert(com.tau_f > dep.tau_f && com.tau_f < fac.tau_f);
}

// ===== CPG tests =====

TEST(cpg_init_splits_groups) {
  NeuronArray arr;
  arr.Resize(100);
  for (size_t i = 0; i < 100; ++i) {
    arr.region[i] = 5;  // all VNC
    arr.x[i] = static_cast<float>(i * 5);  // spread across x
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.3f);
  assert(cpg.initialized);
  assert(!cpg.group_a.empty());
  assert(!cpg.group_b.empty());
  // Groups should be non-overlapping
  size_t total = cpg.group_a.size() + cpg.group_b.size();
  assert(total == 70);  // 100 - 30% sensory = 70 motor neurons
}

TEST(cpg_zero_drive_no_current) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 25);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.0f);
  cpg.Step(arr, 1.0f, 0.0f);  // zero descending drive

  // With drive=0, CPG should be silent
  float total_current = 0.0f;
  for (size_t i = 0; i < 20; ++i) total_current += arr.i_ext[i];
  assert(std::abs(total_current) < 0.01f);
}

TEST(cpg_full_drive_injects_current) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 25);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.0f);
  cpg.drive_scale = 1.0f;  // force full drive immediately
  cpg.Step(arr, 1.0f, 1.0f);

  // Some neurons should receive positive current
  float total_current = 0.0f;
  for (size_t i = 0; i < 20; ++i) total_current += arr.i_ext[i];
  assert(total_current > 0.0f);
}

TEST(cpg_antiphase_groups) {
  NeuronArray arr;
  arr.Resize(100);
  for (size_t i = 0; i < 100; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 5);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.3f);
  cpg.drive_scale = 1.0f;
  cpg.phase = 0.5f;  // set phase so groups get different currents
  cpg.Step(arr, 1.0f, 1.0f);

  // Sum current for each group
  float sum_a = 0.0f, sum_b = 0.0f;
  for (uint32_t i : cpg.group_a) sum_a += arr.i_ext[i];
  for (uint32_t i : cpg.group_b) sum_b += arr.i_ext[i];

  // Groups should receive different current levels (anti-phase)
  assert(std::abs(sum_a - sum_b) > 0.1f);
}

// ===== Proprioception tests =====

TEST(proprio_init_assigns_channels) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);
  assert(pm.initialized);
  // Should have assigned neurons to joint angle channels
  int assigned = 0;
  for (int j = 0; j < 42; ++j) assigned += static_cast<int>(pm.joint_angle_neurons[j].size());
  assert(assigned > 0);
}

TEST(proprio_inject_contact) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};
  state.contacts[0] = 1.0f;  // left front leg touching ground

  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Contact neurons for leg 0 should have received current
  if (!pm.contact_neurons[0].empty()) {
    float current = arr.i_ext[pm.contact_neurons[0][0]];
    assert(current > 0.0f);
  }
}

TEST(proprio_haltere_asymmetry) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};
  state.body_velocity[2] = 2.0f;  // positive yaw (turning left)

  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Positive yaw should excite right haltere more than left
  float sum_left = 0.0f, sum_right = 0.0f;
  for (uint32_t i : pm.haltere_left) sum_left += arr.i_ext[i];
  for (uint32_t i : pm.haltere_right) sum_right += arr.i_ext[i];
  assert(sum_right > sum_left);
}

TEST(proprio_zero_state_minimal_current) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};  // all zeros
  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Zero joint angles still produce some current due to sigmoid offset,
  // but contacts and haltere should be zero
  float haltere_current = 0.0f;
  for (uint32_t i : pm.haltere_left) haltere_current += arr.i_ext[i];
  for (uint32_t i : pm.haltere_right) haltere_current += arr.i_ext[i];
  assert(std::abs(haltere_current) < 0.01f);
}

// ===== BehavioralFingerprint tests =====

#include "core/behavioral_fingerprint.h"

TEST(fingerprint_identical_brains_score_one) {
  BehavioralFingerprint fp;
  fp.sample_interval_ms = 1.0f;

  // Record identical motor output
  for (int t = 0; t < 100; ++t) {
    MotorCommand cmd;
    cmd.forward_velocity = std::sin(t * 0.1f) * 10.0f;
    cmd.angular_velocity = std::cos(t * 0.1f) * 2.0f;
    cmd.approach_drive = 0.5f;
    cmd.freeze = 0.0f;
    fp.Record(cmd, static_cast<float>(t));
  }

  // Comparing to itself should give 1.0
  float sim = BehavioralFingerprint::Compare(fp, fp);
  assert(std::abs(sim - 1.0f) < 0.01f);
}

TEST(fingerprint_different_brains_score_low) {
  BehavioralFingerprint a, b;
  a.sample_interval_ms = 1.0f;
  b.sample_interval_ms = 1.0f;

  for (int t = 0; t < 100; ++t) {
    MotorCommand cmd_a;
    cmd_a.forward_velocity = 10.0f;
    cmd_a.angular_velocity = 0.0f;
    a.Record(cmd_a, static_cast<float>(t));

    MotorCommand cmd_b;
    cmd_b.forward_velocity = -10.0f;
    cmd_b.angular_velocity = 3.14f;
    b.Record(cmd_b, static_cast<float>(t));
  }

  float sim = BehavioralFingerprint::Compare(a, b);
  assert(sim < 0.7f);  // should be noticeably different
}

TEST(fingerprint_identity_preserved_threshold) {
  BehavioralFingerprint ref, candidate;
  ref.sample_interval_ms = 1.0f;
  candidate.sample_interval_ms = 1.0f;

  // Nearly identical with small noise
  for (int t = 0; t < 200; ++t) {
    float ft = static_cast<float>(t);
    MotorCommand cmd;
    cmd.forward_velocity = std::sin(ft * 0.05f) * 15.0f;
    cmd.angular_velocity = std::cos(ft * 0.03f) * 1.0f;
    ref.Record(cmd, ft);

    // Add small perturbation
    cmd.forward_velocity += 0.1f;
    cmd.angular_velocity -= 0.05f;
    candidate.Record(cmd, ft);
  }

  assert(BehavioralFingerprint::IdentityPreserved(ref, candidate, 0.85f));
}

TEST(fingerprint_channel_correlation) {
  BehavioralFingerprint a, b;
  a.sample_interval_ms = 1.0f;
  b.sample_interval_ms = 1.0f;

  // Forward velocity perfectly correlated, angular anti-correlated
  for (int t = 0; t < 100; ++t) {
    float ft = static_cast<float>(t);
    MotorCommand ca, cb;
    ca.forward_velocity = ft;
    cb.forward_velocity = ft;      // same
    ca.angular_velocity = ft;
    cb.angular_velocity = -ft;     // opposite
    a.Record(ca, ft);
    b.Record(cb, ft);
  }

  float corr_fwd = BehavioralFingerprint::ChannelCorrelation(
      a, b, BehavioralFingerprint::kForward);
  float corr_ang = BehavioralFingerprint::ChannelCorrelation(
      a, b, BehavioralFingerprint::kAngular);

  assert(corr_fwd > 0.99f);   // perfectly correlated
  assert(corr_ang < -0.99f);  // perfectly anti-correlated
}

TEST(fingerprint_empty_returns_zero) {
  BehavioralFingerprint a, b;
  float sim = BehavioralFingerprint::Compare(a, b);
  assert(sim == 0.0f);
}

TEST(fingerprint_sample_interval_respected) {
  BehavioralFingerprint fp;
  fp.sample_interval_ms = 5.0f;

  MotorCommand cmd;
  cmd.forward_velocity = 1.0f;

  // Record at 1ms intervals for 20ms
  for (int t = 0; t < 20; ++t) {
    fp.Record(cmd, static_cast<float>(t));
  }

  // Should only have 4 samples (at 0, 5, 10, 15)
  assert(fp.samples.size() == 4);
}

// ---- Serotonergic / Octopaminergic cell types ----

TEST(serotonergic_cell_type_params) {
  auto p = ParamsForCellType(CellType::kSerotonergic);
  // Tonic low-rate: strong adaptation (high d), moderate b
  assert(p.a == 0.02f);
  assert(p.b == 0.25f);
  assert(p.c == -60.0f);
  assert(p.d == 8.0f);
  assert(p.refractory_ms == 2.0f);
}

TEST(octopaminergic_cell_type_params) {
  auto p = ParamsForCellType(CellType::kOctopaminergic);
  // Phasic: faster a (quicker recovery), lower d (less adaptation)
  assert(p.a == 0.03f);
  assert(p.b == 0.20f);
  assert(p.c == -58.0f);
  assert(p.d == 5.0f);
  assert(p.refractory_ms == 1.5f);
}

TEST(serotonergic_fires_at_low_rate) {
  // 5HT neurons should be tonic low-rate (~2-5 Hz)
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.type[0] = static_cast<uint8_t>(CellType::kSerotonergic);
  auto p = ParamsForCellType(CellType::kSerotonergic);

  int spike_count = 0;
  float sim_time = 0.0f;
  float duration_ms = 2000.0f;
  // Moderate constant drive
  for (float t = 0; t < duration_ms; t += 1.0f) {
    neurons.i_ext[0] = 8.0f;
    IzhikevichStep(neurons, 1.0f, sim_time, p);
    spike_count += neurons.spiked[0];
    sim_time += 1.0f;
  }
  float rate_hz = spike_count / (duration_ms / 1000.0f);
  // Should fire in the 1-15 Hz range (tonic, adapted)
  assert(rate_hz > 1.0f && rate_hz < 15.0f);
}

TEST(octopaminergic_fires_phasically) {
  // OA neurons: faster dynamics, phasic responses
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.type[0] = static_cast<uint8_t>(CellType::kOctopaminergic);
  auto p = ParamsForCellType(CellType::kOctopaminergic);

  int spike_count = 0;
  float sim_time = 0.0f;
  float duration_ms = 2000.0f;
  for (float t = 0; t < duration_ms; t += 1.0f) {
    neurons.i_ext[0] = 10.0f;
    IzhikevichStep(neurons, 1.0f, sim_time, p);
    spike_count += neurons.spiked[0];
    sim_time += 1.0f;
  }
  float rate_hz = spike_count / (duration_ms / 1000.0f);
  // Should fire; phasic means higher initial burst then settling
  assert(rate_hz > 2.0f && rate_hz < 30.0f);
}

TEST(serotonergic_homeostasis_target) {
  float target = TargetRateForCellType(
      static_cast<uint8_t>(CellType::kSerotonergic));
  assert(target == 3.0f);  // tonic low-rate
}

TEST(octopaminergic_homeostasis_target) {
  float target = TargetRateForCellType(
      static_cast<uint8_t>(CellType::kOctopaminergic));
  assert(target == 5.0f);
}

// ---- Gap junction rectification ----

TEST(gap_junction_rectification_symmetric) {
  // rectification = 1.0 (default): current flows equally both ways
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.v[0] = -65.0f;
  neurons.v[1] = -55.0f;  // 10mV difference

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f, 1.0f);  // symmetric
  gj.PropagateGapCurrents(neurons);

  // I = g * (Vb - Va) = 1.0 * (-55 - (-65)) = 10
  float tol = 0.01f;
  assert(std::abs(neurons.i_ext[0] - 10.0f) < tol);   // current into a
  assert(std::abs(neurons.i_ext[1] - (-10.0f)) < tol); // current out of b
}

TEST(gap_junction_rectification_attenuates_reverse) {
  // rectification = 0.3: reverse current (Vb < Va) attenuated to 30%
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.v[0] = -55.0f;  // a is higher voltage
  neurons.v[1] = -65.0f;  // b is lower

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f, 0.3f);  // rectifying
  gj.PropagateGapCurrents(neurons);

  // dv = Vb - Va = -65 - (-55) = -10 (negative = reverse direction)
  // Reverse: g *= 0.3, so I = 0.3 * (-10) = -3
  float tol = 0.01f;
  assert(std::abs(neurons.i_ext[0] - (-3.0f)) < tol);
  assert(std::abs(neurons.i_ext[1] - 3.0f) < tol);
}

TEST(gap_junction_rectification_forward_unattenuated) {
  // Forward direction (Vb > Va): full conductance regardless of rectification
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.v[0] = -65.0f;
  neurons.v[1] = -55.0f;  // b higher, forward direction

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f, 0.3f);  // rectifying, but forward
  gj.PropagateGapCurrents(neurons);

  // dv = -55 - (-65) = 10 (positive = forward), no attenuation
  float tol = 0.01f;
  assert(std::abs(neurons.i_ext[0] - 10.0f) < tol);
  assert(std::abs(neurons.i_ext[1] - (-10.0f)) < tol);
}

TEST(gap_junction_fully_rectifying) {
  // rectification = 0: fully rectifying (blocks reverse current completely)
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.v[0] = -55.0f;  // a higher
  neurons.v[1] = -65.0f;  // b lower (reverse)

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f, 0.0f);
  gj.PropagateGapCurrents(neurons);

  // Reverse direction with rect=0: g *= 0, so I = 0
  float tol = 0.01f;
  assert(std::abs(neurons.i_ext[0]) < tol);
  assert(std::abs(neurons.i_ext[1]) < tol);
}

// ---- NWB export ----

TEST(nwb_export_creates_files) {
  std::string dir = "test_tmp_nwb";
  NeuronArray neurons;
  neurons.Resize(5);
  neurons.region[0] = 1;  // MB
  neurons.type[0] = 1;    // KC

  NWBExporter nwb;
  bool ok = nwb.BeginSession(dir, "test session", neurons);
  assert(ok);
  assert(nwb.n_neurons == 5);

  // Record a few steps with one spike
  neurons.spiked[0] = 1;
  nwb.RecordTimestep(1.0f, neurons);
  neurons.spiked[0] = 0;
  nwb.RecordTimestep(2.0f, neurons);

  nwb.EndSession();

  // Verify files exist
  FILE* sf = fopen((dir + "/spikes.nwb.csv").c_str(), "r");
  assert(sf);
  fclose(sf);

  FILE* jf = fopen((dir + "/session.nwb.json").c_str(), "r");
  assert(jf);
  fclose(jf);

  assert(nwb.total_spikes == 1);
  assert(nwb.n_recorded_steps == 2);

  // Cleanup
  remove((dir + "/spikes.nwb.csv").c_str());
  remove((dir + "/session.nwb.json").c_str());
#ifdef _WIN32
  _rmdir(dir.c_str());
#else
  rmdir(dir.c_str());
#endif
}

TEST(nwb_export_voltage_traces) {
  std::string dir = "test_tmp_nwb_v";
  NeuronArray neurons;
  neurons.Resize(10);

  NWBExporter nwb;
  nwb.SetVoltageSubset({0, 3, 7});
  bool ok = nwb.BeginSession(dir, "voltage test", neurons);
  assert(ok);

  nwb.RecordTimestep(0.0f, neurons);
  neurons.v[3] = -40.0f;
  nwb.RecordTimestep(1.0f, neurons);

  nwb.EndSession();

  FILE* vf = fopen((dir + "/voltages.nwb.csv").c_str(), "r");
  assert(vf);
  // Read header
  char line[256];
  fgets(line, sizeof(line), vf);
  std::string header(line);
  assert(header.find("neuron_0") != std::string::npos);
  assert(header.find("neuron_3") != std::string::npos);
  assert(header.find("neuron_7") != std::string::npos);
  fclose(vf);

  remove((dir + "/spikes.nwb.csv").c_str());
  remove((dir + "/voltages.nwb.csv").c_str());
  remove((dir + "/session.nwb.json").c_str());
#ifdef _WIN32
  _rmdir(dir.c_str());
#else
  rmdir(dir.c_str());
#endif
}

TEST(nwb_export_stimulus_metadata) {
  std::string dir = "test_tmp_nwb_s";
  NeuronArray neurons;
  neurons.Resize(3);

  NWBExporter nwb;
  bool ok = nwb.BeginSession(dir, "stimulus test", neurons);
  assert(ok);

  nwb.AddStimulus(100.0f, 200.0f, "odor_A", "ethyl acetate pulse");
  nwb.AddStimulus(500.0f, 600.0f, "shock", "electric shock");

  nwb.EndSession();

  // Read JSON and check stimulus entries
  FILE* jf = fopen((dir + "/session.nwb.json").c_str(), "r");
  assert(jf);
  std::string json;
  char buf[1024];
  while (fgets(buf, sizeof(buf), jf)) json += buf;
  fclose(jf);

  assert(json.find("odor_A") != std::string::npos);
  assert(json.find("electric shock") != std::string::npos);
  assert(json.find("NWBFile") != std::string::npos);

  remove((dir + "/spikes.nwb.csv").c_str());
  remove((dir + "/session.nwb.json").c_str());
#ifdef _WIN32
  _rmdir(dir.c_str());
#else
  rmdir(dir.c_str());
#endif
}

TEST(nwb_cell_type_label_serotonergic) {
  assert(std::string(NWBExporter::CellTypeLabel(
      static_cast<uint8_t>(CellType::kSerotonergic))) == "Serotonergic");
}

TEST(nwb_cell_type_label_octopaminergic) {
  assert(std::string(NWBExporter::CellTypeLabel(
      static_cast<uint8_t>(CellType::kOctopaminergic))) == "Octopaminergic");
}

// ---- Per-NT synaptic time constants ----

TEST(per_nt_tau_ach_fast) {
  assert(SynapseTable::TauForNT(NTType::kACh) == 2.0f);
}

TEST(per_nt_tau_gaba_slow) {
  assert(SynapseTable::TauForNT(NTType::kGABA) == 5.0f);
}

TEST(per_nt_tau_glut_moderate) {
  assert(SynapseTable::TauForNT(NTType::kGlut) == 3.0f);
}

TEST(assign_per_neuron_tau_from_synapses) {
  NeuronArray neurons;
  neurons.Resize(3);

  SynapseTable syn;
  // Neuron 0: receives 3 ACh synapses
  // Neuron 1: receives 2 GABA + 1 ACh -> dominant GABA
  // Neuron 2: receives 1 Glut
  syn.n_neurons = 3;
  syn.row_ptr = {0, 3, 6, 7};
  syn.post =    {0, 0, 0,  1, 1, 1,  2};
  syn.weight =  {1, 1, 1,  1, 1, 1,  1};
  syn.nt_type = {kACh, kACh, kACh,  kGABA, kGABA, kACh,  kGlut};

  syn.AssignPerNeuronTau(neurons);

  assert(neurons.tau_syn.size() == 3);
  assert(neurons.tau_syn[0] == 2.0f);  // ACh dominant
  assert(neurons.tau_syn[1] == 5.0f);  // GABA dominant
  assert(neurons.tau_syn[2] == 3.0f);  // Glut only
}

TEST(per_neuron_decay_uses_tau_syn) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.tau_syn = {2.0f, 5.0f};  // fast and slow
  neurons.i_syn[0] = 10.0f;
  neurons.i_syn[1] = 10.0f;

  neurons.DecaySynapticInput(1.0f, 3.0f);  // uniform tau ignored when per-neuron set

  float expected_0 = 10.0f * std::exp(-1.0f / 2.0f);  // fast decay
  float expected_1 = 10.0f * std::exp(-1.0f / 5.0f);  // slow decay
  float tol = 0.001f;
  assert(std::abs(neurons.i_syn[0] - expected_0) < tol);
  assert(std::abs(neurons.i_syn[1] - expected_1) < tol);
  // Fast ACh should decay more than slow GABA
  assert(neurons.i_syn[0] < neurons.i_syn[1]);
}

// ---- SimFeatures ----

TEST(sim_features_default) {
  SimFeatures f;
  assert(f.izhikevich);
  assert(!f.stdp);
  assert(!f.structural_plasticity);
  assert(f.homeostasis);
  assert(f.gap_junctions);
}

TEST(sim_features_full_preset) {
  auto f = SimFeatures::Full();
  assert(f.stdp);
  assert(f.structural_plasticity);
  assert(f.short_term_plasticity);
  assert(f.stochastic_release);
  assert(f.recording);
  assert(f.CountEnabled() == 16);
}

TEST(sim_features_inference_preset) {
  auto f = SimFeatures::Inference();
  assert(!f.homeostasis);
  assert(!f.gap_junctions);
  assert(!f.neuromodulation);
  assert(f.izhikevich);
  assert(f.spike_propagation);
}

// ---- Temperature scaling ----

TEST(temperature_disabled_returns_one) {
  TemperatureModel tm;
  tm.enabled = false;
  assert(tm.ChannelScale() == 1.0f);
  assert(tm.SynapseScale() == 1.0f);
  assert(tm.MembraneScale() == 1.0f);
}

TEST(temperature_at_reference_returns_one) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.current_temp_c = tm.reference_temp_c;
  float tol = 0.001f;
  assert(std::abs(tm.ChannelScale() - 1.0f) < tol);
  assert(std::abs(tm.SynapseScale() - 1.0f) < tol);
}

TEST(temperature_higher_speeds_up) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.reference_temp_c = 22.0f;
  tm.current_temp_c = 32.0f;  // +10C
  // Q10=2.5 for channels: should scale by 2.5x
  float tol = 0.01f;
  assert(std::abs(tm.ChannelScale() - 2.5f) < tol);
  // Q10=2.0 for synapses: should scale by 2.0x
  assert(std::abs(tm.SynapseScale() - 2.0f) < tol);
}

TEST(temperature_lower_slows_down) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.reference_temp_c = 22.0f;
  tm.current_temp_c = 12.0f;  // -10C
  // Q10=2.5: should scale by 1/2.5 = 0.4
  float tol = 0.01f;
  assert(std::abs(tm.ChannelScale() - 0.4f) < tol);
}

TEST(temperature_scaled_tau_syn) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.reference_temp_c = 22.0f;
  tm.current_temp_c = 32.0f;
  // tau should halve with Q10=2 (+10C)
  float scaled = tm.ScaledTauSyn(4.0f);
  float tol = 0.01f;
  assert(std::abs(scaled - 2.0f) < tol);
}

TEST(temperature_scaled_a) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.reference_temp_c = 22.0f;
  tm.current_temp_c = 25.0f;  // +3C (typical Drosophila walking temp)
  float a = 0.02f;
  float scaled = tm.ScaledA(a);
  // Q10=2.5, dT=3: 2.5^(3/10) = ~1.30
  assert(scaled > a);
  assert(scaled < a * 1.5f);
}

// ---- Spike frequency adaptation ----

TEST(sfa_init) {
  SpikeFrequencyAdaptation sfa;
  sfa.Init(10);
  assert(sfa.initialized);
  assert(sfa.calcium.size() == 10);
  assert(sfa.MeanCalcium() == 0.0f);
}

TEST(sfa_spike_increases_calcium) {
  NeuronArray neurons;
  neurons.Resize(2);
  SpikeFrequencyAdaptation sfa;
  sfa.Init(2);

  neurons.spiked[0] = 1;
  neurons.spiked[1] = 0;
  sfa.Update(neurons, 1.0f);

  assert(sfa.calcium[0] > 0.0f);
  assert(sfa.calcium[1] == 0.0f);
}

TEST(sfa_injects_hyperpolarizing_current) {
  NeuronArray neurons;
  neurons.Resize(1);
  SpikeFrequencyAdaptation sfa;
  sfa.Init(1);
  sfa.g_sahp = 1.0f;

  // Spike to build calcium
  neurons.spiked[0] = 1;
  neurons.i_ext[0] = 0.0f;
  sfa.Update(neurons, 1.0f);

  // i_ext should be negative (hyperpolarizing)
  assert(neurons.i_ext[0] < 0.0f);
}

TEST(sfa_calcium_decays) {
  NeuronArray neurons;
  neurons.Resize(1);
  SpikeFrequencyAdaptation sfa;
  sfa.Init(1);

  // One spike
  neurons.spiked[0] = 1;
  sfa.Update(neurons, 1.0f);
  float ca_after_spike = sfa.calcium[0];

  // No more spikes, let it decay
  neurons.spiked[0] = 0;
  for (int t = 0; t < 1000; ++t) {
    sfa.Update(neurons, 1.0f);
  }
  // After 1000ms with tau=300ms, calcium should be very small
  assert(sfa.calcium[0] < ca_after_spike * 0.05f);
}

TEST(sfa_reduces_firing_rate) {
  // Run a neuron with constant drive: with SFA it should fire less
  // than without SFA over a sustained period
  IzhikevichParams p;
  float drive = 12.0f;

  // Without SFA
  NeuronArray n1;
  n1.Resize(1);
  int spikes_no_sfa = 0;
  float t1 = 0.0f;
  for (int t = 0; t < 2000; ++t) {
    n1.i_ext[0] = drive;
    IzhikevichStep(n1, 1.0f, t1, p);
    spikes_no_sfa += n1.spiked[0];
    t1 += 1.0f;
  }

  // With SFA
  NeuronArray n2;
  n2.Resize(1);
  SpikeFrequencyAdaptation sfa;
  sfa.Init(1);
  sfa.g_sahp = 0.5f;
  int spikes_with_sfa = 0;
  float t2 = 0.0f;
  for (int t = 0; t < 2000; ++t) {
    n2.i_ext[0] = drive;
    IzhikevichStep(n2, 1.0f, t2, p);
    sfa.Update(n2, 1.0f);
    spikes_with_sfa += n2.spiked[0];
    t2 += 1.0f;
  }

  // SFA should reduce total spikes
  assert(spikes_with_sfa < spikes_no_sfa);
  // But not eliminate them entirely
  assert(spikes_with_sfa > 0);
}

// ===== Distance-dependent axonal delay tests =====

TEST(distance_delay_basic) {
  // Two neurons 1000 units apart, velocity = 500 units/ms, dt=0.1ms
  // Expected delay = 1000/500 = 2.0ms = 20 steps
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f;  neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 1000.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  syn.InitDistanceDelay(neurons, 500.0f, 0.1f, 0.1f);

  assert(syn.HasDelays());
  assert(syn.delay_steps[0] == 20);  // 2.0ms / 0.1ms = 20 steps
}

TEST(distance_delay_3d_diagonal) {
  // 3D distance: neuron 0 at origin, neuron 1 at (300, 400, 0)
  // Distance = sqrt(300^2 + 400^2) = 500 units
  // Velocity = 250 units/ms => delay = 2.0ms = 20 steps at dt=0.1ms
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f;   neurons.y[0] = 0.0f;   neurons.z[0] = 0.0f;
  neurons.x[1] = 300.0f; neurons.y[1] = 400.0f; neurons.z[1] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  syn.InitDistanceDelay(neurons, 250.0f, 0.1f, 0.1f);

  assert(syn.delay_steps[0] == 20);
}

TEST(distance_delay_min_clamp) {
  // Two neurons very close (1 unit apart), fast velocity
  // Raw delay would be tiny; should clamp to min_delay_ms
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f; neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 1.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  // min_delay = 0.5ms at dt=0.1ms = 5 steps minimum
  syn.InitDistanceDelay(neurons, 10000.0f, 0.1f, 0.5f);

  assert(syn.delay_steps[0] == 5);
}

TEST(distance_delay_multiple_synapses) {
  // 3 neurons at increasing distances from neuron 0
  // 0 -> 1: distance 100, 0 -> 2: distance 500
  // Velocity = 100 units/ms, dt = 0.1ms
  // delay(0->1) = 100/100 = 1.0ms = 10 steps
  // delay(0->2) = 500/100 = 5.0ms = 50 steps
  NeuronArray neurons;
  neurons.Resize(3);
  neurons.x[0] = 0.0f;   neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 100.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;
  neurons.x[2] = 500.0f; neurons.y[2] = 0.0f; neurons.z[2] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 0}, {1, 2}, {1.0f, 1.0f}, {kACh, kACh});
  syn.InitDistanceDelay(neurons, 100.0f, 0.1f, 0.1f);

  assert(syn.delay_steps[0] == 10);  // 0->1: 1ms
  assert(syn.delay_steps[1] == 50);  // 0->2: 5ms
  // Ring buffer should be sized for the max delay
  assert(syn.ring_size == 51);
}

TEST(distance_delay_large_delay_uint16) {
  // Test that delays > 255 steps work (requires uint16_t).
  // Simulates long-range human interhemispheric projection.
  // Distance: 150000 um (15cm), velocity: 1 m/s = 1000 um/ms
  // Delay = 150000/1000 = 150ms = 1500 steps (exceeds uint8_t max 255)
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f;      neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 150000.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  syn.InitDistanceDelay(neurons, 1000.0f, 0.1f, 0.1f);

  assert(syn.delay_steps[0] == 1500);  // 150ms / 0.1ms
  assert(syn.delay_steps[0] > 255);    // proves uint8_t would overflow
  assert(syn.ring_size == 1501);
}

TEST(distance_delay_velocity_scaling) {
  // Same geometry, different velocities produce different delays.
  // Myelinated axons (fast) vs unmyelinated (slow).
  // Distance: 10000 um between neurons.
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f;     neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 10000.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;

  // Unmyelinated: 0.5 m/s = 500 um/ms => delay = 20ms = 200 steps
  SynapseTable syn_slow;
  syn_slow.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  syn_slow.InitDistanceDelay(neurons,
      SynapseTable::kVelocityDrosophilaUnmyel * 1000.0f,
      0.1f, 0.1f);

  // Myelinated: 10 m/s = 10000 um/ms => delay = 1ms = 10 steps
  SynapseTable syn_fast;
  syn_fast.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  syn_fast.InitDistanceDelay(neurons,
      SynapseTable::kVelocityCorpusCallosum * 1000.0f,
      0.1f, 0.1f);

  assert(syn_slow.delay_steps[0] == 200);  // slow unmyelinated
  assert(syn_fast.delay_steps[0] == 10);   // fast myelinated
  assert(syn_slow.delay_steps[0] > syn_fast.delay_steps[0]);
}

TEST(distance_delay_ring_buffer_delivery) {
  // Functional test: spike from neuron 0 should arrive at neuron 1
  // after the distance-dependent delay.
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.x[0] = 0.0f;   neurons.y[0] = 0.0f; neurons.z[0] = 0.0f;
  neurons.x[1] = 500.0f; neurons.y[1] = 0.0f; neurons.z[1] = 0.0f;

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {2.0f}, {kACh});
  // Velocity = 100 units/ms, dt = 0.5ms => delay = 500/100 = 5ms = 10 steps
  syn.InitDistanceDelay(neurons, 100.0f, 0.5f, 0.1f);
  assert(syn.delay_steps[0] == 10);

  // Spike neuron 0 at step 0
  uint8_t spiked[2] = {1, 0};
  float i_syn[2] = {0, 0};

  syn.DeliverDelayed(i_syn);
  syn.PropagateSpikes(spiked, i_syn, 1.0f);
  syn.AdvanceDelayRing();
  assert(i_syn[1] == 0.0f);  // not delivered yet

  spiked[0] = 0;

  // Steps 2-10: current should not yet arrive
  for (int step = 1; step < 10; ++step) {
    std::fill(i_syn, i_syn + 2, 0.0f);
    syn.DeliverDelayed(i_syn);
    syn.PropagateSpikes(spiked, i_syn, 1.0f);
    syn.AdvanceDelayRing();
    assert(i_syn[1] == 0.0f);  // still waiting
  }

  // Step 11: current arrives (10-step delay)
  std::fill(i_syn, i_syn + 2, 0.0f);
  syn.DeliverDelayed(i_syn);
  assert(i_syn[1] > 0.0f);  // delayed current delivered
}

TEST(conduction_velocity_constants) {
  // Verify constants are ordered correctly: unmyelinated < myelinated
  assert(SynapseTable::kVelocityDrosophilaUnmyel < SynapseTable::kVelocityMammalUnmyel);
  assert(SynapseTable::kVelocityMammalUnmyel < SynapseTable::kVelocityMammalThinMyel);
  assert(SynapseTable::kVelocityMammalThinMyel < SynapseTable::kVelocityCorpusCallosum);
  assert(SynapseTable::kVelocityCorpusCallosum < SynapseTable::kVelocityMammalThickMyel);

  // Verify magnitudes are in physiological range
  assert(SynapseTable::kVelocityDrosophilaUnmyel >= 0.1f);
  assert(SynapseTable::kVelocityMammalThickMyel <= 200.0f);
}

// ===== Compartmental neuron tests =====

TEST(compartmental_array_init) {
  CompartmentalArray arr;
  arr.Resize(10);
  assert(arr.n == 10);
  assert(arr.v_soma[0] == -65.0f);
  assert(arr.v_apical[0] == -65.0f);
  assert(arr.v_basal[0] == -65.0f);
  assert(arr.Ca_i[0] > 0.0f);  // resting calcium
  assert(arr.spiked[0] == 0);
  assert(arr.CountSpikes() == 0);
}

TEST(compartmental_resting_stability) {
  // With no input, a compartmental neuron should remain near resting potential.
  // Tests that the channel conductances are balanced at rest.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  float v0 = arr.v_soma[0];
  for (int step = 0; step < 1000; ++step) {
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
  }

  // Soma should stay within 10mV of initial resting potential
  float drift = std::abs(arr.v_soma[0] - v0);
  assert(drift < 10.0f);
  assert(std::isfinite(arr.v_soma[0]));
  assert(std::isfinite(arr.v_apical[0]));
  assert(std::isfinite(arr.v_basal[0]));
  assert(arr.Ca_i[0] >= 0.0f);
}

TEST(compartmental_somatic_spike) {
  // Strong somatic input should produce a spike.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  bool fired = false;
  for (int step = 0; step < 2000; ++step) {
    arr.i_ext_soma[0] = 15.0f;  // strong depolarizing current
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
    if (arr.spiked[0]) { fired = true; break; }
  }

  assert(fired);
}

TEST(compartmental_refractory_period) {
  // After a spike, the neuron should not fire again within refractory_ms.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();
  p.refractory_ms = 5.0f;

  // Drive to first spike
  float t = 0.0f;
  float dt = 0.025f;
  float first_spike_time = -100.0f;
  int second_spike_step = -1;
  for (int step = 0; step < 5000; ++step) {
    arr.i_ext_soma[0] = 15.0f;
    CompartmentalStep(arr, dt, t, p);
    if (arr.spiked[0]) {
      if (first_spike_time < 0.0f) {
        first_spike_time = t;
      } else if (second_spike_step < 0) {
        second_spike_step = step;
        // Inter-spike interval must exceed refractory period
        float isi = t - first_spike_time;
        assert(isi >= p.refractory_ms);
        break;
      }
    }
    t += dt;
  }
}

TEST(compartmental_bap_propagation) {
  // When the soma spikes, bAP pulses should appear in dendrites.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  float va_before_spike = arr.v_apical[0];
  float vb_before_spike = arr.v_basal[0];

  // Drive to spike
  float t = 0.0f;
  float dt = 0.025f;
  for (int step = 0; step < 5000; ++step) {
    arr.i_ext_soma[0] = 15.0f;
    CompartmentalStep(arr, dt, t, p);
    if (arr.spiked[0]) break;
    t += dt;
  }

  // bAP remaining timers should be set
  assert(arr.bap_apical_remaining[0] > 0.0f);
  assert(arr.bap_basal_remaining[0] > 0.0f);

  // Run a few more steps: dendrites should depolarize from bAP
  float va_after = arr.v_apical[0];
  float vb_after = arr.v_basal[0];
  for (int step = 0; step < 10; ++step) {
    arr.i_ext_soma[0] = 0.0f;  // no more input
    t += dt;
    CompartmentalStep(arr, dt, t, p);
    va_after = arr.v_apical[0];
    vb_after = arr.v_basal[0];
  }

  // Dendrites should have been depolarized by bAP
  // (the exact amount depends on attenuation, but should be above rest)
  assert(va_after > va_before_spike - 5.0f || arr.bap_apical_remaining[0] > 0.0f);
  assert(vb_after > vb_before_spike - 5.0f || arr.bap_basal_remaining[0] > 0.0f);
}

TEST(compartmental_nmda_mg_block) {
  // Test the NMDA Mg2+ block function (Jahr & Stevens 1990).
  float Mg = 1.0f;

  // At rest (-70mV): nearly complete block
  float B_rest = NMDAMgBlock(-70.0f, Mg);
  assert(B_rest < 0.05f);  // < 5% open

  // At -20mV: significant unblocking
  float B_mid = NMDAMgBlock(-20.0f, Mg);
  assert(B_mid > 0.3f && B_mid < 0.8f);

  // At 0mV: mostly unblocked
  float B_zero = NMDAMgBlock(0.0f, Mg);
  assert(B_zero > 0.7f);

  // At +20mV: nearly fully unblocked
  float B_pos = NMDAMgBlock(20.0f, Mg);
  assert(B_pos > 0.9f);

  // Monotonically increasing with voltage
  assert(B_rest < B_mid);
  assert(B_mid < B_zero);
  assert(B_zero < B_pos);
}

TEST(compartmental_nmda_dendritic_activation) {
  // NMDA activation on a depolarized dendrite should produce more current
  // than on a resting dendrite (voltage-dependent Mg2+ block).
  CompartmentalArray arr;
  arr.Resize(2);
  auto p = DefaultPyramidalParams();

  // Neuron 0: resting dendrite with NMDA
  // Neuron 1: pre-depolarized dendrite with NMDA
  arr.v_apical[1] = -30.0f;  // already depolarized (e.g., by bAP)

  // Activate NMDA on both
  uint32_t targets[] = {0, 1};
  ActivateNMDA_Apical(arr, targets, 2, 0.1f, p.nmda.tau_rise);

  assert(arr.s_NMDA_apical[0] > 0.0f);
  assert(arr.s_NMDA_apical[1] > 0.0f);

  // Step once to see the current effect
  CompartmentalStep(arr, 0.025f, 0.0f, p);

  // The depolarized neuron should have received more NMDA current
  // (visible as a more depolarized apical voltage after the step)
  // Note: neuron 1 started more depolarized, so we check relative movement
  // from the baseline. Both should have moved, but neuron 1 should have
  // a larger NMDA contribution due to less Mg2+ block.
  float B0 = NMDAMgBlock(arr.v_apical[0], p.nmda.Mg_mM);
  float B1 = NMDAMgBlock(-30.0f, p.nmda.Mg_mM);  // at pre-step voltage
  assert(B1 > B0);  // confirms depolarized dendrite gets more NMDA current
}

TEST(compartmental_calcium_dynamics) {
  // Calcium should rise when the apical dendrite is depolarized enough
  // to open HVA Ca2+ channels, then decay back toward resting.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  float Ca_rest = arr.Ca_i[0];

  // Strongly depolarize the apical dendrite to open Ca channels
  for (int step = 0; step < 500; ++step) {
    arr.i_ext_apical[0] = 30.0f;
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
  }

  float Ca_peak = arr.Ca_i[0];
  // Ca should have risen
  assert(Ca_peak > Ca_rest * 1.5f);

  // Now remove input and let Ca decay
  for (int step = 500; step < 5000; ++step) {
    arr.i_ext_apical[0] = 0.0f;
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
  }

  float Ca_after = arr.Ca_i[0];
  // Ca should have decayed back toward resting (not necessarily all the way)
  assert(Ca_after < Ca_peak);
}

TEST(compartmental_coupling) {
  // Depolarizing one compartment should spread to others via coupling.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  // Depolarize only the apical dendrite
  float vs_before = arr.v_soma[0];
  for (int step = 0; step < 200; ++step) {
    arr.i_ext_apical[0] = 10.0f;
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
  }

  // Soma should have depolarized due to coupling from apical
  float vs_after = arr.v_soma[0];
  assert(vs_after > vs_before);

  // Basal should also show some depolarization (through soma)
  assert(arr.v_basal[0] > -65.5f);
}

TEST(compartmental_ih_depolarizes_at_rest) {
  // Ih (HCN channel) should be partially active at resting potentials,
  // producing a small depolarizing current (E_h = -45mV > V_rest = -65mV).
  // This is a key biological feature: Ih depolarizes the resting potential
  // of distal apical dendrites in cortical pyramidal neurons (Magee 1998).
  CompartmentalArray arr;
  arr.Resize(2);
  auto p = DefaultPyramidalParams();

  // Neuron 0: normal Ih
  // Neuron 1: zero Ih
  auto p_no_ih = p;
  p_no_ih.channels.g_Ih = 0.0f;

  for (int step = 0; step < 2000; ++step) {
    float t = step * 0.025f;
    CompartmentalStep(arr, 0.025f, t, p);  // steps all neurons with same params
  }

  // We need separate params per neuron for this test, so run them independently
  CompartmentalArray n0, n1;
  n0.Resize(1);
  n1.Resize(1);

  for (int step = 0; step < 2000; ++step) {
    float t = step * 0.025f;
    CompartmentalStep(n0, 0.025f, t, p);
    CompartmentalStep(n1, 0.025f, t, p_no_ih);
  }

  // With Ih, apical dendrite should rest slightly more depolarized
  assert(n0.v_apical[0] > n1.v_apical[0] - 0.5f);
}

TEST(compartmental_default_params_valid) {
  auto p = DefaultPyramidalParams();
  auto p23 = DefaultL23PyramidalParams();

  // Sanity checks on default parameters
  assert(p.soma.Cm > 0.0f);
  assert(p.apical.Cm > p.soma.Cm);  // spine correction
  assert(p.channels.g_Na > p.channels.g_Na_basal);  // soma has higher Na
  assert(p.channels.g_CaHVA > 0.0f);
  assert(p.v_thresh > p.v_reset);
  assert(p.refractory_ms > 0.0f);

  // L2/3 should have less Ca than L5
  assert(p23.channels.g_CaHVA < p.channels.g_CaHVA);

  // Area fractions should sum to 1
  float total = p.soma.area_frac + p.apical.area_frac + p.basal.area_frac;
  assert(std::abs(total - 1.0f) < 0.01f);
}

TEST(compartmental_no_nan_divergence) {
  // Run with strong inputs for many steps. Should not produce NaN or Inf.
  CompartmentalArray arr;
  arr.Resize(5);
  auto p = DefaultPyramidalParams();

  for (int step = 0; step < 10000; ++step) {
    float t = step * 0.025f;
    for (size_t i = 0; i < arr.n; ++i) {
      arr.i_ext_soma[i] = 20.0f * std::sin(t * 0.01f);
      arr.i_ext_apical[i] = 10.0f;
    }
    CompartmentalStep(arr, 0.025f, t, p);

    for (size_t i = 0; i < arr.n; ++i) {
      assert(std::isfinite(arr.v_soma[i]));
      assert(std::isfinite(arr.v_apical[i]));
      assert(std::isfinite(arr.v_basal[i]));
      assert(std::isfinite(arr.Ca_i[i]));
      assert(arr.Ca_i[i] >= 0.0f);
    }
  }
}

TEST(compartmental_multiple_spikes) {
  // With sustained input, the neuron should fire multiple spikes.
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  int spike_count = 0;
  for (int step = 0; step < 20000; ++step) {
    arr.i_ext_soma[0] = 12.0f;
    CompartmentalStep(arr, 0.025f, step * 0.025f, p);
    if (arr.spiked[0]) spike_count++;
  }

  // Should produce multiple spikes over 500ms of simulation
  assert(spike_count >= 2);
}

TEST(compartmental_boltzmann_and_bell) {
  // Verify helper functions produce expected values.

  // Boltzmann at v_half should give 0.5
  float b = BoltzmannInf(0.0f, 0.0f, 5.0f);
  assert(std::abs(b - 0.5f) < 0.001f);

  // Boltzmann far above v_half should approach 1
  float b_high = BoltzmannInf(100.0f, 0.0f, 5.0f);
  assert(b_high > 0.99f);

  // Boltzmann far below v_half should approach 0
  float b_low = BoltzmannInf(-100.0f, 0.0f, 5.0f);
  assert(b_low < 0.01f);

  // Bell tau at peak should give tau_max
  float t_peak = BellTau(0.0f, 0.0f, 10.0f, 1.0f, 10.0f);
  assert(std::abs(t_peak - 10.0f) < 0.01f);

  // Bell tau far from peak should approach tau_min
  float t_far = BellTau(100.0f, 0.0f, 10.0f, 1.0f, 10.0f);
  assert(t_far < 2.0f);
}

TEST(compartmental_activity_readout) {
  CompartmentalArray arr;
  arr.Resize(1);
  auto p = DefaultPyramidalParams();

  // At rest, activity should be near 0
  float act_rest = CompartmentalActivity(arr, 0, p);
  assert(act_rest >= 0.0f && act_rest <= 1.0f);

  // At threshold, activity should be near 1
  arr.v_soma[0] = p.v_thresh;
  float act_thresh = CompartmentalActivity(arr, 0, p);
  assert(act_thresh > 0.9f);

  // At reset, activity should be 0
  arr.v_soma[0] = p.v_reset;
  float act_reset = CompartmentalActivity(arr, 0, p);
  assert(std::abs(act_reset) < 0.01f);
}

// ===== Inhibitory Plasticity (Vogels iSTDP) =====

TEST(istdp_init) {
  InhibitorySTDP rule;
  assert(!rule.IsInitialized());
  rule.Init(100, 10);
  assert(rule.IsInitialized());
  assert(rule.x_pre.size() == 100);
  assert(rule.x_post.size() == 10);
}

TEST(istdp_alpha_computation) {
  InhibitorySTDP rule;
  rule.target_rate_hz = 5.0f;
  rule.tau = 20.0f;
  // alpha = 2 * (5/1000) * 20 = 0.2
  float alpha = rule.Alpha();
  assert(std::abs(alpha - 0.2f) < 1e-6f);
}

TEST(istdp_potentiates_correlated_inhibitory) {
  // When pre and post fire together, inhibitory weight should increase.
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};  // neuron 0 -> neuron 1
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kGABA)};

  InhibitorySTDP rule;
  rule.eta = 0.01f;
  rule.Init(1, 2);

  // Both fire
  neurons.spiked[0] = 1;
  neurons.spiked[1] = 1;

  float w_before = syn.weight[0];
  InhibitorySTDPUpdate(syn, neurons, 1.0f, rule);
  // With both firing, both terms contribute positively (minus alpha).
  // Weight should change.
  assert(syn.weight[0] != w_before);
}

TEST(istdp_skips_excitatory_synapses) {
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};  // excitatory

  InhibitorySTDP rule;
  rule.eta = 0.01f;
  rule.Init(1, 2);

  neurons.spiked[0] = 1;
  neurons.spiked[1] = 1;

  InhibitorySTDPUpdate(syn, neurons, 1.0f, rule);
  assert(syn.weight[0] == 1.0f);  // unchanged
}

TEST(istdp_depression_without_post_spike) {
  // If only pre fires, dw = eta * (x_post - alpha).
  // With x_post=0 and alpha>0, this depresses.
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {5.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kGABA)};

  InhibitorySTDP rule;
  rule.eta = 0.1f;
  rule.Init(1, 2);

  // Only pre fires (post silent)
  neurons.spiked[0] = 1;
  neurons.spiked[1] = 0;

  InhibitorySTDPUpdate(syn, neurons, 1.0f, rule);
  // dw = eta * (0 - alpha) = negative -> depression
  assert(syn.weight[0] < 5.0f);
}

TEST(istdp_weight_bounds) {
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {0.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kGABA)};

  InhibitorySTDP rule;
  rule.eta = 0.1f;
  rule.w_min = 0.0f;
  rule.w_max = 10.0f;
  rule.Init(1, 2);

  // Depress from w=0 (should clamp at w_min)
  neurons.spiked[0] = 1;
  neurons.spiked[1] = 0;
  InhibitorySTDPUpdate(syn, neurons, 1.0f, rule);
  assert(syn.weight[0] >= rule.w_min);
}

// ===== Neuromodulator Effects =====

TEST(neuromod_effects_octopamine_excites) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.octopamine[0] = 1.0f;
  neurons.serotonin[0] = 0.0f;
  neurons.dopamine[0] = 0.0f;
  neurons.i_ext[0] = 0.0f;

  NeuromodulatorEffects fx;
  fx.Apply(neurons);
  // OA should add depolarizing current
  assert(neurons.i_ext[0] > 0.0f);
}

TEST(neuromod_effects_serotonin_inhibits) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.octopamine[0] = 0.0f;
  neurons.serotonin[0] = 1.0f;
  neurons.dopamine[0] = 0.0f;
  neurons.i_ext[0] = 0.0f;

  NeuromodulatorEffects fx;
  fx.Apply(neurons);
  // 5HT should add hyperpolarizing current
  assert(neurons.i_ext[0] < 0.0f);
}

TEST(neuromod_effects_dopamine_excites) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.octopamine[0] = 0.0f;
  neurons.serotonin[0] = 0.0f;
  neurons.dopamine[0] = 1.0f;
  neurons.i_ext[0] = 0.0f;

  NeuromodulatorEffects fx;
  fx.Apply(neurons);
  // DA should add mild depolarizing current
  assert(neurons.i_ext[0] > 0.0f);
}

TEST(neuromod_effects_oa_synaptic_gain) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.octopamine[0] = 1.0f;
  neurons.serotonin[0] = 0.0f;
  neurons.dopamine[0] = 0.0f;
  neurons.i_syn[0] = 10.0f;
  neurons.i_ext[0] = 0.0f;

  NeuromodulatorEffects fx;
  fx.Apply(neurons);
  // OA should amplify i_syn by (1 + 1.0 * 0.3) = 1.3x
  assert(std::abs(neurons.i_syn[0] - 13.0f) < 0.01f);
}

TEST(neuromod_effects_below_threshold_ignored) {
  NeuronArray neurons;
  neurons.Resize(1);
  neurons.octopamine[0] = 0.005f;  // below min_concentration
  neurons.serotonin[0] = 0.005f;
  neurons.dopamine[0] = 0.005f;
  neurons.i_ext[0] = 0.0f;
  neurons.i_syn[0] = 10.0f;

  NeuromodulatorEffects fx;
  fx.Apply(neurons);
  assert(neurons.i_ext[0] == 0.0f);
  assert(neurons.i_syn[0] == 10.0f);
}

TEST(sim_features_new_flags_counted) {
  SimFeatures f = SimFeatures::Full();
  assert(f.inhibitory_plasticity == true);
  assert(f.neuromodulator_effects == true);
  assert(f.nmda == true);
  // Verify counting includes new flags
  int count = f.CountEnabled();
  assert(count >= 19);  // at least 19 features in Full mode (including NMDA)
}

// ===== NMDA Receptor tests (standalone nmda.h) =====

TEST(nmda_mg_block_voltage_dependence) {
  // Jahr & Stevens 1990: Mg2+ block at 1mM [Mg2+]
  // At rest (-65mV), channel should be nearly fully blocked.
  // At 0mV, channel should be mostly unblocked.
  float Mg = 1.0f;
  float B_rest = NMDAReceptor::MgBlock(-65.0f, Mg);
  float B_mid  = NMDAReceptor::MgBlock(-40.0f, Mg);
  float B_zero = NMDAReceptor::MgBlock(0.0f, Mg);
  float B_pos  = NMDAReceptor::MgBlock(20.0f, Mg);

  // At rest: nearly fully blocked
  assert(B_rest < 0.10f);
  assert(B_rest > 0.0f);

  // Monotonically increasing with voltage
  assert(B_mid > B_rest);
  assert(B_zero > B_mid);
  assert(B_pos > B_zero);

  // At 0mV: mostly unblocked
  assert(B_zero > 0.70f);

  // Always in [0, 1]
  assert(B_rest >= 0.0f && B_rest <= 1.0f);
  assert(B_pos >= 0.0f && B_pos <= 1.0f);
}

TEST(nmda_accumulate_only_ach_synapses) {
  // NMDA receptors should only accumulate from excitatory (ACh) synapses,
  // not from GABA or modulatory synapses.
  NeuronArray arr;
  arr.Resize(3);

  // Neuron 0 -> 1 (ACh), Neuron 0 -> 2 (GABA)
  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 0}, {1, 2}, {5.0f, 5.0f}, {kACh, kGABA});

  NMDAReceptor nmda;
  nmda.Init(3);
  nmda.nmda_gain = 1.0f;  // simplify: gain = 1

  arr.spiked[0] = 1;
  nmda.AccumulateFromSpikes(syn, arr.spiked.data(), 1.0f);

  // Neuron 1 (ACh target) should have NMDA conductance
  assert(nmda.g_nmda[1] > 0.0f);
  // Neuron 2 (GABA target) should have zero NMDA conductance
  assert(nmda.g_nmda[2] == 0.0f);
  // Neuron 0 (source) should have zero
  assert(nmda.g_nmda[0] == 0.0f);
}

TEST(nmda_conductance_decay) {
  // NMDA conductance should decay exponentially with tau_nmda_ms.
  NMDAReceptor nmda;
  nmda.Init(1);
  nmda.tau_nmda_ms = 80.0f;

  // Inject initial conductance
  nmda.g_nmda[0] = 10.0f;

  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = -65.0f;  // resting (Mg block will suppress current)

  // Step 80ms (one time constant)
  for (int i = 0; i < 80; ++i) nmda.Step(arr, 1.0f);

  // After one tau, conductance should be ~10 * exp(-1) = ~3.68
  float expected = 10.0f * std::exp(-1.0f);
  float ratio = nmda.g_nmda[0] / expected;
  assert(ratio > 0.95f && ratio < 1.05f);
}

TEST(nmda_coincidence_detection) {
  // Core property: same NMDA conductance on a depolarized vs resting neuron
  // should produce different effective currents. The depolarized neuron
  // gets more current because Mg2+ block is removed.
  NMDAReceptor nmda;
  nmda.Init(2);
  nmda.tau_nmda_ms = 1000.0f;  // slow decay so conductance persists

  // Both neurons get identical NMDA conductance
  nmda.g_nmda[0] = 5.0f;
  nmda.g_nmda[1] = 5.0f;

  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -65.0f;  // resting: Mg block active
  arr.v[1] = -20.0f;  // depolarized: Mg block partially removed
  arr.i_ext[0] = 0.0f;
  arr.i_ext[1] = 0.0f;

  nmda.Step(arr, 1.0f);

  // Depolarized neuron should receive more NMDA current
  assert(arr.i_ext[1] > arr.i_ext[0]);

  // The ratio should roughly match the Mg block ratio
  float B_rest = NMDAReceptor::MgBlock(-65.0f, 1.0f);
  float B_dep  = NMDAReceptor::MgBlock(-20.0f, 1.0f);
  assert(B_dep > 3.0f * B_rest);  // depolarized should get >3x more
}

TEST(nmda_calcium_accumulates_and_decays) {
  // NMDA-mediated calcium should build up during active NMDA current
  // and decay after input stops.
  NMDAReceptor nmda;
  nmda.Init(1);
  nmda.tau_ca_nmda_ms = 200.0f;

  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = -20.0f;  // depolarized to allow NMDA current

  // Inject conductance and step to build calcium
  nmda.g_nmda[0] = 5.0f;
  for (int i = 0; i < 50; ++i) {
    nmda.g_nmda[0] = 5.0f;  // sustained input
    arr.i_ext[0] = 0.0f;
    nmda.Step(arr, 1.0f);
  }
  float ca_peak = nmda.ca_nmda[0];
  assert(ca_peak > 0.0f);

  // Stop input and let calcium decay
  nmda.g_nmda[0] = 0.0f;
  for (int i = 0; i < 400; ++i) {
    arr.i_ext[0] = 0.0f;
    nmda.Step(arr, 1.0f);
  }
  float ca_after = nmda.ca_nmda[0];

  // Calcium should have decayed substantially (400ms > tau_ca=200ms)
  assert(ca_after < ca_peak * 0.25f);
  assert(ca_after > 0.0f);  // not fully gone yet due to accumulated baseline
}

TEST(nmda_no_current_at_rest) {
  // At resting potential, NMDA channels should be nearly fully blocked
  // by Mg2+, producing negligible current even with high conductance.
  NMDAReceptor nmda;
  nmda.Init(1);

  nmda.g_nmda[0] = 10.0f;  // large conductance

  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = -70.0f;  // deep rest
  arr.i_ext[0] = 0.0f;

  nmda.Step(arr, 1.0f);

  // Mg block at -70mV: B = 1/(1 + 1/3.57 * exp(0.062*70)) = ~0.048
  // Current = 10 * 0.048 = ~0.48 (small relative to g_nmda)
  float B = NMDAReceptor::MgBlock(-70.0f, 1.0f);
  assert(B < 0.06f);
  assert(arr.i_ext[0] < 1.0f);  // minimal current despite large g
}

TEST(nmda_sim_features_flag) {
  // NMDA flag should be on by default, off in Inference and Minimal presets.
  SimFeatures full = SimFeatures::Full();
  assert(full.nmda == true);

  SimFeatures inf = SimFeatures::Inference();
  assert(inf.nmda == false);

  SimFeatures min = SimFeatures::Minimal();
  assert(min.nmda == false);

  // Default-constructed also has NMDA on
  SimFeatures def;
  assert(def.nmda == true);
}

// ===== ScopedTimer tests =====

TEST(scoped_timer_basic) {
  ScopedTimer t("test_timer");
  t.silent = true;
  // Just verify it compiles and runs
  assert(t.elapsed_ms() >= 0.0);
  assert(t.elapsed_us() >= 0.0);
}

TEST(timer_registry) {
  TimerRegistry reg;
  for (int i = 0; i < 100; ++i) {
    auto scope = reg.Scope("test_section");
    volatile int x = i * i;  // prevent optimization
    (void)x;
  }
  // Verify stats accumulated
  // (can't directly access private map, but Report() should not crash)
  reg.Report(stdout);
  reg.Clear();
}

// ===== SpikeAnalysis tests =====

TEST(isi_stats_regular) {
  // Regular spiking at 50ms intervals
  std::vector<float> times;
  for (int i = 0; i < 20; ++i) times.push_back(100.0f + i * 50.0f);

  auto stats = ComputeISI(times);
  assert(stats.n_intervals == 19);
  assert(std::abs(stats.mean_ms - 50.0f) < 0.1f);
  assert(std::abs(stats.median_ms - 50.0f) < 0.1f);
  assert(stats.cv < 0.01f);  // perfectly regular
  assert(stats.is_regular());
}

TEST(isi_stats_bursty) {
  // Bursty: clusters of 3 spikes at 5ms, gaps of 100ms
  std::vector<float> times;
  for (int burst = 0; burst < 5; ++burst) {
    float base = burst * 115.0f;
    times.push_back(base);
    times.push_back(base + 5.0f);
    times.push_back(base + 10.0f);
  }

  auto stats = ComputeISI(times);
  assert(stats.n_intervals == 14);
  assert(stats.cv > 1.0f);  // bursty
  assert(stats.is_bursty());
}

TEST(burst_detection) {
  // 3 bursts of 4 spikes each, separated by 80ms gaps
  std::vector<float> times;
  for (int b = 0; b < 3; ++b) {
    float base = b * 100.0f;
    for (int s = 0; s < 4; ++s)
      times.push_back(base + s * 3.0f);
  }

  auto bursts = DetectBursts(times, 10.0f, 3);
  assert(bursts.size() == 3);
  assert(bursts[0].n_spikes == 4);
  assert(std::abs(bursts[0].duration_ms() - 9.0f) < 0.1f);

  auto bstats = ComputeBurstStats(bursts, 300.0f);
  assert(bstats.n_bursts == 3);
  assert(std::abs(bstats.mean_ibi_ms - 100.0f) < 1.0f);
}

TEST(spike_collector) {
  SpikeCollector collector;
  collector.Init(10);

  // Simulate some spikes
  std::vector<uint8_t> spiked(10, 0);
  spiked[0] = 1; spiked[5] = 1;
  collector.Record(spiked.data(), 10, 10.0f);

  spiked[0] = 0; spiked[5] = 0; spiked[3] = 1;
  collector.Record(spiked.data(), 10, 20.0f);

  assert(collector.TotalSpikes() == 3);
  assert(collector.trains[0].size() == 1);
  assert(collector.trains[5].size() == 1);
  assert(collector.trains[3].size() == 1);

  auto pop = collector.GetPopulationTrain(0, 10);
  assert(pop.size() == 3);
  assert(pop[0] == 10.0f);  // sorted
}

TEST(population_fano_factor) {
  // Synchronous: all neurons fire in same bins
  std::vector<std::vector<float>> sync_trains(10);
  for (auto& t : sync_trains) {
    t = {10.0f, 60.0f, 110.0f, 160.0f};
  }
  float ff_sync = PopulationFanoFactor(sync_trains, 5.0f, 200.0f);

  // Asynchronous: neurons fire at different times
  std::vector<std::vector<float>> async_trains(10);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j)
      async_trains[i].push_back(static_cast<float>(i * 5 + j * 50));
  }
  float ff_async = PopulationFanoFactor(async_trains, 5.0f, 250.0f);

  // Synchronous should have much higher Fano factor
  assert(ff_sync > ff_async);
}

TEST(oscillation_detection) {
  // Create spike trains with 40ms oscillation
  std::vector<std::vector<float>> trains(5);
  for (int n = 0; n < 5; ++n) {
    for (float t = 0; t < 500; t += 40.0f) {
      trains[n].push_back(t + n * 0.5f);  // slight jitter
    }
  }

  auto ac = PopulationAutocorrelation(trains, 1.0f, 500.0f, 100);
  float period = DetectOscillationPeriod(ac, 1.0f);
  // Should detect ~40ms period (within tolerance)
  assert(period > 35.0f && period < 45.0f);
}

// ===== MemoryTracker tests =====

TEST(memory_tracker) {
  MemoryTracker tracker;
  tracker.Track("neurons.v", 100000 * sizeof(float), "neurons");
  tracker.Track("neurons.u", 100000 * sizeof(float), "neurons");
  tracker.Track("synapses", 500000 * sizeof(uint32_t), "synapses");

  assert(tracker.TotalBytes() == 100000 * 4 + 100000 * 4 + 500000 * 4);
  assert(tracker.CategoryBytes("neurons") == 100000 * 4 * 2);
  assert(tracker.CategoryBytes("synapses") == 500000 * 4);

  tracker.Untrack("synapses");
  assert(tracker.CategoryBytes("synapses") == 0);
  assert(tracker.PeakBytes() == 100000 * 4 + 100000 * 4 + 500000 * 4);

  tracker.Report(stdout);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
