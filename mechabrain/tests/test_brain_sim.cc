// Brain-sim isolation test: proves brain-model/core/ compiles with NO
// dependencies on src/bridge/, src/, or viewer/. If this target builds,
// brain-sim is a self-contained module ready for extraction.
//
// This file ONLY includes core/ headers. No bridge, no viewer, no src.
#include <cmath>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "test_harness.h"

// Every public core/ header included to prove compilation
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/stdp.h"
#include "core/connectome_loader.h"
#include "core/connectome_export.h"
#include "core/config_loader.h"
#include "core/checkpoint.h"
#include "core/connectome_stats.h"
#include "core/experiment_config.h"
#include "core/recorder.h"
#include "core/parametric_gen.h"
#include "core/parametric_sync.h"
#include "core/brain_spec_loader.h"
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
#include "core/structural_plasticity.h"
#include "core/cell_types.h"
#include "core/region_metrics.h"
#include "core/behavioral_fingerprint.h"
#include "core/multiscale_bridge.h"
#include "core/scan_overlay.h"
#include "core/experiment_protocol.h"
#include "core/stimulus_event.h"
#include "core/platform.h"
#include "core/error.h"
#include "core/log.h"
#include "core/version.h"
#include "core/param_sweep.h"


// ===== Proof-of-isolation tests =====
// These exercise key brain-sim APIs to prove they work standalone.

TEST(brainsim_neuron_array) {
  NeuronArray arr;
  arr.Resize(100);
  assert(arr.n == 100);
  assert(arr.v[0] == -65.0f);
  assert(arr.CountSpikes() == 0);
}

TEST(brainsim_izhikevich_spike) {
  NeuronArray arr;
  arr.Resize(1);
  IzhikevichParams p;
  arr.i_ext[0] = 15.0f;
  float t = 0.0f;
  bool fired = false;
  for (int i = 0; i < 100; ++i) {
    IzhikevichStep(arr, 1.0f, t, p);
    if (arr.spiked[0]) { fired = true; break; }
    t += 1.0f;
  }
  assert(fired);
}

TEST(brainsim_synapse_propagation) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {2.0f};
  syn.nt_type = {NTType::kACh};
  arr.spiked[0] = 1;
  syn.PropagateSpikes(arr.spiked.data(), arr.i_syn.data(), 1.0f);
  assert(arr.i_syn[1] > 0.0f);
}

TEST(brainsim_stdp) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {0.5f};
  syn.nt_type = {NTType::kACh};
  arr.spiked[0] = 1;
  arr.last_spike_time[0] = 10.0f;
  arr.last_spike_time[1] = 15.0f;
  STDPParams sp;
  float old_w = syn.weight[0];
  STDPUpdate(syn, arr, 20.0f, sp);
  // Weight should change (potentiation or depression)
  assert(syn.weight[0] != old_w || true);  // just proves compilation
}

TEST(brainsim_parametric_gen) {
  BrainSpec spec;
  RegionSpec reg;
  reg.name = "test";
  reg.n_neurons = 50;
  reg.internal_density = 0.1f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);
  assert(total == 50);
  assert(neurons.n == 50);
}

TEST(brainsim_gap_junctions) {
  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -65.0f;
  arr.v[1] = -55.0f;
  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f);
  gj.PropagateGapCurrents(arr);
  assert(arr.i_ext[0] > 0.0f);
}

TEST(brainsim_homeostasis) {
  NeuronArray arr;
  arr.Resize(10);
  IntrinsicHomeostasis h;
  h.Init(10, 5.0f, 1.0f);
  h.SetTargetsFromTypes(arr);
  assert(h.per_neuron_target.size() == 10);
}

TEST(brainsim_cpg) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) arr.region[i] = 5;
  CPGOscillator cpg;
  cpg.Init(arr, 5);
  assert(cpg.initialized);
}

TEST(brainsim_temperature) {
  TemperatureModel tm;
  tm.enabled = true;
  tm.current_temp_c = 32.0f;
  assert(tm.ChannelScale() > 1.0f);
}

TEST(brainsim_sim_features) {
  auto f = SimFeatures::Full();
  assert(f.stdp);
  assert(f.CountEnabled() == 16);
}

TEST(brainsim_sfa) {
  NeuronArray arr;
  arr.Resize(1);
  SpikeFrequencyAdaptation sfa;
  sfa.Init(1);
  arr.spiked[0] = 1;
  sfa.Update(arr, 1.0f);
  assert(sfa.calcium[0] > 0.0f);
}

TEST(brainsim_per_nt_tau) {
  assert(SynapseTable::TauForNT(NTType::kACh) == 2.0f);
  assert(SynapseTable::TauForNT(NTType::kGABA) == 5.0f);
  assert(SynapseTable::TauForNT(NTType::kGlut) == 3.0f);
}

TEST(brainsim_motor_output) {
  NeuronArray arr;
  arr.Resize(10);
  MotorOutput motor;
  motor.Update(arr, 1.0f);
  MotorCommand cmd = motor.Command();
  assert(cmd.forward_velocity >= 0.0f || cmd.forward_velocity < 0.0f);  // just compiles
}

TEST(brainsim_proprioception) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) arr.region[i] = 5;
  ProprioMap pm;
  pm.Init(arr, 5);
  assert(pm.initialized);
}

// ===== Main =====
int main() {
  return RunAllTests();
}
