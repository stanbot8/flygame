#ifndef FWMC_COURTSHIP_EXPERIMENT_H_
#define FWMC_COURTSHIP_EXPERIMENT_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/rate_monitor.h"
#include "core/spike_analysis.h"
#include "core/spike_frequency_adaptation.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Results from a courtship song generation experiment.
//
// Drosophila melanogaster males produce two types of courtship song
// by vibrating one wing:
//   Pulse song: discrete clicks at ~35ms inter-pulse interval (IPI)
//   Sine song: continuous ~150 Hz hum
//
// The circuit uses nested pathways: sine neurons are a subset of pulse
// neurons, with additional pulse-specific neurons recruited to switch
// from sine to pulse mode (Calhoun et al. 2024 Curr Biol).
struct CourtshipResult {
  // Pulse song metrics
  float mean_ipi_ms = 0.0f;         // inter-pulse interval (lit: 30-45ms)
  float ipi_cv = 0.0f;              // coefficient of variation of IPI
  int n_pulses = 0;                  // number of detected pulses
  float pulse_bout_duration_ms = 0.0f; // total time producing pulses

  // Sine song metrics
  float sine_rate_hz = 0.0f;        // motor neuron rate during sine (lit: 150-160 Hz)
  float sine_duration_ms = 0.0f;    // total time producing sine

  // Motor output
  float motor_peak_rate_hz = 0.0f;
  int motor_total_spikes = 0;

  // Circuit activation
  float p1_rate_hz = 0.0f;
  float pip10_rate_hz = 0.0f;
  float dpr1_rate_hz = 0.0f;
  float vpr9_rate_hz = 0.0f;        // inhibitory feedback rate

  // Timing
  double elapsed_seconds = 0.0;

  // Validation: pulse song should have IPI in range 25-50ms
  // (D. melanogaster ~35ms, von Philipsborn et al. 2011)
  bool produced_pulses() const { return n_pulses >= 3; }

  bool ipi_plausible() const {
    if (!produced_pulses()) return false;
    return mean_ipi_ms > 20.0f && mean_ipi_ms < 60.0f;
  }
};

// Self-contained Drosophila courtship song generation experiment.
//
// Circuit (Ding et al. 2019, Calhoun et al. 2024):
//   P1 (10) -> pIP10 (20) -> dPR1 (10) -> MN_pulse (20)
//                         \-> TN1A (10) -> MN_sine (10)
//              vPR9 (10) -| dPR1, pIP10 (inhibitory feedback)
//              vPR6 (10) -> dPR1 (IPI timing)
//   Total: 100 neurons
//
// P1: male-specific command neurons, activated by female pheromones.
//     Persistent internal arousal state (Hoopfer et al. 2015 eLife).
// pIP10: descending interneurons, bridge brain to thorax VNC.
//     Cholinergic, drive both song types (Ding et al. 2019).
// dPR1: pulse song pattern generator. Cholinergic, excitatory.
//     Optogenetic activation produces normal pulse trains.
// TN1A: sine song pathway neuron. Active during sine song.
// vPR6: IPI timing controller. Modulates pulse rate.
// vPR9: GABAergic inhibitory feedback within circuit.
//     Recurrent inhibition creates oscillatory dynamics.
// MN_pulse: haltere/wing motor neurons for pulse production.
// MN_sine: wing motor neurons for sine production.
//
// Key insight: the sine pathway is a subset of the pulse pathway.
// Low drive -> sine only (TN1A pathway).
// High drive -> pulse (dPR1 recruited, overrides sine).
// vPR9 feedback creates rhythmic pulse timing.
//
// References:
//   von Philipsborn et al. 2011 Neuron 69:509 - song neuron identification
//   Ding et al. 2019 Curr Biol 29:1089 - pIP10 descending interneurons
//   Calhoun et al. 2024 Curr Biol 34:1 - nested circuit architecture
//   Hoopfer et al. 2015 eLife 4:e11346 - P1 persistent state
//   Shirangi et al. 2013 Cell 155:1610 - fru+ neuron song circuit
struct CourtshipExperiment {
  // Circuit sizes
  uint32_t n_p1 = 10;          // command neurons
  uint32_t n_pip10 = 20;       // descending interneurons
  uint32_t n_dpr1 = 10;        // pulse pattern generator
  uint32_t n_tn1a = 10;        // sine pathway
  uint32_t n_vpr6 = 10;        // IPI timing
  uint32_t n_vpr9 = 10;        // inhibitory feedback
  uint32_t n_mn_pulse = 20;    // pulse motor neurons
  uint32_t n_mn_sine = 10;     // sine motor neurons

  // Stimulus: P1 activation (simulates female pheromone detection)
  float p1_onset_ms = 200.0f;
  float p1_duration_ms = 500.0f;
  float p1_drive = 15.0f;      // pA to P1 neurons

  // Timing
  float dt_ms = 0.1f;
  float total_duration_ms = 1000.0f;

  // Pulse detection threshold: motor neuron spike count in bin
  int pulse_threshold = 3;
  float pulse_bin_ms = 2.0f;  // finer resolution for IPI measurement

  CourtshipResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    CourtshipResult result;

    uint32_t total = n_p1 + n_pip10 + n_dpr1 + n_tn1a +
                     n_vpr6 + n_vpr9 + n_mn_pulse + n_mn_sine;

    // Region boundaries
    uint32_t r_p1 = 0;
    uint32_t r_pip10 = n_p1;
    uint32_t r_dpr1 = r_pip10 + n_pip10;
    uint32_t r_tn1a = r_dpr1 + n_dpr1;
    uint32_t r_vpr6 = r_tn1a + n_tn1a;
    uint32_t r_vpr9 = r_vpr6 + n_vpr6;
    uint32_t r_mn_pulse = r_vpr9 + n_vpr9;
    uint32_t r_mn_sine = r_mn_pulse + n_mn_pulse;

    // Build neurons
    NeuronArray neurons;
    neurons.Resize(total);
    if (neurons.tau_syn.empty()) neurons.tau_syn.resize(total, 3.0f);

    // Assign regions
    for (uint32_t i = r_p1; i < r_pip10; ++i) neurons.region[i] = 0;
    for (uint32_t i = r_pip10; i < r_dpr1; ++i) neurons.region[i] = 1;
    for (uint32_t i = r_dpr1; i < r_tn1a; ++i) neurons.region[i] = 2;
    for (uint32_t i = r_tn1a; i < r_vpr6; ++i) neurons.region[i] = 3;
    for (uint32_t i = r_vpr6; i < r_vpr9; ++i) neurons.region[i] = 4;
    for (uint32_t i = r_vpr9; i < r_mn_pulse; ++i) neurons.region[i] = 5;
    for (uint32_t i = r_mn_pulse; i < r_mn_sine; ++i) neurons.region[i] = 6;
    for (uint32_t i = r_mn_sine; i < total; ++i) neurons.region[i] = 7;

    // Assign cell types
    // P1: bursting (persistent arousal state)
    for (uint32_t i = r_p1; i < r_pip10; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kBursting);
    // pIP10: regular spiking descending
    for (uint32_t i = r_pip10; i < r_dpr1; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kPN_excitatory);
    // dPR1: pulse pattern generator (params overridden below)
    for (uint32_t i = r_dpr1; i < r_tn1a; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kPN_excitatory);
    // TN1A: regular spiking (sine pathway)
    for (uint32_t i = r_tn1a; i < r_vpr6; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kPN_excitatory);
    // vPR6: regular spiking (timing control)
    for (uint32_t i = r_vpr6; i < r_vpr9; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    // vPR9: fast spiking inhibitory (feedback)
    for (uint32_t i = r_vpr9; i < r_mn_pulse; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kFastSpiking);
    // Motor neurons: regular spiking
    for (uint32_t i = r_mn_pulse; i < total; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);

    CellTypeManager types;
    types.AssignFromTypes(neurons);

    // Override Izhikevich parameters for pulse timing.
    //
    // Key insight: all neurons except dPR1 must fire tonically.
    // Default cell type params (d=6, a=0.02) cause adaptation-driven
    // bursting at ~60ms that overrides dPR1's pulse rhythm. Making
    // relay neurons tonic (a=0.10, d=2, c=-65) lets dPR1 alone
    // control the pulse timing.
    //
    // dPR1 uses intrinsic bursting dynamics (a=0.10, c=-55, d=10):
    // strong adaptation (d=10) creates burst-pause behavior, fast
    // recovery (a=0.10, 1/a = 10ms) yields ~35-45ms IPI matching
    // D. melanogaster courtship pulse song.
    // Ref: Izhikevich 2003 IEEE (IB/chattering neuron class).
    auto set_tonic = [&](uint32_t start, uint32_t end) {
      for (uint32_t i = start; i < end; ++i) {
        types.pa[i] = 0.10f;
        types.pb[i] = 0.2f;
        types.pc[i] = -65.0f;
        types.pd[i] = 2.0f;
      }
    };
    set_tonic(r_p1, r_dpr1);         // P1, pIP10: relay tonic drive
    set_tonic(r_tn1a, r_mn_pulse);   // TN1A, vPR6, vPR9: tonic relay
    set_tonic(r_mn_pulse, total);    // MN_pulse, MN_sine: faithful readout

    // dPR1: intrinsic burster (pulse pattern generator)
    for (uint32_t i = r_dpr1; i < r_tn1a; ++i) {
      types.pa[i] = 0.10f;   // fast recovery (1/a ~10ms)
      types.pb[i] = 0.2f;
      types.pc[i] = -55.0f;  // moderate reset (burst mode)
      types.pd[i] = 10.0f;   // strong adaptation
    }

    // Build synapses via COO
    std::vector<uint32_t> pre_list, post_list;
    std::vector<float> weight_list;
    std::vector<uint8_t> nt_list;
    std::mt19937 rng(seed);

    auto connect = [&](uint32_t src_start, uint32_t src_end,
                       uint32_t dst_start, uint32_t dst_end,
                       float density, float weight, uint8_t nt) {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (uint32_t s = src_start; s < src_end; ++s) {
        for (uint32_t d = dst_start; d < dst_end; ++d) {
          if (dist(rng) < density) {
            pre_list.push_back(s);
            post_list.push_back(d);
            weight_list.push_back(weight);
            nt_list.push_back(nt);
          }
        }
      }
    };

    uint8_t nt_ach = 0;   // excitatory
    uint8_t nt_gaba = 1;  // inhibitory

    // P1 -> pIP10 (command, persistent drive)
    connect(r_p1, r_pip10, r_pip10, r_dpr1, 0.40f, 6.0f, nt_ach);

    // pIP10 -> dPR1 (descending, moderate drive)
    connect(r_pip10, r_dpr1, r_dpr1, r_tn1a, 0.25f, 3.0f, nt_ach);

    // pIP10 -> TN1A (sine pathway)
    connect(r_pip10, r_dpr1, r_tn1a, r_vpr6, 0.25f, 4.0f, nt_ach);

    // pIP10 -> vPR6 (timing control)
    connect(r_pip10, r_dpr1, r_vpr6, r_vpr9, 0.20f, 3.0f, nt_ach);

    // dPR1 -> MN_pulse (pulse motor drive, strong to produce burst)
    connect(r_dpr1, r_tn1a, r_mn_pulse, r_mn_sine, 0.50f, 8.0f, nt_ach);

    // TN1A -> MN_sine (sine motor drive)
    connect(r_tn1a, r_vpr6, r_mn_sine, total, 0.50f, 5.0f, nt_ach);

    // vPR6 -> dPR1 (timing modulation, weak)
    connect(r_vpr6, r_vpr9, r_dpr1, r_tn1a, 0.15f, 1.5f, nt_ach);

    // dPR1 -> vPR9 (excite inhibitory feedback)
    connect(r_dpr1, r_tn1a, r_vpr9, r_mn_pulse, 0.40f, 5.0f, nt_ach);

    // vPR9 -> dPR1 (light inhibitory feedback, sharpens pulse edges
    // without dominating timing which is set by dPR1 intrinsic dynamics)
    connect(r_vpr9, r_mn_pulse, r_dpr1, r_tn1a, 0.20f, -1.5f, nt_gaba);

    // vPR9 -> pIP10 (feedback to descending)
    connect(r_vpr9, r_mn_pulse, r_pip10, r_dpr1, 0.20f, -3.0f, nt_gaba);

    // vPR9 -> TN1A (inhibit sine during pulse)
    connect(r_vpr9, r_mn_pulse, r_tn1a, r_vpr6, 0.30f, -4.0f, nt_gaba);

    SynapseTable synapses;
    synapses.BuildFromCOO(total, pre_list, post_list, weight_list, nt_list);

    // SFA
    SpikeFrequencyAdaptation sfa;
    sfa.Init(total);

    // Rate monitor
    std::vector<std::string> region_names = {
        "P1", "pIP10", "dPR1", "TN1A", "vPR6", "vPR9", "MN_pulse", "MN_sine"};
    RateMonitor rate_mon;
    rate_mon.Init(neurons, region_names, dt_ms);

    // Background noise (very low - this is a motor circuit, not sensory)
    std::mt19937 bg_rng(seed + 1);
    std::normal_distribution<float> bg_noise(0.5f, 0.5f);

    int total_steps = static_cast<int>(total_duration_ms / dt_ms);

    // Pulse detection: track motor neuron activity in 5ms bins
    int bin_size_steps = static_cast<int>(pulse_bin_ms / dt_ms);
    std::vector<int> pulse_bins;
    int current_bin_spikes = 0;
    int bin_step_counter = 0;

    // Track individual pulse times for IPI computation
    std::vector<float> pulse_times;
    bool in_pulse = false;
    float last_pulse_time = -100.0f;
    const float pulse_refractory_ms = 15.0f;  // min gap between distinct pulses

    // Spike collector for post-hoc analysis
    SpikeCollector spikes;
    spikes.Init(total);

    Log(LogLevel::kInfo, "[courtship] %u neurons, %zu synapses",
        total, synapses.Size());

    for (int step = 0; step < total_steps; ++step) {
      float t = step * dt_ms;
      neurons.ClearExternalInput();

      // Low background noise
      for (size_t i = 0; i < neurons.n; ++i) {
        neurons.i_ext[i] = std::max(0.0f, bg_noise(bg_rng));
      }

      // P1 activation (female pheromone stimulus)
      if (t >= p1_onset_ms && t < p1_onset_ms + p1_duration_ms) {
        for (uint32_t i = r_p1; i < r_pip10; ++i) {
          neurons.i_ext[i] = p1_drive;
        }
      }

      // Step dynamics
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
      sfa.Update(neurons, dt_ms);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, t, types);
      rate_mon.RecordStep(neurons);
      spikes.Record(neurons.spiked.data(), neurons.n, t);

      // Count pulse motor neuron spikes
      int mn_pulse_spikes = 0;
      for (uint32_t i = r_mn_pulse; i < r_mn_sine; ++i)
        mn_pulse_spikes += neurons.spiked[i];
      result.motor_total_spikes += mn_pulse_spikes;

      // Bin-based pulse detection
      current_bin_spikes += mn_pulse_spikes;
      bin_step_counter++;
      if (bin_step_counter >= bin_size_steps) {
        if (current_bin_spikes >= pulse_threshold) {
          if (!in_pulse && (t - last_pulse_time) > pulse_refractory_ms) {
            // New pulse detected (with refractory period)
            pulse_times.push_back(t);
            last_pulse_time = t;
            in_pulse = true;
          }
        } else {
          in_pulse = false;
        }
        pulse_bins.push_back(current_bin_spikes);
        current_bin_spikes = 0;
        bin_step_counter = 0;
      }
    }

    // Compute IPI from pulse times
    result.n_pulses = static_cast<int>(pulse_times.size());
    if (pulse_times.size() >= 2) {
      std::vector<float> ipis;
      for (size_t i = 1; i < pulse_times.size(); ++i) {
        ipis.push_back(pulse_times[i] - pulse_times[i - 1]);
      }
      float sum = std::accumulate(ipis.begin(), ipis.end(), 0.0f);
      result.mean_ipi_ms = sum / ipis.size();

      // CV of IPI
      float mean = result.mean_ipi_ms;
      float sq_sum = 0.0f;
      for (float ipi : ipis) sq_sum += (ipi - mean) * (ipi - mean);
      float std_dev = std::sqrt(sq_sum / ipis.size());
      result.ipi_cv = (mean > 0.0f) ? std_dev / mean : 0.0f;
    }

    // Pulse bout duration
    if (!pulse_times.empty()) {
      result.pulse_bout_duration_ms =
          pulse_times.back() - pulse_times.front() + 5.0f;  // +1 bin
    }

    // Population rates
    auto rates = rate_mon.ComputeRates();
    for (const auto& rr : rates) {
      if (rr.name == "P1") result.p1_rate_hz = rr.rate_hz;
      if (rr.name == "pIP10") result.pip10_rate_hz = rr.rate_hz;
      if (rr.name == "dPR1") result.dpr1_rate_hz = rr.rate_hz;
      if (rr.name == "vPR9") result.vpr9_rate_hz = rr.rate_hz;
      if (rr.name == "MN_pulse") result.motor_peak_rate_hz = rr.rate_hz;
      if (rr.name == "MN_sine") result.sine_rate_hz = rr.rate_hz;
    }

    // Spike train analysis: dPR1 burst detection
    auto dpr1_train = spikes.GetPopulationTrain(r_dpr1, r_tn1a);
    auto dpr1_bursts = DetectBursts(dpr1_train, 15.0f, 2);  // 15ms max ISI, min 2 spikes
    auto dpr1_bstats = ComputeBurstStats(dpr1_bursts, total_duration_ms);

    // dPR1 ISI statistics
    auto dpr1_isi = ComputeISI(dpr1_train);

    // Population synchrony of MN_pulse (high Fano = synchronized bursts)
    auto mn_trains = spikes.GetTrains(r_mn_pulse, r_mn_sine);
    float mn_fano = PopulationFanoFactor(mn_trains, 2.0f, total_duration_ms);

    // Log burst analysis
    Log(LogLevel::kInfo, "[courtship] dPR1 bursts=%d, IBI=%.1fms (CV=%.2f), "
        "spk/burst=%.1f, ISI: mean=%.1fms CV=%.2f, MN Fano=%.1f",
        dpr1_bstats.n_bursts, dpr1_bstats.mean_ibi_ms, dpr1_bstats.ibi_cv,
        dpr1_bstats.mean_spikes_per_burst,
        dpr1_isi.mean_ms, dpr1_isi.cv, mn_fano);

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[courtship] done in %.2fs: pulses=%d, IPI=%.1fms (CV=%.2f), "
        "dPR1=%.1fHz, vPR9=%.1fHz, MN=%.1fHz",
        result.elapsed_seconds,
        result.n_pulses, result.mean_ipi_ms, result.ipi_cv,
        result.dpr1_rate_hz, result.vpr9_rate_hz, result.motor_peak_rate_hz);

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_COURTSHIP_EXPERIMENT_H_
