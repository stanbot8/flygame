#ifndef FWMC_PREY_CAPTURE_EXPERIMENT_H_
#define FWMC_PREY_CAPTURE_EXPERIMENT_H_

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
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/spike_frequency_adaptation.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Moving dot stimulus for zebrafish prey capture.
// Models a paramecium-sized object (~50-200 um) moving across the
// visual field at 1-4 mm/s. Zebrafish larvae (6-7 dpf) execute
// J-turn prey capture maneuvers for small, slow-moving targets.
//
// Bianco & Engert 2011 Neuron: prey detection requires 2-10 deg spot.
// Mearns et al. 2020 Current Biology: size selectivity in tectum.
struct PreyStimulus {
  float onset_ms = 200.0f;          // stimulus appearance time
  float duration_ms = 500.0f;       // how long the dot moves
  float dot_size_deg = 5.0f;        // angular size of prey (2-10 deg)
  float speed_deg_s = 60.0f;        // angular velocity (30-120 deg/s)
  float direction_deg = 90.0f;      // movement direction

  // Position (deg) at time t_ms relative to onset
  float PositionDeg(float t_ms) const {
    float rel_t = t_ms - onset_ms;
    if (rel_t < 0.0f || rel_t > duration_ms) return -1.0f;
    return (rel_t / 1000.0f) * speed_deg_s;
  }

  bool IsActive(float t_ms) const {
    return t_ms >= onset_ms && t_ms < onset_ms + duration_ms;
  }
};

// Results from a zebrafish prey capture experiment.
struct PreyCaptureResult {
  // Tectal response
  float tectal_onset_ms = -1.0f;      // first tectal evoked response
  float tectal_peak_rate_hz = 0.0f;   // peak PVN firing rate
  float sin_peak_rate_hz = 0.0f;      // peak SIN (inhibitory) rate

  // Direction selectivity: tectal response ratio for preferred vs null direction.
  // Niell & Smith 2005: zebrafish tectal DS neurons have DSI ~0.3-0.6.
  float direction_selectivity = 0.0f;

  // Size selectivity: response to small (prey) vs large (non-prey) stimulus.
  // Del Bene et al. 2010: SINs suppress responses to large stimuli.
  // Measured as ratio: small_response / large_response (>1 = size selective).
  float size_selectivity = 0.0f;

  // Motor output
  float hindbrain_onset_ms = -1.0f;   // first hindbrain motor response
  float capture_latency_ms = -1.0f;   // time from stimulus to motor output
  int hindbrain_spikes = 0;           // total motor spikes

  // Eye convergence proxy: pretectal activation correlates with
  // binocular convergence during prey capture (Bianco et al. 2011).
  float pretectal_rate_hz = 0.0f;

  // Population rates
  float retina_rate_hz = 0.0f;
  float pvn_rate_hz = 0.0f;
  float cerebellum_rate_hz = 0.0f;

  // Circuit size
  uint32_t total_neurons = 0;
  size_t total_synapses = 0;

  // Timing
  double elapsed_seconds = 0.0;

  // Validation: prey capture J-turn latency ~150-400ms from detection
  // (Bianco & Engert 2011, Patterson et al. 2013).
  bool responded() const { return tectal_onset_ms > 0.0f; }

  bool capture_plausible() const {
    if (!responded()) return false;
    return capture_latency_ms > 50.0f && capture_latency_ms < 500.0f;
  }
};

// Self-contained zebrafish larval prey capture experiment.
//
// Circuit (Robles et al. 2011, Del Bene et al. 2010, Bianco et al. 2011):
//   Retina (800 RGCs) -> Tectum PVN (2000) + SIN (500) -> Pretectum (300)
//                                                       -> Hindbrain (200)
//                         Cerebellum (1200) -> Hindbrain
//   Total: 5000 neurons
//
// The optic tectum receives retinotopic input and processes visual features:
//   - PVNs: excitatory periventricular neurons, encode prey position/motion
//   - SINs: GABAergic superficial interneurons, mediate size selectivity
//     (Del Bene et al. 2010: SINs inhibit PVN responses to large objects)
//   - Pretectum: drives eye convergence for binocular prey tracking
//   - Hindbrain reticulospinal: generates J-turn swim bouts
//   - Cerebellum: motor adaptation and timing
//
// References:
//   Bianco & Engert 2011 Neuron 72:735 - prey capture behavior
//   Del Bene et al. 2010 Science 330:669 - SIN size selectivity
//   Mearns et al. 2020 Curr Biol 30:4370 - size filtering in tectum
//   Niell & Smith 2005 Neuron 47:757 - direction selectivity
//   Robles et al. 2011 Curr Biol 21:402 - tectal laminar organization
//   Orger et al. 2008 J Neurophysiol 100:1975 - reticulospinal swim commands
struct PreyCaptureExperiment {
  // Circuit sizes (from zebrafish_optic_tectum.brain)
  uint32_t n_retina = 800;
  uint32_t n_pvn = 2000;
  uint32_t n_sin = 500;
  uint32_t n_pretectum = 300;
  uint32_t n_hindbrain = 200;
  uint32_t n_cerebellum = 1200;

  // Stimulus
  PreyStimulus stimulus;

  // Timing
  float dt_ms = 0.1f;
  float total_duration_ms = 1000.0f;

  // Response window (relative to stimulus onset)
  float response_window_ms = 600.0f;

  // Background drive
  float background_mean = 4.0f;
  float background_std = 1.5f;

  PreyCaptureResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    PreyCaptureResult result;

    // Build circuit using ParametricGenerator
    BrainSpec spec;
    spec.name = "zebrafish_optic_tectum";
    spec.species = Species::kZebrafish;
    spec.seed = seed;
    spec.global_weight_mean = 0.6f;
    spec.global_weight_std = 0.2f;
    spec.background_current_mean = background_mean;
    spec.background_current_std = background_std;

    // Region 0: Tectum PVN (main excitatory)
    {
      RegionSpec r;
      r.name = "tectum_pvn";
      r.n_neurons = n_pvn;
      r.internal_density = 0.03f;
      r.default_nt = kGlut;
      r.cell_types = {{CellType::kTectalPVN, 0.70f},
                       {CellType::kPV_Basket, 0.15f},
                       {CellType::kSST_Martinotti, 0.15f}};
      r.nt_distribution = {{kGlut, 0.70f}, {kGABA, 0.30f}};
      spec.regions.push_back(r);
    }

    // Region 1: Tectum SIN (inhibitory gain control)
    {
      RegionSpec r;
      r.name = "tectum_sin";
      r.n_neurons = n_sin;
      r.internal_density = 0.05f;
      r.default_nt = kGABA;
      r.cell_types = {{CellType::kTectalSIN, 0.60f},
                       {CellType::kPV_Basket, 0.40f}};
      r.nt_distribution = {{kGABA, 1.0f}};
      spec.regions.push_back(r);
    }

    // Region 2: Retina (RGC input layer)
    {
      RegionSpec r;
      r.name = "retina";
      r.n_neurons = n_retina;
      r.internal_density = 0.02f;
      r.default_nt = kGlut;
      r.cell_types = {{CellType::kRGC, 1.0f}};
      r.nt_distribution = {{kGlut, 1.0f}};
      spec.regions.push_back(r);
    }

    // Region 3: Pretectum
    {
      RegionSpec r;
      r.name = "pretectum";
      r.n_neurons = n_pretectum;
      r.internal_density = 0.06f;
      r.default_nt = kGlut;
      r.cell_types = {{CellType::kTectalPVN, 0.80f},
                       {CellType::kTectalSIN, 0.20f}};
      r.nt_distribution = {{kGlut, 0.80f}, {kGABA, 0.20f}};
      spec.regions.push_back(r);
    }

    // Region 4: Hindbrain (reticulospinal motor output)
    {
      RegionSpec r;
      r.name = "hindbrain";
      r.n_neurons = n_hindbrain;
      r.internal_density = 0.08f;
      r.default_nt = kGlut;
      r.cell_types = {{CellType::kReticulospinal, 0.80f},
                       {CellType::kTectalSIN, 0.20f}};
      r.nt_distribution = {{kGlut, 0.80f}, {kGABA, 0.20f}};
      spec.regions.push_back(r);
    }

    // Region 5: Cerebellum
    {
      RegionSpec r;
      r.name = "cerebellum";
      r.n_neurons = n_cerebellum;
      r.internal_density = 0.02f;
      r.default_nt = kGlut;
      r.cell_types = {{CellType::kGranuleCell, 0.90f},
                       {CellType::kPV_Basket, 0.10f}};
      r.nt_distribution = {{kGlut, 0.90f}, {kGABA, 0.10f}};
      spec.regions.push_back(r);
    }

    // Projections
    // Retina -> tectum PVN (retinotectal, topographic)
    spec.projections.push_back({"retina", "tectum_pvn", 0.04f, kGlut, 1.8f, 0.4f});
    // Retina -> tectum SIN (feedforward inhibition)
    spec.projections.push_back({"retina", "tectum_sin", 0.03f, kGlut, 1.5f, 0.3f});
    // SIN -> PVN (GABAergic gain control, critical for size selectivity)
    spec.projections.push_back({"tectum_sin", "tectum_pvn", 0.04f, kGABA, 1.8f, 0.4f});
    // Tectum PVN -> pretectum (tectal output)
    spec.projections.push_back({"tectum_pvn", "pretectum", 0.03f, kGlut, 2.0f, 0.3f});
    // Pretectum -> hindbrain (motor command)
    spec.projections.push_back({"pretectum", "hindbrain", 0.05f, kGlut, 3.0f, 0.4f});
    // Tectum PVN -> hindbrain (direct tectofugal, Helmbrecht et al. 2018)
    spec.projections.push_back({"tectum_pvn", "hindbrain", 0.02f, kGlut, 2.0f, 0.3f});
    // Tectum PVN -> cerebellum (efference copy)
    spec.projections.push_back({"tectum_pvn", "cerebellum", 0.01f, kGlut, 0.6f, 0.2f});
    // Cerebellum -> hindbrain (motor adaptation)
    spec.projections.push_back({"cerebellum", "hindbrain", 0.02f, kGlut, 0.8f, 0.2f});

    // Generate circuit
    NeuronArray neurons;
    SynapseTable synapses;
    CellTypeManager types;
    ParametricGenerator gen;
    gen.Generate(spec, neurons, synapses, types);

    result.total_neurons = static_cast<uint32_t>(neurons.n);
    result.total_synapses = synapses.Size();

    // Region ranges
    uint32_t r_pvn_start = gen.region_ranges[0].start;
    uint32_t r_pvn_end = gen.region_ranges[0].end;
    uint32_t r_sin_start = gen.region_ranges[1].start;  (void)r_sin_start;
    uint32_t r_sin_end = gen.region_ranges[1].end;      (void)r_sin_end;
    uint32_t r_ret_start = gen.region_ranges[2].start;
    uint32_t r_ret_end = gen.region_ranges[2].end;
    uint32_t r_pre_start = gen.region_ranges[3].start;  (void)r_pre_start;
    uint32_t r_pre_end = gen.region_ranges[3].end;      (void)r_pre_end;
    uint32_t r_hb_start = gen.region_ranges[4].start;
    uint32_t r_hb_end = gen.region_ranges[4].end;
    uint32_t r_cb_start = gen.region_ranges[5].start;   (void)r_cb_start;
    uint32_t r_cb_end = gen.region_ranges[5].end;       (void)r_cb_end;

    // SFA
    SpikeFrequencyAdaptation sfa;
    sfa.Init(neurons.n);

    // Rate monitor
    std::vector<std::string> region_names = {
        "tectum_pvn", "tectum_sin", "retina", "pretectum", "hindbrain", "cerebellum"};
    RateMonitor rate_mon;
    rate_mon.Init(neurons, region_names, dt_ms);

    // Background noise
    std::mt19937 bg_rng(seed + 1);
    std::normal_distribution<float> bg_noise(background_mean, background_std);

    int total_steps = static_cast<int>(total_duration_ms / dt_ms);

    // Retinotopic stimulus: moving dot activates a sliding window of RGCs
    // Each RGC covers ~(180/n_retina) deg of visual field
    float deg_per_rgc = 180.0f / n_retina;

    Log(LogLevel::kInfo, "[prey_capture] %u neurons, %zu synapses, species=zebrafish",
        result.total_neurons, result.total_synapses);

    for (int step = 0; step < total_steps; ++step) {
      float t = step * dt_ms;
      neurons.ClearExternalInput();

      // Background activity (all non-retinal regions)
      for (size_t i = 0; i < neurons.n; ++i) {
        if (neurons.region[i] != 2) {  // not retina
          neurons.i_ext[i] = bg_noise(bg_rng);
        }
      }

      // Spontaneous retinal activity (low rate)
      {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (uint32_t i = r_ret_start; i < r_ret_end; ++i) {
          if (u(bg_rng) < 0.15f) {
            neurons.i_ext[i] = 2.0f;
          }
        }
      }

      // Moving dot stimulus: activate RGCs in a sliding retinotopic window
      if (stimulus.IsActive(t)) {
        float pos_deg = stimulus.PositionDeg(t);
        float half_size = stimulus.dot_size_deg / 2.0f;
        float low_deg = pos_deg - half_size;
        float high_deg = pos_deg + half_size;

        // Map to RGC indices
        int rgc_low = static_cast<int>(low_deg / deg_per_rgc);
        int rgc_high = static_cast<int>(high_deg / deg_per_rgc);
        rgc_low = std::max(0, rgc_low);
        rgc_high = std::min(static_cast<int>(n_retina) - 1, rgc_high);

        for (int r = rgc_low; r <= rgc_high; ++r) {
          neurons.i_ext[r_ret_start + r] = 12.0f;  // strong drive
        }
      }

      // Step dynamics
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
      sfa.Update(neurons, dt_ms);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, t, types);
      rate_mon.RecordStep(neurons);

      // Detect evoked tectal response
      float rel_t = t - stimulus.onset_ms;
      if (rel_t >= 0.0f && rel_t < response_window_ms) {
        auto count_spikes = [&](uint32_t start, uint32_t end) -> int {
          int c = 0;
          for (uint32_t i = start; i < end; ++i) c += neurons.spiked[i];
          return c;
        };

        // Tectal onset: >=5 PVNs spike simultaneously (above background)
        if (result.tectal_onset_ms < 0.0f &&
            count_spikes(r_pvn_start, r_pvn_end) >= 5)
          result.tectal_onset_ms = rel_t;

        // Hindbrain onset: >=2 reticulospinal neurons fire simultaneously
        if (result.hindbrain_onset_ms < 0.0f &&
            count_spikes(r_hb_start, r_hb_end) >= 2) {
          result.hindbrain_onset_ms = rel_t;
          result.capture_latency_ms = rel_t;
        }

        // Count hindbrain spikes
        result.hindbrain_spikes += count_spikes(r_hb_start, r_hb_end);
      }
    }

    // Compute population rates
    auto rates = rate_mon.ComputeRates();
    for (const auto& rr : rates) {
      if (rr.name == "retina") result.retina_rate_hz = rr.rate_hz;
      if (rr.name == "tectum_pvn") {
        result.pvn_rate_hz = rr.rate_hz;
        result.tectal_peak_rate_hz = rr.rate_hz;
      }
      if (rr.name == "tectum_sin") result.sin_peak_rate_hz = rr.rate_hz;
      if (rr.name == "pretectum") result.pretectal_rate_hz = rr.rate_hz;
      if (rr.name == "cerebellum") result.cerebellum_rate_hz = rr.rate_hz;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[prey_capture] done in %.2fs: tectal=%.1fms, motor=%.1fms, "
        "PVN=%.1fHz, SIN=%.1fHz, hb_spikes=%d",
        result.elapsed_seconds,
        result.tectal_onset_ms, result.capture_latency_ms,
        result.tectal_peak_rate_hz, result.sin_peak_rate_hz,
        result.hindbrain_spikes);

    return result;
  }
};

}  // namespace mechabrain

#endif  // FWMC_PREY_CAPTURE_EXPERIMENT_H_
