#ifndef FWMC_VISUAL_ESCAPE_EXPERIMENT_H_
#define FWMC_VISUAL_ESCAPE_EXPERIMENT_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "core/experiment_optimizer.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/spike_frequency_adaptation.h"
#include "core/stdp.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Looming stimulus: expanding disc projected onto the visual field.
// Models an approaching predator (bird, mantis) with known l/v ratio.
// Drosophila escape latency depends on l/v (Fotowat & Bhatt 2015).
struct LoomingStimulus {
  float l_mm = 10.0f;           // half-size of approaching object (mm)
  float v_mm_s = 500.0f;        // approach velocity (mm/s)
  float start_angle_deg = 2.0f; // initial angular subtense
  float max_angle_deg = 120.0f; // stimulus covers full visual field

  // Angular subtense (deg) at time t_ms from start.
  // theta(t) = 2 * atan(l / (v * (t_collision - t)))
  // where t_collision accounts for the starting distance.
  float AngleDeg(float t_ms) const {
    float t_collision_ms = CollisionTimeMs();
    float remaining = t_collision_ms - t_ms;
    if (remaining <= 0.01f) return max_angle_deg;
    float theta_rad = 2.0f * std::atan(l_mm / (v_mm_s * (remaining / 1000.0f)));
    float theta_deg = theta_rad * 180.0f / kPi;
    return std::min(theta_deg, max_angle_deg);
  }

  // Angular velocity (deg/s) at time t_ms.
  // Looming-sensitive neurons (LPLC2) respond to expansion rate.
  float AngularVelocityDegS(float t_ms) const {
    float dt = 0.5f;  // numerical derivative step
    float a0 = AngleDeg(t_ms - dt);
    float a1 = AngleDeg(t_ms + dt);
    return (a1 - a0) / (2.0f * dt) * 1000.0f;  // deg/s
  }

  // Time from stimulus onset to collision.
  // Object starts at a distance where it subtends start_angle_deg,
  // approaches at v_mm_s. Much longer than l/v.
  float CollisionTimeMs() const {
    float start_rad = start_angle_deg * kPi / 180.0f;
    float d_mm = l_mm / std::tan(start_rad / 2.0f);  // starting distance
    return (d_mm / v_mm_s) * 1000.0f;
  }
};

// Results from a visual escape experiment.
struct VisualEscapeResult {
  // GF (giant fiber) spike timing
  float gf_first_spike_ms = -1.0f;     // time of first GF spike
  float escape_latency_ms = -1.0f;     // GF spike relative to collision
  float angle_at_escape_deg = 0.0f;    // angular size when GF fires

  // LPLC2 population response
  float lplc2_peak_rate_hz = 0.0f;     // peak firing rate of LPLC2 population
  float lplc2_onset_ms = -1.0f;        // when LPLC2 rate exceeds threshold

  // LC population response (lobula columnar neurons)
  float lc_mean_rate_hz = 0.0f;

  // Motor neuron output
  float motor_peak_ms = -1.0f;         // peak motor neuron activity
  int motor_total_spikes = 0;

  // Stimulus parameters used
  float l_v_ratio = 0.0f;              // l/v in ms
  float collision_time_ms = 0.0f;

  // Timing
  double elapsed_seconds = 0.0;

  // Validation: Drosophila escape ~50-100ms before collision
  // for l/v = 20-40ms (von Reyn et al. 2017)
  bool escaped() const { return gf_first_spike_ms > 0.0f; }

  bool timing_plausible() const {
    if (!escaped()) return false;
    // GF should fire 30-150ms before collision
    return escape_latency_ms > 30.0f && escape_latency_ms < 200.0f;
  }
};

// Self-contained visual looming escape experiment.
//
// Circuit (von Reyn et al. 2017, Ache et al. 2019):
//   Photoreceptors (100) -> LC neurons (200) -> LPLC2 (20) -> GF (2) -> MN (20)
//                                                   ^
//                                          visual expansion detectors
//
// LC neurons: lobula columnar neurons, encode local motion/edge features
// LPLC2: looming-sensitive interneurons, integrate expanding-disc motion
// GF: giant fiber, command neuron for escape takeoff
// MN: leg/wing motor neurons for escape jump
//
// The LPLC2 neurons compute an angular-velocity-weighted signal that
// peaks at a specific time-to-collision, triggering the GF threshold.
// This produces the characteristic l/v-dependent escape timing.
struct VisualEscapeExperiment {
  // Circuit sizes (based on Drosophila visual system anatomy)
  uint32_t n_photo = 100;    // photoreceptor-like input layer
  uint32_t n_lc = 200;       // lobula columnar neurons (LC4, LC6, etc.)
  uint32_t n_lplc2 = 20;     // looming-sensitive integration neurons
  uint32_t n_gf = 2;         // giant fiber pair (left/right)
  uint32_t n_mn = 20;        // motor neurons (leg extensors, wing depressors)
  uint32_t n_inh = 30;       // local inhibitory interneurons (for gain control)

  // Connectivity densities
  float photo_lc_density = 0.15f;    // retinotopic, convergent
  float lc_lplc2_density = 0.25f;    // convergent integration
  float lplc2_gf_density = 0.80f;    // strong, reliable synapse
  float gf_mn_density = 0.60f;       // command -> motor (electrical + chemical)
  float inh_lc_density = 0.10f;      // lateral inhibition for contrast
  float lc_inh_density = 0.10f;      // feedback inhibition

  // Synaptic weights
  float w_photo_lc = 3.0f;    // pA, moderate drive
  float w_lc_lplc2 = 4.0f;    // pA, convergent summation
  float w_lplc2_gf = 12.0f;   // pA, strong (GF needs reliable threshold crossing)
  float w_gf_mn = 8.0f;       // pA, command synapse
  float w_inh = -3.0f;        // pA, inhibitory

  // Stimulus
  LoomingStimulus stimulus;

  // Timing
  float dt_ms = 0.1f;
  float pre_stim_ms = 200.0f;      // baseline before looming starts
  float post_collision_ms = 100.0f; // record after collision time

  // GF threshold: minimum current to consider GF activated
  float gf_threshold_rate_hz = 50.0f;

  VisualEscapeResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    VisualEscapeResult result;
    result.l_v_ratio = stimulus.l_mm / stimulus.v_mm_s * 1000.0f;
    result.collision_time_ms = stimulus.CollisionTimeMs();

    // Total neurons
    uint32_t total = n_photo + n_lc + n_lplc2 + n_gf + n_mn + n_inh;

    // Region boundaries
    uint32_t r_photo = 0;
    uint32_t r_lc = n_photo;
    uint32_t r_lplc2 = r_lc + n_lc;
    uint32_t r_gf = r_lplc2 + n_lplc2;
    uint32_t r_mn = r_gf + n_gf;
    uint32_t r_inh = r_mn + n_mn;

    // Build circuit
    NeuronArray neurons;
    neurons.Resize(total);
    if (neurons.tau_syn.empty()) neurons.tau_syn.resize(total, 3.0f);

    // Assign regions
    for (uint32_t i = r_photo; i < r_lc; ++i) neurons.region[i] = 0;
    for (uint32_t i = r_lc; i < r_lplc2; ++i) neurons.region[i] = 1;
    for (uint32_t i = r_lplc2; i < r_gf; ++i) neurons.region[i] = 2;
    for (uint32_t i = r_gf; i < r_mn; ++i) neurons.region[i] = 3;
    for (uint32_t i = r_mn; i < r_inh; ++i) neurons.region[i] = 4;
    for (uint32_t i = r_inh; i < total; ++i) neurons.region[i] = 5;

    // Assign cell types: photoreceptors=tonic, LC/LPLC2=regular spiking,
    // GF=fast spiking (low threshold), MN=regular, inh=fast spiking
    for (uint32_t i = r_photo; i < r_lc; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    for (uint32_t i = r_lc; i < r_lplc2; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    for (uint32_t i = r_lplc2; i < r_gf; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    for (uint32_t i = r_gf; i < r_mn; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kFastSpiking);
    for (uint32_t i = r_mn; i < r_inh; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    for (uint32_t i = r_inh; i < total; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kFastSpiking);

    CellTypeManager types;
    types.AssignFromTypes(neurons);

    // Build synapses via COO lists
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

    uint8_t nt_ach = 0;   // excitatory (acetylcholine)
    uint8_t nt_gaba = 1;  // inhibitory (GABA)

    connect(r_photo, r_lc, r_lc, r_lplc2, photo_lc_density, w_photo_lc, nt_ach);
    connect(r_lc, r_lplc2, r_lplc2, r_gf, lc_lplc2_density, w_lc_lplc2, nt_ach);
    connect(r_lplc2, r_gf, r_gf, r_mn, lplc2_gf_density, w_lplc2_gf, nt_ach);
    connect(r_gf, r_mn, r_mn, r_inh, gf_mn_density, w_gf_mn, nt_ach);
    connect(r_lc, r_lplc2, r_inh, total, lc_inh_density, w_photo_lc, nt_ach);
    connect(r_inh, total, r_lc, r_lplc2, inh_lc_density, w_inh, nt_gaba);

    SynapseTable synapses;
    synapses.BuildFromCOO(total, pre_list, post_list, weight_list, nt_list);

    // SFA for adaptation during sustained looming
    SpikeFrequencyAdaptation sfa;
    sfa.Init(total);

    // Simulation
    float total_ms = pre_stim_ms + stimulus.CollisionTimeMs() + post_collision_ms;
    int total_steps = static_cast<int>(total_ms / dt_ms);

    // Rate monitoring
    std::vector<std::string> region_names = {
        "photo", "LC", "LPLC2", "GF", "MN", "INH"};
    RateMonitor rate_mon;
    rate_mon.Init(neurons, region_names, dt_ms);

    // Track GF spikes
    bool gf_fired = false;

    Log(LogLevel::kInfo, "[visual_escape] %u neurons, %zu synapses, l/v=%.1fms",
        total, synapses.Size(), result.l_v_ratio);

    for (int step = 0; step < total_steps; ++step) {
      float t = step * dt_ms;
      float stim_t = t - pre_stim_ms;  // time relative to looming start

      // Clear external input
      neurons.ClearExternalInput();

      // Inject looming stimulus into photoreceptors
      if (stim_t >= 0.0f && stim_t <= stimulus.CollisionTimeMs()) {
        float angle = stimulus.AngleDeg(stim_t);
        float angular_vel = stimulus.AngularVelocityDegS(stim_t);

        // Photoreceptor drive proportional to angular velocity
        // (edge detection: rate of expansion matters more than absolute size)
        float drive = std::min(angular_vel / 500.0f, 1.0f) * 8.0f;  // max 8 pA

        // Spatial pattern: more photoreceptors active as stimulus grows
        float fraction_active = std::min(angle / 90.0f, 1.0f);
        uint32_t n_active = static_cast<uint32_t>(fraction_active * n_photo);

        for (uint32_t i = r_photo; i < r_photo + n_active; ++i) {
          neurons.i_ext[i] += drive;
        }
      }

      // Step dynamics
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
      sfa.Update(neurons, dt_ms);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, t, types);

      // Update rate monitor
      rate_mon.RecordStep(neurons);

      // Check GF firing
      if (!gf_fired) {
        for (uint32_t i = r_gf; i < r_mn; ++i) {
          if (neurons.spiked[i]) {
            gf_fired = true;
            result.gf_first_spike_ms = t;
            result.escape_latency_ms = stimulus.CollisionTimeMs() + pre_stim_ms - t;
            if (stim_t >= 0.0f) {
              result.angle_at_escape_deg = stimulus.AngleDeg(stim_t);
            }
            break;
          }
        }
      }

      // Count motor spikes
      for (uint32_t i = r_mn; i < r_inh; ++i) {
        result.motor_total_spikes += neurons.spiked[i];
      }
    }

    // Collect population rates
    auto rates = rate_mon.ComputeRates();
    for (const auto& rr : rates) {
      if (rr.name == "LPLC2") result.lplc2_peak_rate_hz = rr.rate_hz;
      if (rr.name == "LC") result.lc_mean_rate_hz = rr.rate_hz;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[visual_escape] done in %.2fs: escaped=%d, latency=%.1fms, angle=%.1fdeg",
        result.elapsed_seconds, result.escaped(),
        result.escape_latency_ms, result.angle_at_escape_deg);

    return result;
  }
};

// Visual escape parameter optimizer using ExperimentOptimizer.
// Targets biologically plausible escape timing: 30-150ms before collision,
// angular subtense 20-90 deg at escape (von Reyn et al. 2017).
struct VisualEscapeOptimizer {
  int n_iterations = 200;
  uint32_t seed = 42;
  int n_seeds = 3;

  struct Result {
    VisualEscapeResult escape;
    std::vector<float> params;
    float fitness = 0.0f;
  };

  Result Run() {
    ExperimentOptimizer opt;
    opt.axes = {
      {"w_photo_lc",  1.0f, 10.0f},
      {"w_lc_lplc2",  1.0f, 15.0f},
      {"w_lplc2_gf",  3.0f, 30.0f},
      {"w_gf_mn",     2.0f, 20.0f},
      {"w_inh",      -8.0f, -0.5f},
      {"photo_lc_d",  0.05f, 0.40f},
      {"lc_lplc2_d",  0.05f, 0.50f},
      {"lplc2_gf_d",  0.30f, 1.00f},
    };
    opt.n_iterations = n_iterations;
    opt.n_seeds = n_seeds;
    opt.base_seed = seed;

    VisualEscapeResult best_escape;

    auto best = opt.Run([&](const std::vector<float>& params, uint32_t s) -> float {
      VisualEscapeExperiment exp;
      exp.w_photo_lc       = params[0];
      exp.w_lc_lplc2       = params[1];
      exp.w_lplc2_gf       = params[2];
      exp.w_gf_mn          = params[3];
      exp.w_inh            = params[4];
      exp.photo_lc_density = params[5];
      exp.lc_lplc2_density = params[6];
      exp.lplc2_gf_density = params[7];

      auto r = exp.Run(s);

      float fitness = 0.0f;
      if (!r.escaped()) return -500.0f;  // must escape

      // Reward latency in [30, 150]ms range
      if (r.escape_latency_ms >= 30.0f && r.escape_latency_ms <= 150.0f) {
        fitness += 200.0f;
        // Bonus for being near center of range (60-80ms is ideal)
        float center_dist = std::abs(r.escape_latency_ms - 70.0f);
        fitness -= center_dist;
      } else if (r.escape_latency_ms < 30.0f) {
        fitness -= (30.0f - r.escape_latency_ms) * 5.0f;  // penalize too early
      } else {
        fitness -= (r.escape_latency_ms - 150.0f) * 3.0f;  // penalize too late
      }

      // Reward angle at escape in [20, 90] deg
      if (r.angle_at_escape_deg >= 20.0f && r.angle_at_escape_deg <= 90.0f) {
        fitness += 50.0f;
      }

      // Reward motor activity
      fitness += std::min(r.motor_total_spikes, 100) * 0.5f;

      if (s == seed) best_escape = r;
      return fitness;
    });

    Log(LogLevel::kInfo,
        "[escape-opt] BEST (robust %d seeds): latency=%.1fms angle=%.1fdeg motor=%d",
        n_seeds, best_escape.escape_latency_ms,
        best_escape.angle_at_escape_deg, best_escape.motor_total_spikes);

    return {best_escape, best.params, best.fitness};
  }
};

}  // namespace mechabrain

#endif  // FWMC_VISUAL_ESCAPE_EXPERIMENT_H_
