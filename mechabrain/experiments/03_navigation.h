#ifndef FWMC_NAVIGATION_EXPERIMENT_H_
#define FWMC_NAVIGATION_EXPERIMENT_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/cell_types.h"
#include "optimizer/core/optimizer.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/spike_frequency_adaptation.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Heading direction stimulus: rotating bar projected onto the visual field.
// Models a single cue (bright bar) at a specific azimuthal position.
// The bar position maps to a subset of E-PG neurons via their
// preferred heading (Seelig & Jayaraman 2015 Nature).
struct HeadingStimulus {
  float bar_angle_deg = 0.0f;     // current azimuthal position (0-360)
  float bar_width_deg = 30.5f;    // angular width of the visual cue (auto-tuned, robust)
  float intensity = 33.1f;        // pA drive to matching ring neurons (auto-tuned, robust)

  // Returns drive for a neuron whose preferred heading is pref_deg.
  // Gaussian tuning curve centered on bar position, sigma = bar_width/2.
  float DriveForHeading(float pref_deg) const {
    float diff = pref_deg - bar_angle_deg;
    // Wrap to [-180, 180]
    while (diff > 180.0f) diff -= 360.0f;
    while (diff < -180.0f) diff += 360.0f;
    float sigma = bar_width_deg / 2.0f;
    return intensity * std::exp(-(diff * diff) / (2.0f * sigma * sigma));
  }
};

// Angular velocity input: simulates head rotation detected by
// haltere/optic flow. Asymmetric drive to left vs right P-EN neurons
// (Green et al. 2017 Nature, Turner-Evans et al. 2017 eLife).
struct AngularVelocityInput {
  float omega_deg_s = 0.0f;       // angular velocity (deg/s, positive = CW)
  float gain = 7.2f;              // pA per (deg/s) scaling (auto-tuned, robust)

  // Drive to left P-EN (positive omega = CW = more left P-EN activity)
  float LeftDrive() const {
    return (omega_deg_s > 0.0f) ? gain * omega_deg_s / 100.0f : 0.0f;
  }
  // Drive to right P-EN
  float RightDrive() const {
    return (omega_deg_s < 0.0f) ? gain * (-omega_deg_s) / 100.0f : 0.0f;
  }
};

// Trial protocol segment for the navigation experiment.
struct NavSegment {
  float duration_ms = 1000.0f;
  float bar_angle_deg = 0.0f;     // visual bar position
  float angular_vel_deg_s = 0.0f; // head angular velocity
  bool darkness = false;           // if true, no visual cue (test persistence)
};

// Results from a CX navigation experiment.
struct NavigationResult {
  // Bump quality metrics
  float bump_amplitude = 0.0f;       // peak-to-trough E-PG firing ratio
  float bump_width_deg = 0.0f;       // FWHM of the heading bump
  float bump_stability = 0.0f;       // 1.0 = perfectly stable, 0.0 = no bump

  // Heading accuracy
  float heading_error_deg = 0.0f;    // mean |decoded - actual| heading
  float heading_error_dark_deg = 0.0f; // same, during darkness (persistence)

  // Angular velocity integration
  float rotation_tracking_r2 = 0.0f; // R^2 of bump position vs actual heading

  // Path integration (cumulative)
  float path_integration_error_deg = 0.0f; // drift after closed-loop walk

  // Population rates
  float epg_mean_rate_hz = 0.0f;
  float pen_mean_rate_hz = 0.0f;
  float pfl_mean_rate_hz = 0.0f;
  float delta7_mean_rate_hz = 0.0f;

  // PFL steering output: left vs right asymmetry
  float pfl_left_mean_hz = 0.0f;
  float pfl_right_mean_hz = 0.0f;

  // Timing
  double elapsed_seconds = 0.0;
  int total_segments = 0;

  // Validation: E-PG bump should persist in darkness for ~30s
  // (Seelig & Jayaraman 2015), heading error < 30 deg in light,
  // < 60 deg in short darkness (< 5s).
  bool bump_present() const { return bump_amplitude > 1.2f; }

  bool heading_plausible() const {
    return heading_error_deg < 30.0f && bump_present();
  }

  bool dark_persistence_ok() const {
    return heading_error_dark_deg < 60.0f;
  }
};

// Self-contained central complex navigation experiment.
//
// Circuit (Hulse et al. 2021 eLife, connectome-derived):
//   Ring neurons (16) -> E-PG (16) <-> P-EN (16) -> PFL (32) -> motor
//                              ^           |
//                        Delta7 (16) lateral inhibition
//
// Ring neurons: R1-R4d, convey visual landmark position (bar) to EB
// E-PG: compass neurons, maintain heading bump via ring attractor
// P-EN: angular velocity neurons, shift the bump left/right
// Delta7: inhibitory interneurons, enforce single bump (winner-take-all)
// PFL: steering neurons, translate heading to left/right motor commands
//
// The E-PG + P-EN + Delta7 network forms a ring attractor that:
//   1. Anchors to visual landmarks (via ring neuron input)
//   2. Updates with angular velocity (via P-EN asymmetric drive)
//   3. Persists in darkness (attractor dynamics maintain the bump)
//
// References:
//   Seelig & Jayaraman 2015 Nature 521:186 - heading bump in E-PG
//   Green et al. 2017 Nature 546:101 - P-EN angular velocity integration
//   Kim et al. 2017 Science 356:849 - ring attractor dynamics
//   Turner-Evans et al. 2017 eLife 6:e23496 - P-EN/E-PG connectivity
//   Hulse et al. 2021 eLife 10:e62576 - full CX connectome
struct NavigationExperiment {
  // Circuit sizes.
  // 16 E-PG columns map to 16 heading directions (22.5 deg each),
  // matching the 18-glomerulus PB with wraparound (Wolff et al. 2015).
  // We use 16 for clean power-of-2 ring topology.
  uint32_t n_ring = 16;       // ring neurons (visual input to EB)
  uint32_t n_epg = 16;        // E-PG compass neurons
  uint32_t n_pen = 16;        // P-EN angular velocity neurons (8L + 8R)
  uint32_t n_delta7 = 16;     // Delta7 inhibitory interneurons
  uint32_t n_pfl = 32;        // PFL steering output (16L + 16R)
  uint32_t n_fc = 20;         // fan-shaped body columnar neurons

  // Connectivity
  // Ring -> E-PG: topographic, each ring neuron excites its matched E-PG
  float ring_epg_weight = 34.9f;    // pA, strong visual anchor (auto-tuned, robust)
  // E-PG -> P-EN: local excitation (same column)
  float epg_pen_weight = 14.6f;     // pA (auto-tuned, robust)
  // P-EN -> E-PG: offset by +/-1 column (bump rotation mechanism)
  float pen_epg_weight = 2.1f;      // pA, shifted connectivity (auto-tuned, robust)
  // E-PG -> E-PG: local excitation (bump self-sustaining)
  float epg_epg_weight = 4.2f;      // pA, crucial for attractor stability (auto-tuned, robust)
  // Delta7 -> E-PG: global inhibition (minus local exemption)
  float delta7_epg_weight = -5.5f;  // pA, inhibitory (auto-tuned, robust)
  // E-PG -> Delta7: excitatory drive to inhibitory pool
  float epg_delta7_weight = 1.3f;   // pA (auto-tuned, robust)
  // E-PG -> PFL: heading to steering (topographic, L/R split)
  float epg_pfl_weight = 3.5f;      // pA
  // PFL -> FC: steering to FB integration
  float pfl_fc_weight = 2.0f;       // pA

  // Stimulus
  HeadingStimulus heading_stim;
  AngularVelocityInput angular_vel;

  // Protocol: default is bar rotation + darkness test
  std::vector<NavSegment> protocol;

  // Timing
  float dt_ms = 0.1f;
  float pre_stim_ms = 500.0f;    // let network settle

  // Background tonic drive to keep E-PG population weakly active.
  // Must be strong enough to sustain bump without visual input.
  float tonic_drive = 2.0f;      // pA to all E-PG (auto-tuned, robust)

  NavigationResult Run(uint32_t seed = 42) {
    auto t_start = std::chrono::steady_clock::now();
    NavigationResult result;

    // Default protocol if none specified.
    // Modeled on Seelig & Jayaraman 2015 experiments: bar rotation,
    // darkness test, and re-anchoring. Avoids instant jumps (which
    // are unrealistic and exploit the bump's inertia).
    if (protocol.empty()) {
      // Phase 1: Bar at 0 deg, stationary (establish bump) - 2s
      protocol.push_back({2000.0f, 0.0f, 0.0f, false});
      // Phase 2: Slow CW rotation 0->90 deg at 45 deg/s - 2s
      protocol.push_back({2000.0f, 0.0f, 45.0f, false});
      // Phase 3: Slow CCW rotation 90->0 deg at -45 deg/s - 2s
      protocol.push_back({2000.0f, 90.0f, -45.0f, false});
      // Phase 4: Darkness (test bump persistence) - 3s
      protocol.push_back({3000.0f, 0.0f, 0.0f, true});
      // Phase 5: Bar returns at 0 deg (test re-anchoring from dark) - 1s
      protocol.push_back({1000.0f, 0.0f, 0.0f, false});
    }
    result.total_segments = static_cast<int>(protocol.size());

    // Total neurons
    uint32_t total = n_ring + n_epg + n_pen + n_delta7 + n_pfl + n_fc;

    // Region boundaries
    uint32_t r_ring = 0;
    uint32_t r_epg = n_ring;
    uint32_t r_pen = r_epg + n_epg;
    uint32_t r_delta7 = r_pen + n_pen;
    uint32_t r_pfl = r_delta7 + n_delta7;
    uint32_t r_fc = r_pfl + n_pfl;

    // Build circuit
    NeuronArray neurons;
    neurons.Resize(total);
    neurons.tau_syn.resize(total, 3.0f);
    // Slower synaptic time constant for E-PG (sustains bump between spikes)
    for (uint32_t i = r_epg; i < r_pen; ++i) neurons.tau_syn[i] = 8.0f;

    // Assign regions
    for (uint32_t i = r_ring; i < r_epg; ++i) neurons.region[i] = 0;
    for (uint32_t i = r_epg; i < r_pen; ++i) neurons.region[i] = 1;
    for (uint32_t i = r_pen; i < r_delta7; ++i) neurons.region[i] = 2;
    for (uint32_t i = r_delta7; i < r_pfl; ++i) neurons.region[i] = 3;
    for (uint32_t i = r_pfl; i < r_fc; ++i) neurons.region[i] = 4;
    for (uint32_t i = r_fc; i < total; ++i) neurons.region[i] = 5;

    // Assign cell types
    for (uint32_t i = r_ring; i < r_epg; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kRingNeuron);
    for (uint32_t i = r_epg; i < r_pen; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kEPG);
    for (uint32_t i = r_pen; i < r_delta7; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kPEN);
    for (uint32_t i = r_delta7; i < r_pfl; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kDelta7);
    for (uint32_t i = r_pfl; i < r_fc; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kPFL);
    for (uint32_t i = r_fc; i < total; ++i)
      neurons.type[i] = static_cast<uint8_t>(CellType::kFC);

    CellTypeManager types;
    types.AssignFromTypes(neurons);

    // Build synapses via COO lists
    std::vector<uint32_t> pre_list, post_list;
    std::vector<float> weight_list;
    std::vector<uint8_t> nt_list;
    std::mt19937 rng(seed);

    uint8_t nt_ach = 0;
    uint8_t nt_gaba = 1;

    auto add_syn = [&](uint32_t pre, uint32_t post, float w, uint8_t nt) {
      pre_list.push_back(pre);
      post_list.push_back(post);
      weight_list.push_back(w);
      nt_list.push_back(nt);
    };

    // Ring -> E-PG: topographic 1-to-1 (visual landmark anchoring)
    for (uint32_t i = 0; i < n_ring; ++i) {
      add_syn(r_ring + i, r_epg + (i % n_epg), ring_epg_weight, nt_ach);
    }

    // E-PG -> E-PG: local excitation (self + nearest 2 neighbors on ring).
    // Self-connection is crucial for bump maintenance in small networks.
    for (uint32_t i = 0; i < n_epg; ++i) {
      uint32_t src = r_epg + i;
      // Self
      add_syn(src, src, epg_epg_weight, nt_ach);
      // +/-1 neighbors
      add_syn(src, r_epg + ((i + n_epg - 1) % n_epg), epg_epg_weight * 0.8f, nt_ach);
      add_syn(src, r_epg + ((i + 1) % n_epg), epg_epg_weight * 0.8f, nt_ach);
      // +/-2 neighbors (weaker)
      add_syn(src, r_epg + ((i + n_epg - 2) % n_epg), epg_epg_weight * 0.4f, nt_ach);
      add_syn(src, r_epg + ((i + 2) % n_epg), epg_epg_weight * 0.4f, nt_ach);
    }

    // E-PG -> P-EN: same-column excitation
    for (uint32_t i = 0; i < n_epg; ++i) {
      add_syn(r_epg + i, r_pen + (i % n_pen), epg_pen_weight, nt_ach);
    }

    // P-EN -> E-PG: offset connectivity (bump rotation).
    // Left P-EN (first half) shift bump CW (+1),
    // Right P-EN (second half) shift bump CCW (-1).
    // This is the key mechanism for angular velocity integration
    // (Turner-Evans et al. 2017).
    for (uint32_t i = 0; i < n_pen; ++i) {
      int offset = (i < n_pen / 2) ? +1 : -1;  // L=CW, R=CCW
      uint32_t target_col = static_cast<uint32_t>(
          (static_cast<int>(i % n_epg) + offset + static_cast<int>(n_epg))
          % static_cast<int>(n_epg));
      add_syn(r_pen + i, r_epg + target_col, pen_epg_weight, nt_ach);
    }

    // E-PG -> Delta7: all E-PG excite all Delta7 (global pool)
    for (uint32_t i = 0; i < n_epg; ++i) {
      for (uint32_t j = 0; j < n_delta7; ++j) {
        add_syn(r_epg + i, r_delta7 + j,
                epg_delta7_weight / static_cast<float>(n_epg), nt_ach);
      }
    }

    // Delta7 -> E-PG: global inhibition except local column.
    // Each Delta7 neuron j inhibits all E-PG except column j.
    // This enforces a single activity bump (ring attractor).
    for (uint32_t j = 0; j < n_delta7; ++j) {
      for (uint32_t i = 0; i < n_epg; ++i) {
        if (i != j) {
          add_syn(r_delta7 + j, r_epg + i,
                  delta7_epg_weight / static_cast<float>(n_delta7), nt_gaba);
        }
      }
    }

    // E-PG -> PFL: heading to steering (topographic, split L/R)
    for (uint32_t i = 0; i < n_epg; ++i) {
      uint32_t pfl_l = r_pfl + (i % (n_pfl / 2));
      add_syn(r_epg + i, pfl_l, epg_pfl_weight, nt_ach);
      uint32_t pfl_r = r_pfl + n_pfl / 2 + (i % (n_pfl / 2));
      add_syn(r_epg + i, pfl_r, epg_pfl_weight, nt_ach);
    }

    // PFL -> FC: steering to fan-shaped body
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < n_pfl; ++i) {
      for (uint32_t j = 0; j < n_fc; ++j) {
        if (dist(rng) < 0.15f) {
          add_syn(r_pfl + i, r_fc + j, pfl_fc_weight, nt_ach);
        }
      }
    }

    SynapseTable synapses;
    synapses.BuildFromCOO(total, pre_list, post_list, weight_list, nt_list);

    // SFA for adaptation
    SpikeFrequencyAdaptation sfa;
    sfa.Init(total);

    // Preferred heading for each E-PG neuron (uniform around ring)
    std::vector<float> epg_pref_heading(n_epg);
    for (uint32_t i = 0; i < n_epg; ++i) {
      epg_pref_heading[i] = i * 360.0f / n_epg;
    }

    // Rate monitoring
    std::vector<std::string> region_names = {
        "ring", "EPG", "PEN", "Delta7", "PFL", "FC"};
    RateMonitor rate_mon;
    rate_mon.Init(neurons, region_names, dt_ms);

    // Run simulation
    float sim_time = 0.0f;

    // Tracking for heading decode
    std::vector<float> decoded_headings;
    std::vector<float> actual_headings;
    std::vector<float> decoded_dark;
    std::vector<float> actual_dark;
    float current_heading = 0.0f;  // actual heading (integrated)

    Log(LogLevel::kInfo, "[navigation] %u neurons, %zu synapses, %d segments",
        total, synapses.Size(), result.total_segments);

    // Pre-stimulus settling
    int settle_steps = static_cast<int>(pre_stim_ms / dt_ms);
    for (int step = 0; step < settle_steps; ++step) {
      neurons.ClearExternalInput();
      for (uint32_t i = r_epg; i < r_pen; ++i) {
        neurons.i_ext[i] = tonic_drive;
      }
      neurons.DecaySynapticInput(dt_ms, 3.0f);
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
      sfa.Update(neurons, dt_ms);
      IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
      rate_mon.RecordStep(neurons);
      sim_time += dt_ms;
    }

    // Per-E-PG spike accumulator for heading decode (sliding window)
    std::vector<float> epg_spike_accum(n_epg, 0.0f);
    float accum_decay = std::exp(-dt_ms / 50.0f);  // 50ms window

    // Run protocol segments
    for (const auto& seg : protocol) {
      int n_steps = static_cast<int>(seg.duration_ms / dt_ms);

      // Update actual heading from angular velocity
      float seg_heading_start = current_heading;

      for (int step = 0; step < n_steps; ++step) {
        float seg_t = step * dt_ms;
        neurons.ClearExternalInput();

        // Tonic drive to E-PG
        for (uint32_t i = r_epg; i < r_pen; ++i) {
          neurons.i_ext[i] = tonic_drive;
        }

        // Update actual heading
        current_heading = seg_heading_start + seg.angular_vel_deg_s * (seg_t / 1000.0f);
        while (current_heading >= 360.0f) current_heading -= 360.0f;
        while (current_heading < 0.0f) current_heading += 360.0f;

        // Visual landmark input (ring neurons)
        if (!seg.darkness) {
          float bar_pos = seg.bar_angle_deg +
                          seg.angular_vel_deg_s * (seg_t / 1000.0f);
          while (bar_pos >= 360.0f) bar_pos -= 360.0f;
          while (bar_pos < 0.0f) bar_pos += 360.0f;

          heading_stim.bar_angle_deg = bar_pos;
          for (uint32_t i = 0; i < n_ring; ++i) {
            float pref = i * 360.0f / n_ring;
            neurons.i_ext[r_ring + i] += heading_stim.DriveForHeading(pref);
          }
        }

        // Angular velocity input to P-EN
        angular_vel.omega_deg_s = seg.angular_vel_deg_s;
        float left_drive = angular_vel.LeftDrive();
        float right_drive = angular_vel.RightDrive();
        for (uint32_t i = 0; i < n_pen / 2; ++i) {
          neurons.i_ext[r_pen + i] += left_drive;
        }
        for (uint32_t i = n_pen / 2; i < n_pen; ++i) {
          neurons.i_ext[r_pen + i] += right_drive;
        }

        // Step dynamics
        neurons.DecaySynapticInput(dt_ms, 3.0f);
        synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
        sfa.Update(neurons, dt_ms);
        IzhikevichStepHeterogeneousFast(neurons, dt_ms, sim_time, types);
        rate_mon.RecordStep(neurons);
        sim_time += dt_ms;

        // Update E-PG spike accumulator
        for (uint32_t i = 0; i < n_epg; ++i) {
          epg_spike_accum[i] *= accum_decay;
          if (neurons.spiked[r_epg + i]) epg_spike_accum[i] += 1.0f;
        }

        // Decode heading from E-PG population every 10ms
        if (step % 100 == 0) {
          float decoded = DecodeHeadingFromAccum(epg_spike_accum, n_epg, epg_pref_heading);
          if (seg.darkness) {
            decoded_dark.push_back(decoded);
            actual_dark.push_back(current_heading);
          } else {
            // In light: the bump should anchor to the visual landmark.
            // The "actual" heading is the bar position, not the
            // angular-velocity-integrated heading.
            float bar_pos = seg.bar_angle_deg +
                            seg.angular_vel_deg_s * (step * dt_ms / 1000.0f);
            while (bar_pos >= 360.0f) bar_pos -= 360.0f;
            while (bar_pos < 0.0f) bar_pos += 360.0f;
            decoded_headings.push_back(decoded);
            actual_headings.push_back(bar_pos);
          }
        }
      }
    }

    // Compute results

    // Heading error (light)
    if (!decoded_headings.empty()) {
      float err_sum = 0.0f;
      for (size_t i = 0; i < decoded_headings.size(); ++i) {
        float diff = decoded_headings[i] - actual_headings[i];
        while (diff > 180.0f) diff -= 360.0f;
        while (diff < -180.0f) diff += 360.0f;
        err_sum += std::abs(diff);
      }
      result.heading_error_deg = err_sum / decoded_headings.size();
    }

    // Heading error (darkness)
    if (!decoded_dark.empty()) {
      float err_sum = 0.0f;
      for (size_t i = 0; i < decoded_dark.size(); ++i) {
        float diff = decoded_dark[i] - actual_dark[i];
        while (diff > 180.0f) diff -= 360.0f;
        while (diff < -180.0f) diff += 360.0f;
        err_sum += std::abs(diff);
      }
      result.heading_error_dark_deg = err_sum / decoded_dark.size();
    }

    // Collect population rates from RateMonitor
    auto rates = rate_mon.ComputeRates();
    for (const auto& rr : rates) {
      if (rr.name == "EPG") result.epg_mean_rate_hz = rr.rate_hz;
      if (rr.name == "PEN") result.pen_mean_rate_hz = rr.rate_hz;
      if (rr.name == "Delta7") result.delta7_mean_rate_hz = rr.rate_hz;
      if (rr.name == "PFL") result.pfl_mean_rate_hz = rr.rate_hz;
    }

    // Bump amplitude: estimate from heading consistency.
    // If decoded headings cluster tightly, a bump is present.
    // Use mean resultant length of decoded headings as proxy.
    if (!decoded_headings.empty()) {
      float s = 0.0f, c = 0.0f;
      for (float h : decoded_headings) {
        float rad = h * kPi / 180.0f;
        s += std::sin(rad);
        c += std::cos(rad);
      }
      float mrl = std::sqrt(s * s + c * c) / decoded_headings.size();
      // mrl near 1.0 = tight cluster (strong bump), near 0 = diffuse
      result.bump_amplitude = mrl * 3.0f;  // scale so >1.5 = bump present
    }
    result.bump_width_deg = 90.0f;  // nominal for 16-column ring

    // Bump stability: heading consistency in light periods
    result.bump_stability = (result.bump_amplitude > 1.5f) ? 1.0f : 0.0f;

    // Rotation tracking R^2
    if (decoded_headings.size() > 2) {
      result.rotation_tracking_r2 = ComputeCircularR2(
          decoded_headings, actual_headings);
    }

    // PFL left vs right: approximate from total PFL rate
    // (without per-neuron rates, use equal split as baseline)
    result.pfl_left_mean_hz = result.pfl_mean_rate_hz;
    result.pfl_right_mean_hz = result.pfl_mean_rate_hz;

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    Log(LogLevel::kInfo,
        "[navigation] done in %.2fs: bump_amp=%.1f, heading_err=%.1fdeg, "
        "dark_err=%.1fdeg, R2=%.3f",
        result.elapsed_seconds, result.bump_amplitude,
        result.heading_error_deg, result.heading_error_dark_deg,
        result.rotation_tracking_r2);

    return result;
  }

 private:
  // Decode heading from E-PG spike accumulator (population vector average).
  // Each E-PG neuron votes for its preferred heading, weighted by
  // recent spike count. Returns decoded angle in [0, 360).
  static float DecodeHeadingFromAccum(const std::vector<float>& accum,
                                      uint32_t n_epg,
                                      const std::vector<float>& pref_headings) {
    float sin_sum = 0.0f, cos_sum = 0.0f;
    for (uint32_t i = 0; i < n_epg; ++i) {
      float rad = pref_headings[i] * kPi / 180.0f;
      sin_sum += accum[i] * std::sin(rad);
      cos_sum += accum[i] * std::cos(rad);
    }
    float decoded_rad = std::atan2(sin_sum, cos_sum);
    float decoded_deg = decoded_rad * 180.0f / kPi;
    if (decoded_deg < 0.0f) decoded_deg += 360.0f;
    return decoded_deg;
  }

  // Circular R^2: measures how well decoded headings track actual headings.
  static float ComputeCircularR2(const std::vector<float>& decoded,
                                 const std::vector<float>& actual) {
    if (decoded.size() != actual.size() || decoded.empty()) return 0.0f;

    // Mean resultant length of angle differences
    float sin_sum = 0.0f, cos_sum = 0.0f;
    for (size_t i = 0; i < decoded.size(); ++i) {
      float diff_rad = (decoded[i] - actual[i]) * kPi / 180.0f;
      sin_sum += std::sin(diff_rad);
      cos_sum += std::cos(diff_rad);
    }
    float n = static_cast<float>(decoded.size());
    float mean_resultant = std::sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / n;

    // R^2 analog for circular data: 1 - circular_variance
    // mean_resultant = 1 means perfect agreement, 0 = random
    return mean_resultant;
  }
};

// Navigation parameter optimizer using the general-purpose ExperimentOptimizer.
// Evaluates each parameter combination across multiple seeds for robustness
// against chaotic spiking dynamics.
struct NavigationOptimizer {
  int n_iterations = 200;
  uint32_t seed = 42;
  int n_seeds = 3;  // multi-seed robustness (min fitness across seeds)

  struct Result {
    NavigationResult nav;
    std::vector<float> params;
    float fitness = 0.0f;
  };

  Result Run() {
    fwmc::ExperimentOptimizer opt;
    opt.log_fn = [](const std::string& msg) {
        Log(LogLevel::kInfo, "%s", msg.c_str());
    };
    opt.axes = {
      {"ring_epg",   5.0f,  40.0f},
      {"epg_epg",    2.0f,  20.0f},
      {"delta7_epg", -6.0f, -0.5f},
      {"tonic",      2.0f,  15.0f},
      {"intensity",  5.0f,  40.0f},
      {"pen_epg",    2.0f,  15.0f},
      {"epg_pen",    2.0f,  15.0f},
      {"epg_d7",     0.5f,   5.0f},
      {"bar_width",  15.0f, 60.0f},
      {"ang_gain",   1.0f,  15.0f},
    };
    opt.n_iterations = n_iterations;
    opt.n_seeds = n_seeds;
    opt.base_seed = seed;

    NavigationResult best_nav;

    auto best = opt.Run([&](const std::vector<float>& params, uint32_t s) -> float {
      NavigationExperiment exp;
      exp.ring_epg_weight          = params[0];
      exp.epg_epg_weight           = params[1];
      exp.delta7_epg_weight        = params[2];
      exp.tonic_drive              = params[3];
      exp.heading_stim.intensity   = params[4];
      exp.pen_epg_weight           = params[5];
      exp.epg_pen_weight           = params[6];
      exp.epg_delta7_weight        = params[7];
      exp.heading_stim.bar_width_deg = params[8];
      exp.angular_vel.gain         = params[9];

      auto r = exp.Run(s);

      float fitness = 0.0f;
      fitness -= r.heading_error_deg * 2.0f;
      fitness -= r.heading_error_dark_deg * 1.0f;
      fitness += r.rotation_tracking_r2 * 100.0f;
      fitness += std::min(r.bump_amplitude, 3.0f) * 30.0f;
      if (r.bump_present()) fitness += 50.0f;
      if (r.heading_error_deg < 30.0f) fitness += 100.0f;
      if (r.heading_error_dark_deg < 60.0f) fitness += 50.0f;

      // Save the nav result from the base seed for reporting
      if (s == seed) best_nav = r;

      return fitness;
    });

    Log(LogLevel::kInfo,
        "[nav-opt] BEST (robust %d seeds): err=%.1f dark=%.1f R2=%.3f bump=%.1f",
        n_seeds, best_nav.heading_error_deg, best_nav.heading_error_dark_deg,
        best_nav.rotation_tracking_r2, best_nav.bump_amplitude);

    return {best_nav, best.params, best.fitness};
  }
};

}  // namespace mechabrain

#endif  // FWMC_NAVIGATION_EXPERIMENT_H_
