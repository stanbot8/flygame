#ifndef FWMC_VALIDATION_H_
#define FWMC_VALIDATION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include "bridge/bridge_channel.h"
#include "core/neuron_array.h"

namespace mechabrain {

// ============================================================
// Validation framework for comparing simulated neural activity
// against experimentally recorded data.
//
// Provides multiple comparison metrics at different scales:
//   - Single neuron:  spike timing, voltage trace correlation
//   - Population:     firing rate, synchrony, spatial patterns
//   - Temporal:       sliding window statistics, stationarity
//
// References:
//   Victor & Purpura 1996, doi:10.1152/jn.1996.76.2.1310
//     (spike train distance metrics)
//   Schreiber et al. 2003, doi:10.1016/S0925-2312(03)00390-1
//     (correlation-based similarity)
//   Kreuz et al. 2007, doi:10.1016/j.jneumeth.2007.06.019
//     (ISI-distance, SPIKE-distance)
// ============================================================

// A recorded spike train: sorted list of spike times (ms) for one neuron
struct SpikeTrain {
  uint32_t neuron_idx;
  std::vector<float> times_ms;
};

// Comparison result for a single neuron
struct NeuronValidation {
  uint32_t neuron_idx;
  float spike_rate_sim;       // simulated firing rate (Hz)
  float spike_rate_rec;       // recorded firing rate (Hz)
  float rate_error;           // |sim - rec| / max(sim, rec)
  float correlation;          // Pearson correlation of binned spike counts
  float van_rossum_dist;      // van Rossum distance (lower = more similar)
  float precision;            // fraction of sim spikes matched in recording
  float recall;               // fraction of rec spikes matched in simulation
  float f1_score;             // harmonic mean of precision and recall
};

// Population-level validation summary
struct PopulationValidation {
  float mean_rate_error;
  float mean_correlation;
  float mean_f1;
  float population_sync_sim;   // population synchrony index (sim)
  float population_sync_rec;   // population synchrony index (recorded)
  float spatial_correlation;   // correlation of spatial firing patterns
  size_t n_neurons;
  size_t n_well_matched;       // neurons with correlation > 0.7
  size_t n_poorly_matched;     // neurons with correlation < 0.3
};

// Sliding window statistics for temporal analysis
struct WindowStats {
  float time_ms;
  float spike_count_sim;
  float spike_count_rec;
  float correlation;
  float rate_ratio;  // sim_rate / rec_rate
};

// Main validation engine
struct ValidationEngine {
  float spike_match_window_ms = 2.0f;  // tolerance for spike time matching
  float bin_width_ms = 10.0f;          // bin width for rate correlation
  float van_rossum_tau_ms = 10.0f;     // time constant for van Rossum distance
  float analysis_window_ms = 100.0f;   // sliding window width

  // Compute binned spike counts from a spike train
  static std::vector<float> BinSpikes(const std::vector<float>& times,
                                       float bin_width, float duration) {
    size_t n_bins = static_cast<size_t>(std::ceil(duration / bin_width));
    std::vector<float> bins(n_bins, 0.0f);
    for (float t : times) {
      size_t bin = static_cast<size_t>(t / bin_width);
      if (bin < n_bins) bins[bin] += 1.0f;
    }
    return bins;
  }

  // Pearson correlation between two vectors
  static float PearsonCorrelation(const std::vector<float>& a,
                                   const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0f;

    float sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (size_t i = 0; i < n; ++i) {
      sum_a += a[i];
      sum_b += b[i];
      sum_ab += a[i] * b[i];
      sum_a2 += a[i] * a[i];
      sum_b2 += b[i] * b[i];
    }

    float nf = static_cast<float>(n);
    float denom = std::sqrt((nf * sum_a2 - sum_a * sum_a) *
                             (nf * sum_b2 - sum_b * sum_b));
    if (denom < 1e-12f) return 0.0f;
    return (nf * sum_ab - sum_a * sum_b) / denom;
  }

  // Van Rossum distance: convolve spike trains with exponential kernel,
  // then compute L2 distance. Lower = more similar.
  // (van Rossum 2001, doi:10.1162/089976601300014321)
  static float VanRossumDistance(const std::vector<float>& train_a,
                                  const std::vector<float>& train_b,
                                  float tau_ms, float duration_ms,
                                  float dt_ms = 0.5f) {
    size_t n_bins = static_cast<size_t>(duration_ms / dt_ms);
    std::vector<float> conv_a(n_bins, 0.0f), conv_b(n_bins, 0.0f);

    // Convolve with exponential kernel
    float decay = std::exp(-dt_ms / tau_ms);
    for (float t : train_a) {
      size_t start = static_cast<size_t>(t / dt_ms);
      float val = 1.0f;
      for (size_t i = start; i < n_bins && val > 0.001f; ++i) {
        conv_a[i] += val;
        val *= decay;
      }
    }
    for (float t : train_b) {
      size_t start = static_cast<size_t>(t / dt_ms);
      float val = 1.0f;
      for (size_t i = start; i < n_bins && val > 0.001f; ++i) {
        conv_b[i] += val;
        val *= decay;
      }
    }

    // L2 distance
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n_bins; ++i) {
      float diff = conv_a[i] - conv_b[i];
      sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq * dt_ms / tau_ms);
  }

  // Spike matching: find pairs of spikes within the tolerance window.
  // Returns (precision, recall, f1).
  struct MatchResult {
    float precision;
    float recall;
    float f1;
    size_t matched;
  };

  static MatchResult MatchSpikes(const std::vector<float>& sim_spikes,
                                  const std::vector<float>& rec_spikes,
                                  float window_ms) {
    if (sim_spikes.empty() && rec_spikes.empty()) {
      return {1.0f, 1.0f, 1.0f, 0};
    }
    if (sim_spikes.empty() || rec_spikes.empty()) {
      return {0.0f, 0.0f, 0.0f, 0};
    }

    // Greedy nearest-neighbor matching
    std::vector<bool> rec_used(rec_spikes.size(), false);
    size_t matched = 0;

    for (float st : sim_spikes) {
      float best_dist = window_ms + 1.0f;
      size_t best_idx = 0;
      for (size_t j = 0; j < rec_spikes.size(); ++j) {
        if (rec_used[j]) continue;
        float dist = std::abs(st - rec_spikes[j]);
        if (dist < best_dist) {
          best_dist = dist;
          best_idx = j;
        }
      }
      if (best_dist <= window_ms) {
        rec_used[best_idx] = true;
        matched++;
      }
    }

    float precision = static_cast<float>(matched) / static_cast<float>(sim_spikes.size());
    float recall = static_cast<float>(matched) / static_cast<float>(rec_spikes.size());
    float f1 = (precision + recall > 0)
        ? 2.0f * precision * recall / (precision + recall) : 0.0f;
    return {precision, recall, f1, matched};
  }

  // Validate a single neuron: compare simulated spike train to recording
  NeuronValidation ValidateNeuron(const SpikeTrain& sim,
                                   const SpikeTrain& rec,
                                   float duration_ms) const {
    NeuronValidation v;
    v.neuron_idx = sim.neuron_idx;

    // Firing rates
    float dur_sec = duration_ms / 1000.0f;
    v.spike_rate_sim = static_cast<float>(sim.times_ms.size()) / dur_sec;
    v.spike_rate_rec = static_cast<float>(rec.times_ms.size()) / dur_sec;
    float max_rate = std::max(v.spike_rate_sim, v.spike_rate_rec);
    v.rate_error = (max_rate > 0) ? std::abs(v.spike_rate_sim - v.spike_rate_rec) / max_rate : 0.0f;

    // Binned correlation
    auto bins_sim = BinSpikes(sim.times_ms, bin_width_ms, duration_ms);
    auto bins_rec = BinSpikes(rec.times_ms, bin_width_ms, duration_ms);
    v.correlation = PearsonCorrelation(bins_sim, bins_rec);

    // Van Rossum distance
    v.van_rossum_dist = VanRossumDistance(sim.times_ms, rec.times_ms,
                                          van_rossum_tau_ms, duration_ms);

    // Spike matching
    auto match = MatchSpikes(sim.times_ms, rec.times_ms, spike_match_window_ms);
    v.precision = match.precision;
    v.recall = match.recall;
    v.f1_score = match.f1;

    return v;
  }

  // Validate an entire population
  PopulationValidation ValidatePopulation(
      const std::vector<SpikeTrain>& sim_trains,
      const std::vector<SpikeTrain>& rec_trains,
      float duration_ms) const {

    PopulationValidation pv{};
    pv.n_neurons = sim_trains.size();

    if (sim_trains.empty()) return pv;

    // Build index for O(1) lookup of recorded trains by neuron_idx.
    // Eliminates O(N*M) linear scan that dominated runtime.
    std::unordered_map<uint32_t, size_t> rec_index;
    rec_index.reserve(rec_trains.size());
    for (size_t r = 0; r < rec_trains.size(); ++r) {
      rec_index[rec_trains[r].neuron_idx] = r;
    }

    float sum_rate_err = 0, sum_corr = 0, sum_f1 = 0;
    std::vector<float> sim_rates, rec_rates;

    for (size_t i = 0; i < sim_trains.size(); ++i) {
      // Find matching recorded train via hash map
      auto it = rec_index.find(sim_trains[i].neuron_idx);
      if (it == rec_index.end()) continue;
      const SpikeTrain* rec = &rec_trains[it->second];

      auto nv = ValidateNeuron(sim_trains[i], *rec, duration_ms);
      sum_rate_err += nv.rate_error;
      sum_corr += nv.correlation;
      sum_f1 += nv.f1_score;
      sim_rates.push_back(nv.spike_rate_sim);
      rec_rates.push_back(nv.spike_rate_rec);

      if (nv.correlation > 0.7f) pv.n_well_matched++;
      if (nv.correlation < 0.3f) pv.n_poorly_matched++;
    }

    float nf = static_cast<float>(pv.n_neurons);
    pv.mean_rate_error = sum_rate_err / nf;
    pv.mean_correlation = sum_corr / nf;
    pv.mean_f1 = sum_f1 / nf;
    pv.spatial_correlation = PearsonCorrelation(sim_rates, rec_rates);

    // Population synchrony: fraction of neurons spiking in each bin
    pv.population_sync_sim = ComputeSynchrony(sim_trains, duration_ms);
    pv.population_sync_rec = ComputeSynchrony(rec_trains, duration_ms);

    return pv;
  }

  // Compute population synchrony index (coefficient of variation of
  // population spike count across time bins). Higher = more synchronous.
  float ComputeSynchrony(const std::vector<SpikeTrain>& trains,
                          float duration_ms) const {
    size_t n_bins = static_cast<size_t>(std::ceil(duration_ms / bin_width_ms));
    std::vector<float> pop_count(n_bins, 0.0f);

    for (const auto& train : trains) {
      for (float t : train.times_ms) {
        size_t bin = static_cast<size_t>(t / bin_width_ms);
        if (bin < n_bins) pop_count[bin] += 1.0f;
      }
    }

    // CV = std / mean
    float sum = 0, sum_sq = 0;
    for (float c : pop_count) {
      sum += c;
      sum_sq += c * c;
    }
    float nf = static_cast<float>(n_bins);
    float mean = sum / nf;
    if (mean < 1e-6f) return 0.0f;
    float var = sum_sq / nf - mean * mean;
    return std::sqrt(std::max(0.0f, var)) / mean;
  }

  // Sliding window analysis: compute metrics over time
  std::vector<WindowStats> SlidingWindowAnalysis(
      const SpikeTrain& sim, const SpikeTrain& rec,
      float duration_ms) const {

    std::vector<WindowStats> windows;
    float step = analysis_window_ms / 2.0f;  // 50% overlap

    for (float t = 0; t + analysis_window_ms <= duration_ms; t += step) {
      WindowStats ws;
      ws.time_ms = t + analysis_window_ms / 2.0f;

      // Count spikes in window
      ws.spike_count_sim = 0;
      ws.spike_count_rec = 0;
      std::vector<float> sim_in_window, rec_in_window;

      for (float st : sim.times_ms) {
        if (st >= t && st < t + analysis_window_ms) {
          ws.spike_count_sim += 1.0f;
          sim_in_window.push_back(st - t);
        }
      }
      for (float rt : rec.times_ms) {
        if (rt >= t && rt < t + analysis_window_ms) {
          ws.spike_count_rec += 1.0f;
          rec_in_window.push_back(rt - t);
        }
      }

      // Local correlation
      auto bins_s = BinSpikes(sim_in_window, bin_width_ms, analysis_window_ms);
      auto bins_r = BinSpikes(rec_in_window, bin_width_ms, analysis_window_ms);
      ws.correlation = PearsonCorrelation(bins_s, bins_r);

      ws.rate_ratio = (ws.spike_count_rec > 0)
          ? ws.spike_count_sim / ws.spike_count_rec : 0.0f;

      windows.push_back(ws);
    }
    return windows;
  }

  // Extract spike trains from a NeuronArray simulation run.
  // Call this after each step, accumulating spike times.
  static void RecordSpikes(const NeuronArray& neurons, float sim_time_ms,
                           std::vector<SpikeTrain>& trains) {
    // Initialize trains if needed
    if (trains.size() != neurons.n) {
      trains.resize(neurons.n);
      for (size_t i = 0; i < neurons.n; ++i) {
        trains[i].neuron_idx = static_cast<uint32_t>(i);
      }
    }
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.spiked[i]) {
        trains[i].times_ms.push_back(sim_time_ms);
      }
    }
  }

  // Convert BioReadings stream to spike trains using threshold crossing
  static void RecordBioSpikes(const std::vector<BioReading>& readings,
                               float sim_time_ms, float threshold,
                               std::vector<SpikeTrain>& trains) {
    for (const auto& r : readings) {
      if (r.spike_prob >= threshold) {
        // Ensure train exists
        while (trains.size() <= r.neuron_idx) {
          SpikeTrain st;
          st.neuron_idx = static_cast<uint32_t>(trains.size());
          trains.push_back(st);
        }
        trains[r.neuron_idx].times_ms.push_back(sim_time_ms);
      }
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_VALIDATION_H_
