#ifndef PARABRAIN_SPIKE_ANALYSIS_H_
#define PARABRAIN_SPIKE_ANALYSIS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mechabrain {

// Spike train analysis utilities for validating neural circuit dynamics.
//
// Provides ISI statistics, burst detection, oscillation detection, and
// population synchrony measures. All functions operate on spike time
// vectors (in ms) and are designed for post-hoc analysis.

// Inter-spike interval statistics for a single neuron's spike train.
struct ISIStats {
  float mean_ms = 0.0f;
  float median_ms = 0.0f;
  float cv = 0.0f;           // coefficient of variation (std/mean)
  float cv2 = 0.0f;          // local CV (mean of |ISI[i+1]-ISI[i]| / (ISI[i+1]+ISI[i]))
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  size_t n_intervals = 0;

  // CV > 1: bursty. CV < 1: regular. CV ~1: Poisson-like.
  bool is_regular() const { return cv < 0.5f && n_intervals >= 3; }
  bool is_bursty()  const { return cv > 1.0f && n_intervals >= 3; }
  bool is_poisson() const { return cv > 0.8f && cv < 1.2f && n_intervals >= 10; }
};

// Compute ISI statistics from sorted spike times.
inline ISIStats ComputeISI(const std::vector<float>& spike_times) {
  ISIStats stats;
  if (spike_times.size() < 2) return stats;

  std::vector<float> isi;
  isi.reserve(spike_times.size() - 1);
  for (size_t i = 1; i < spike_times.size(); ++i)
    isi.push_back(spike_times[i] - spike_times[i - 1]);

  stats.n_intervals = isi.size();
  stats.min_ms = *std::min_element(isi.begin(), isi.end());
  stats.max_ms = *std::max_element(isi.begin(), isi.end());

  float sum = std::accumulate(isi.begin(), isi.end(), 0.0f);
  stats.mean_ms = sum / isi.size();

  auto sorted = isi;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  stats.median_ms = (n % 2 == 0)
      ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0f
      : sorted[n/2];

  // Standard CV
  float sq = 0.0f;
  for (float v : isi) sq += (v - stats.mean_ms) * (v - stats.mean_ms);
  float stddev = std::sqrt(sq / isi.size());
  stats.cv = (stats.mean_ms > 0.0f) ? stddev / stats.mean_ms : 0.0f;

  // Local CV2 (Holt et al. 1996): sensitive to local regularity,
  // robust to slow rate changes.
  if (isi.size() >= 2) {
    float cv2_sum = 0.0f;
    int cv2_n = 0;
    for (size_t i = 0; i + 1 < isi.size(); ++i) {
      float s = isi[i] + isi[i + 1];
      if (s > 0.0f) {
        cv2_sum += 2.0f * std::abs(isi[i + 1] - isi[i]) / s;
        cv2_n++;
      }
    }
    stats.cv2 = (cv2_n > 0) ? cv2_sum / cv2_n : 0.0f;
  }

  return stats;
}

// ISI histogram (bin counts) for visualization or distribution analysis.
struct ISIHistogram {
  std::vector<int> counts;
  float bin_width_ms = 1.0f;
  float max_isi_ms = 0.0f;

  // Get the mode (most common ISI bin center).
  float mode_ms() const {
    if (counts.empty()) return 0.0f;
    auto it = std::max_element(counts.begin(), counts.end());
    size_t idx = static_cast<size_t>(it - counts.begin());
    return (idx + 0.5f) * bin_width_ms;
  }
};

inline ISIHistogram ComputeISIHistogram(
    const std::vector<float>& spike_times,
    float bin_width_ms = 1.0f,
    float max_isi_ms = 200.0f) {
  ISIHistogram hist;
  hist.bin_width_ms = bin_width_ms;
  hist.max_isi_ms = max_isi_ms;
  int n_bins = static_cast<int>(max_isi_ms / bin_width_ms) + 1;
  hist.counts.assign(n_bins, 0);

  for (size_t i = 1; i < spike_times.size(); ++i) {
    float isi = spike_times[i] - spike_times[i - 1];
    int bin = static_cast<int>(isi / bin_width_ms);
    if (bin >= 0 && bin < n_bins) hist.counts[bin]++;
  }

  return hist;
}

// Detected burst: a cluster of spikes with short ISIs.
struct Burst {
  float onset_ms = 0.0f;
  float offset_ms = 0.0f;
  int n_spikes = 0;
  float mean_isi_ms = 0.0f;     // within-burst ISI
  float duration_ms() const { return offset_ms - onset_ms; }
  float intra_rate_hz() const {
    return duration_ms() > 0.0f ? (n_spikes - 1) * 1000.0f / duration_ms() : 0.0f;
  }
};

// Detect bursts using max-ISI criterion (Legendy & Salcman 1985).
// A burst starts when ISI < max_isi_ms and contains at least min_spikes.
inline std::vector<Burst> DetectBursts(
    const std::vector<float>& spike_times,
    float max_isi_ms = 10.0f,
    int min_spikes = 3) {

  std::vector<Burst> bursts;
  if (spike_times.size() < static_cast<size_t>(min_spikes)) return bursts;

  Burst current;
  current.onset_ms = spike_times[0];
  current.n_spikes = 1;
  float isi_sum = 0.0f;

  for (size_t i = 1; i < spike_times.size(); ++i) {
    float isi = spike_times[i] - spike_times[i - 1];
    if (isi <= max_isi_ms) {
      current.n_spikes++;
      isi_sum += isi;
      current.offset_ms = spike_times[i];
    } else {
      // End of burst candidate
      if (current.n_spikes >= min_spikes) {
        current.mean_isi_ms = isi_sum / (current.n_spikes - 1);
        bursts.push_back(current);
      }
      // Start new candidate
      current.onset_ms = spike_times[i];
      current.offset_ms = spike_times[i];
      current.n_spikes = 1;
      isi_sum = 0.0f;
    }
  }
  // Final burst
  if (current.n_spikes >= min_spikes) {
    current.mean_isi_ms = (current.n_spikes > 1) ? isi_sum / (current.n_spikes - 1) : 0.0f;
    bursts.push_back(current);
  }

  return bursts;
}

// Inter-burst interval statistics.
struct BurstStats {
  int n_bursts = 0;
  float mean_ibi_ms = 0.0f;    // inter-burst interval
  float mean_spikes_per_burst = 0.0f;
  float mean_duration_ms = 0.0f;
  float burst_fraction = 0.0f; // fraction of time in bursts
  float ibi_cv = 0.0f;
};

inline BurstStats ComputeBurstStats(
    const std::vector<Burst>& bursts,
    float total_duration_ms) {
  BurstStats stats;
  stats.n_bursts = static_cast<int>(bursts.size());
  if (bursts.empty()) return stats;

  float dur_sum = 0.0f, spk_sum = 0.0f;
  for (auto& b : bursts) {
    dur_sum += b.duration_ms();
    spk_sum += b.n_spikes;
  }
  stats.mean_duration_ms = dur_sum / bursts.size();
  stats.mean_spikes_per_burst = spk_sum / bursts.size();
  stats.burst_fraction = dur_sum / total_duration_ms;

  if (bursts.size() >= 2) {
    std::vector<float> ibis;
    for (size_t i = 1; i < bursts.size(); ++i)
      ibis.push_back(bursts[i].onset_ms - bursts[i - 1].onset_ms);
    float sum = std::accumulate(ibis.begin(), ibis.end(), 0.0f);
    stats.mean_ibi_ms = sum / ibis.size();

    float sq = 0.0f;
    for (float v : ibis) sq += (v - stats.mean_ibi_ms) * (v - stats.mean_ibi_ms);
    float sd = std::sqrt(sq / ibis.size());
    stats.ibi_cv = (stats.mean_ibi_ms > 0.0f) ? sd / stats.mean_ibi_ms : 0.0f;
  }

  return stats;
}

// Population synchrony: Fano factor of binned spike counts.
// High Fano factor (>1) indicates synchronous population activity.
// Fano factor ~1 is Poisson (asynchronous). <1 is anti-correlated.
inline float PopulationFanoFactor(
    const std::vector<std::vector<float>>& spike_trains,
    float bin_ms = 5.0f,
    float duration_ms = 1000.0f) {

  int n_bins = static_cast<int>(duration_ms / bin_ms) + 1;
  std::vector<int> pop_counts(n_bins, 0);

  for (auto& train : spike_trains) {
    for (float t : train) {
      int bin = static_cast<int>(t / bin_ms);
      if (bin >= 0 && bin < n_bins) pop_counts[bin]++;
    }
  }

  // Compute Fano factor = var / mean of bin counts
  float sum = 0.0f;
  for (int c : pop_counts) sum += c;
  float mean = sum / n_bins;
  if (mean < 0.01f) return 0.0f;

  float sq = 0.0f;
  for (int c : pop_counts) sq += (c - mean) * (c - mean);
  float var = sq / n_bins;

  return var / mean;
}

// Auto-correlation of population spike count time series.
// Returns correlation at each lag (in bins). Useful for detecting
// oscillation frequency: first peak in auto-correlation gives period.
inline std::vector<float> PopulationAutocorrelation(
    const std::vector<std::vector<float>>& spike_trains,
    float bin_ms = 1.0f,
    float duration_ms = 1000.0f,
    int max_lag_bins = 200) {

  int n_bins = static_cast<int>(duration_ms / bin_ms) + 1;
  std::vector<float> counts(n_bins, 0.0f);

  for (auto& train : spike_trains) {
    for (float t : train) {
      int bin = static_cast<int>(t / bin_ms);
      if (bin >= 0 && bin < n_bins) counts[bin] += 1.0f;
    }
  }

  // Mean-subtract
  float mean = 0.0f;
  for (float c : counts) mean += c;
  mean /= n_bins;
  for (float& c : counts) c -= mean;

  // Compute autocorrelation
  max_lag_bins = std::min(max_lag_bins, n_bins / 2);
  std::vector<float> ac(max_lag_bins, 0.0f);

  float var = 0.0f;
  for (float c : counts) var += c * c;
  if (var < 1e-10f) return ac;

  for (int lag = 0; lag < max_lag_bins; ++lag) {
    float s = 0.0f;
    for (int i = 0; i < n_bins - lag; ++i)
      s += counts[i] * counts[i + lag];
    ac[lag] = s / var;
  }

  return ac;
}

// Find dominant oscillation frequency from autocorrelation.
// Returns period in ms (0 if no clear oscillation found).
inline float DetectOscillationPeriod(
    const std::vector<float>& autocorr,
    float bin_ms = 1.0f,
    float min_peak_height = 0.1f) {

  // Skip lag 0, find first peak after first trough
  if (autocorr.size() < 10) return 0.0f;

  // Find first trough (AC goes below 0)
  int trough_idx = 1;
  while (trough_idx < static_cast<int>(autocorr.size()) - 1 &&
         autocorr[trough_idx] > 0.0f)
    trough_idx++;

  if (trough_idx >= static_cast<int>(autocorr.size()) - 2) return 0.0f;

  // Find first peak after trough
  float best_val = -1.0f;
  int best_idx = -1;
  for (int i = trough_idx + 1; i < static_cast<int>(autocorr.size()) - 1; ++i) {
    if (autocorr[i] > autocorr[i - 1] && autocorr[i] > autocorr[i + 1] &&
        autocorr[i] > min_peak_height) {
      if (autocorr[i] > best_val) {
        best_val = autocorr[i];
        best_idx = i;
      }
      break;  // take first peak
    }
  }

  return (best_idx > 0) ? best_idx * bin_ms : 0.0f;
}

// Extract per-neuron spike times from simulation step data.
// Call each timestep with the spiked array; builds per-neuron spike trains.
struct SpikeCollector {
  std::vector<std::vector<float>> trains;  // trains[neuron_idx] = sorted spike times

  void Init(size_t n_neurons) {
    trains.resize(n_neurons);
  }

  void Record(const uint8_t* spiked, size_t n, float t_ms) {
    for (size_t i = 0; i < n; ++i) {
      if (spiked[i]) trains[i].push_back(t_ms);
    }
  }

  // Get spike times for a range of neurons (combined, sorted).
  std::vector<float> GetPopulationTrain(size_t start, size_t end) const {
    std::vector<float> combined;
    end = std::min(end, trains.size());
    for (size_t i = start; i < end; ++i) {
      combined.insert(combined.end(), trains[i].begin(), trains[i].end());
    }
    std::sort(combined.begin(), combined.end());
    return combined;
  }

  // Get per-neuron trains for a range (for synchrony analysis).
  std::vector<std::vector<float>> GetTrains(size_t start, size_t end) const {
    end = std::min(end, trains.size());
    return std::vector<std::vector<float>>(
        trains.begin() + start, trains.begin() + end);
  }

  void Clear() {
    for (auto& t : trains) t.clear();
  }

  // Total spike count across all neurons.
  size_t TotalSpikes() const {
    size_t n = 0;
    for (auto& t : trains) n += t.size();
    return n;
  }
};

}  // namespace mechabrain

#endif  // PARABRAIN_SPIKE_ANALYSIS_H_
