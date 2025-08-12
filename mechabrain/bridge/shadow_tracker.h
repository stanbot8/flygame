#ifndef FWMC_SHADOW_TRACKER_H_
#define FWMC_SHADOW_TRACKER_H_

#include <cmath>
#include <deque>
#include <numeric>
#include <vector>
#include "bridge/bridge_channel.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Shadow mode: digital twin runs in parallel with biological brain,
// receiving the same inputs but NOT writing back. Measures how
// quickly the digital prediction diverges from biological reality.
// Phase 2 of the twinning protocol.
struct ShadowTracker {
  struct DriftSnapshot {
    float time_ms;
    float spike_correlation;
    float population_rmse;
    float mean_v_error;
    int n_false_positive;
    int n_false_negative;
    float time_since_resync;

    // Population firing rates (Hz-like: fraction of neurons spiking this frame)
    float digital_spike_rate = 0.0f;  // fraction of monitored neurons spiking
    float bio_spike_rate = 0.0f;      // fraction of bio readings with spike_prob > 0.5
    float rate_ratio = 1.0f;          // digital/bio ratio (1.0 = matched, >1 = overactive)
  };

  std::deque<DriftSnapshot> history;
  size_t max_history_size = 10000;  // cap to prevent unbounded growth
  float last_resync_time = 0.0f;

  DriftSnapshot Measure(const NeuronArray& digital,
                        const std::vector<BioReading>& bio,
                        float sim_time_ms) {
    DriftSnapshot snap{};
    snap.time_ms = sim_time_ms;
    snap.time_since_resync = sim_time_ms - last_resync_time;

    if (bio.empty()) {
      history.push_back(snap);
      while (history.size() > max_history_size) history.pop_front();
      return snap;
    }

    std::vector<float> predicted, observed;
    float v_error_sum = 0.0f;
    int v_count = 0;

    for (const auto& b : bio) {
      if (b.neuron_idx >= digital.n) continue;
      float p = digital.spiked[b.neuron_idx] ? 1.0f : 0.0f;
      float o = b.spike_prob;
      predicted.push_back(p);
      observed.push_back(o);

      if (p > 0.5f && o < 0.3f) snap.n_false_positive++;
      if (p < 0.5f && o > 0.7f) snap.n_false_negative++;

      if (!std::isnan(b.voltage_mv)) {
        v_error_sum += std::abs(digital.v[b.neuron_idx] - b.voltage_mv);
        v_count++;
      }
    }

    size_t nn = predicted.size();
    if (nn == 0) {
      history.push_back(snap);
      while (history.size() > max_history_size) history.pop_front();
      return snap;
    }

    // Pearson correlation
    float mean_p = std::accumulate(predicted.begin(), predicted.end(), 0.0f) / nn;
    float mean_o = std::accumulate(observed.begin(), observed.end(), 0.0f) / nn;
    float cov = 0, var_p = 0, var_o = 0;
    for (size_t i = 0; i < nn; ++i) {
      float dp = predicted[i] - mean_p;
      float d_o = observed[i] - mean_o;
      cov += dp * d_o;
      var_p += dp * dp;
      var_o += d_o * d_o;
    }
    float denom = std::sqrt(var_p * var_o);
    snap.spike_correlation = (denom > 1e-9f) ? (cov / denom) : 0.0f;

    // RMSE
    float mse = 0;
    for (size_t i = 0; i < nn; ++i) {
      float diff = predicted[i] - observed[i];
      mse += diff * diff;
    }
    snap.population_rmse = std::sqrt(mse / nn);
    snap.mean_v_error = (v_count > 0) ? (v_error_sum / v_count) : 0.0f;

    // Population firing rate comparison
    float n_digital_spike = 0.0f, n_bio_spike = 0.0f;
    for (size_t i = 0; i < nn; ++i) {
      if (predicted[i] > 0.5f) n_digital_spike += 1.0f;
      if (observed[i] > 0.5f) n_bio_spike += 1.0f;
    }
    float fn = static_cast<float>(nn);
    snap.digital_spike_rate = n_digital_spike / fn;
    snap.bio_spike_rate = n_bio_spike / fn;
    snap.rate_ratio = (n_bio_spike > 0.5f)
        ? (n_digital_spike / n_bio_spike) : 1.0f;

    history.push_back(snap);
    while (history.size() > max_history_size) {
      history.pop_front();
    }
    return snap;
  }

  // Resynchronize: copy biological state into digital twin
  void Resync(NeuronArray& digital, const std::vector<BioReading>& bio,
              float sim_time_ms) {
    for (const auto& b : bio) {
      if (b.neuron_idx >= digital.n) continue;
      if (!std::isnan(b.voltage_mv)) {
        digital.v[b.neuron_idx] = b.voltage_mv;
      } else if (b.spike_prob > 0.7f) {
        digital.v[b.neuron_idx] = 30.0f;
        digital.spiked[b.neuron_idx] = 1;
      }
    }
    last_resync_time = sim_time_ms;
  }

  bool DriftExceedsThreshold(float threshold = 0.5f) const {
    if (history.empty()) return false;
    return history.back().spike_correlation < threshold;
  }

  // Check if drift is accelerating: compare correlation in recent window
  // vs earlier window. Returns true if correlation is dropping faster
  // than the tolerance rate. Useful for early warning before hard threshold.
  bool DriftAccelerating(size_t window = 20, float tolerance = 0.05f) const {
    if (history.size() < 2 * window) return false;
    float recent_avg = 0.0f, earlier_avg = 0.0f;
    size_t hs = history.size();
    for (size_t i = 0; i < window; ++i) {
      recent_avg += history[hs - 1 - i].spike_correlation;
      earlier_avg += history[hs - 1 - window - i].spike_correlation;
    }
    recent_avg /= static_cast<float>(window);
    earlier_avg /= static_cast<float>(window);
    return (earlier_avg - recent_avg) > tolerance;
  }

  // Check if digital twin has pathological firing rate (>5x or <0.2x bio).
  // Catches runaway excitation or quiescence that correlation alone misses.
  bool RateDiverged(float max_ratio = 5.0f) const {
    if (history.empty()) return false;
    const auto& snap = history.back();
    return snap.rate_ratio > max_ratio || snap.rate_ratio < (1.0f / max_ratio);
  }
};

}  // namespace mechabrain

#endif  // FWMC_SHADOW_TRACKER_H_
