#ifndef FWMC_SPIKE_DECODER_H_
#define FWMC_SPIKE_DECODER_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "io/channel.h"

namespace mechabrain {

// Multi-timescale calcium-to-spike deconvolution.
// Approximates CASCADE (Rupprecht et al., 2021) using a sum of
// exponential kernels at three timescales, matched to GCaMP dynamics.
//
// The three kernels capture:
//   1. Fast transient (rise, ~20ms): onset detection
//   2. Medium decay (~100ms): standard GCaMP8f decay
//   3. Slow component (~500ms): sustained/burst activity
//
// Each kernel produces a deconvolved signal; the spike probability
// is a nonlinear combination of all three, with adaptive thresholding
// based on running noise statistics.
//
// For production use with real calcium imaging, replace this with
// the actual CASCADE neural network or MLspike.
struct SpikeDecoder {
  // GCaMP8f time constants (Zhang et al., 2023)
  float tau_fast_ms = 20.0f;     // rise/fast transient
  float tau_medium_ms = 100.0f;  // standard decay
  float tau_slow_ms = 500.0f;    // sustained component

  // Kernel weights (how much each timescale contributes)
  float w_fast = 0.5f;
  float w_medium = 0.35f;
  float w_slow = 0.15f;

  // Detection parameters
  float base_threshold = 0.3f;   // minimum spike probability threshold
  float noise_floor = 0.05f;     // signals below this are noise
  float saturation = 3.0f;       // calcium saturation (df/f clips here)

  struct NeuronState {
    float prev_df_f = 0.0f;
    float deconv_fast = 0.0f;
    float deconv_medium = 0.0f;
    float deconv_slow = 0.0f;
    float baseline = 0.0f;
    float noise_std = 0.1f;      // running noise estimate
    int n_baseline_samples = 0;
  };

  std::vector<NeuronState> states;

  void Init(size_t n_neurons) { states.resize(n_neurons); }

  std::vector<BioReading> Decode(
      const std::vector<float>& raw_calcium,
      const std::vector<uint32_t>& neuron_indices,
      float dt_ms) {

    std::vector<BioReading> readings;
    readings.reserve(raw_calcium.size());

    for (size_t i = 0; i < raw_calcium.size(); ++i) {
      if (i >= states.size() || i >= neuron_indices.size()) break;
      readings.push_back(DecodeFull(i, raw_calcium[i], neuron_indices[i], dt_ms));
    }
    return readings;
  }

  // Adaptive resolution decoding: only run full multi-timescale
  // deconvolution for neurons in the active_set (MONITORED/BRIDGED).
  // For all others, use a cheap population-rate approximation:
  // just threshold the raw calcium without deconvolution.
  //
  // This cuts decode cost proportional to the fraction of neurons
  // actively tracked, which is critical when monitoring ~2000 neurons
  // out of ~140k total.
  std::vector<BioReading> DecodeSelective(
      const std::vector<float>& raw_calcium,
      const std::vector<uint32_t>& neuron_indices,
      float dt_ms,
      const std::vector<bool>& active_set) {

    std::vector<BioReading> readings;
    readings.reserve(raw_calcium.size());

    for (size_t i = 0; i < raw_calcium.size(); ++i) {
      if (i >= states.size()) break;
      uint32_t idx = neuron_indices[i];
      bool is_active = (idx < active_set.size()) && active_set[idx];

      if (is_active) {
        readings.push_back(DecodeFull(i, raw_calcium[i], idx, dt_ms));
      } else {
        // Cheap population-rate approximation: simple threshold on raw calcium
        float ca = raw_calcium[i];
        float baseline_est = (states[i].baseline > 0.01f)
                                 ? states[i].baseline : 1.0f;
        float df_f = (ca - baseline_est) / baseline_est;

        BioReading r;
        r.neuron_idx = idx;
        r.spike_prob = (df_f > 0.5f) ? std::min(1.0f, df_f * 0.5f) : 0.0f;
        r.calcium_raw = ca;
        r.voltage_mv = std::nanf("");
        readings.push_back(r);
      }
    }
    return readings;
  }

 private:
  BioReading DecodeFull(size_t state_idx, float ca, uint32_t neuron_idx, float dt_ms) {
    auto& s = states[state_idx];

    // Adaptive baseline: slow during warmup, very slow after
    if (s.n_baseline_samples < 100) {
      s.baseline += (ca - s.baseline) / (s.n_baseline_samples + 1);
      s.n_baseline_samples++;
    } else {
      float diff = ca - s.baseline;
      if (std::abs(diff) < 2.0f * s.noise_std) {
        s.baseline += 0.0005f * diff;
      }
    }

    // dF/F with saturation
    float df_f = (ca - s.baseline) / std::max(s.baseline, 0.01f);
    df_f = std::clamp(df_f, -0.5f, saturation);

    // Derivative (for onset detection)
    float deriv = (df_f - s.prev_df_f) / std::max(dt_ms, 0.01f);

    // Multi-timescale deconvolution
    float alpha_f = std::min(1.0f, dt_ms / tau_fast_ms);
    float alpha_m = std::min(1.0f, dt_ms / tau_medium_ms);
    float alpha_s = std::min(1.0f, dt_ms / tau_slow_ms);

    s.deconv_fast = (1.0f - alpha_f) * s.deconv_fast +
                    alpha_f * (df_f + tau_fast_ms * deriv);
    s.deconv_medium = (1.0f - alpha_m) * s.deconv_medium +
                      alpha_m * (df_f + tau_medium_ms * deriv);
    s.deconv_slow = (1.0f - alpha_s) * s.deconv_slow +
                    alpha_s * df_f;  // no derivative for slow component

    // Weighted combination
    float combined = w_fast * std::max(0.0f, s.deconv_fast) +
                     w_medium * std::max(0.0f, s.deconv_medium) +
                     w_slow * std::max(0.0f, s.deconv_slow);

    // Adaptive threshold based on noise statistics
    if (combined < base_threshold * 0.5f) {
      float noise_sample = std::abs(s.deconv_fast);
      s.noise_std += 0.01f * (noise_sample - s.noise_std);
    }
    float threshold = std::max(base_threshold,
                                base_threshold + 2.0f * s.noise_std);

    // Nonlinear spike probability (sigmoid-like)
    float spike_prob = 0.0f;
    if (combined > threshold) {
      float x = (combined - threshold) / std::max(threshold, 0.01f);
      spike_prob = std::min(1.0f, x * x / (1.0f + x * x));
    }

    s.prev_df_f = df_f;

    BioReading r;
    r.neuron_idx = neuron_idx;
    r.spike_prob = spike_prob;
    r.calcium_raw = ca;
    r.voltage_mv = std::nanf("");
    return r;
  }
};

}  // namespace mechabrain

#endif  // FWMC_SPIKE_DECODER_H_
