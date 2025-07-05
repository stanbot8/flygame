#ifndef FWMC_BEHAVIORAL_FINGERPRINT_H_
#define FWMC_BEHAVIORAL_FINGERPRINT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/motor_output.h"

namespace mechabrain {

// Identity continuity metric for mind uploading validation.
//
// Tracks the behavioral output of a brain (motor commands over time)
// and compares two brains' fingerprints to measure identity preservation.
// Unlike spike-level shadow tracking (which measures neural fidelity),
// this measures whether the organism *acts the same way* given the
// same sensory input.
//
// The fingerprint is a time series of discretized motor commands.
// Comparison produces a similarity score in [0, 1] where:
//   1.0 = identical behavioral output
//   0.0 = completely uncorrelated behavior
//
// Usage for gradual neuron replacement:
//   1. Run reference brain with stimulus protocol, record fingerprint
//   2. Replace N% of neurons with digital equivalents
//   3. Re-run same protocol, record fingerprint
//   4. Compare: if similarity stays above threshold, identity is preserved
//
// This is the behavioral Ship of Theseus test.
struct BehavioralFingerprint {
  // Sampled motor commands at regular intervals
  struct Sample {
    float time_ms;
    float forward_velocity;
    float angular_velocity;
    float approach_drive;
    float freeze;
  };

  std::vector<Sample> samples;
  float sample_interval_ms = 1.0f;   // how often to sample
  float last_sample_time = -1e9f;

  void Clear() {
    samples.clear();
    last_sample_time = -1e9f;
  }

  // Record a motor command at the current time.
  // Only records if enough time has elapsed since the last sample.
  void Record(const MotorCommand& cmd, float time_ms) {
    if (time_ms - last_sample_time < sample_interval_ms) return;
    samples.push_back({time_ms, cmd.forward_velocity, cmd.angular_velocity,
                       cmd.approach_drive, cmd.freeze});
    last_sample_time = time_ms;
  }

  // Compare two fingerprints. Returns similarity in [0, 1].
  // Uses normalized cross-correlation across all motor channels,
  // averaged over the overlapping time window.
  static float Compare(const BehavioralFingerprint& a,
                        const BehavioralFingerprint& b) {
    if (a.samples.empty() || b.samples.empty()) return 0.0f;

    // Align by sample index (assumes same sample_interval_ms and
    // same stimulus protocol). Compare min(len_a, len_b) samples.
    size_t n = std::min(a.samples.size(), b.samples.size());

    // Per-channel similarity via 1 / (1 + mean_absolute_error)
    float err_fwd = 0.0f, err_ang = 0.0f, err_app = 0.0f, err_frz = 0.0f;
    float scale_fwd = 0.0f, scale_ang = 0.0f, scale_app = 0.0f;

    for (size_t i = 0; i < n; ++i) {
      err_fwd += std::abs(a.samples[i].forward_velocity -
                          b.samples[i].forward_velocity);
      err_ang += std::abs(a.samples[i].angular_velocity -
                          b.samples[i].angular_velocity);
      err_app += std::abs(a.samples[i].approach_drive -
                          b.samples[i].approach_drive);
      err_frz += std::abs(a.samples[i].freeze - b.samples[i].freeze);

      // Track signal magnitude for normalization
      scale_fwd += std::abs(a.samples[i].forward_velocity) +
                   std::abs(b.samples[i].forward_velocity);
      scale_ang += std::abs(a.samples[i].angular_velocity) +
                   std::abs(b.samples[i].angular_velocity);
      scale_app += std::abs(a.samples[i].approach_drive) +
                   std::abs(b.samples[i].approach_drive);
    }

    float fn = static_cast<float>(n);

    // Normalized error per channel: error / (signal_magnitude + epsilon)
    // This handles channels with near-zero signal gracefully.
    float eps = 1e-6f;
    float norm_fwd = (err_fwd / fn) / (scale_fwd / (2.0f * fn) + eps);
    float norm_ang = (err_ang / fn) / (scale_ang / (2.0f * fn) + eps);
    float norm_app = (err_app / fn) / (scale_app / (2.0f * fn) + eps);
    float norm_frz = err_frz / fn;  // freeze is already [0,1]

    // Convert to similarity: 1 / (1 + normalized_error)
    float sim_fwd = 1.0f / (1.0f + norm_fwd);
    float sim_ang = 1.0f / (1.0f + norm_ang);
    float sim_app = 1.0f / (1.0f + norm_app);
    float sim_frz = 1.0f / (1.0f + norm_frz);

    // Average across channels
    return (sim_fwd + sim_ang + sim_app + sim_frz) / 4.0f;
  }

  // Check if identity is preserved above a threshold.
  // Default threshold 0.85 allows for noise but catches behavioral drift.
  static bool IdentityPreserved(const BehavioralFingerprint& reference,
                                 const BehavioralFingerprint& candidate,
                                 float threshold = 0.85f) {
    return Compare(reference, candidate) >= threshold;
  }

  // Compute the Pearson correlation between two fingerprints on a
  // single motor channel. Useful for per-channel diagnostics.
  enum Channel { kForward, kAngular, kApproach, kFreeze };

  static float ChannelCorrelation(const BehavioralFingerprint& a,
                                   const BehavioralFingerprint& b,
                                   Channel ch) {
    size_t n = std::min(a.samples.size(), b.samples.size());
    if (n < 2) return 0.0f;

    auto get = [ch](const Sample& s) -> float {
      switch (ch) {
        case kForward:  return s.forward_velocity;
        case kAngular:  return s.angular_velocity;
        case kApproach: return s.approach_drive;
        case kFreeze:   return s.freeze;
      }
      return 0.0f;
    };

    // Compute means
    float mean_a = 0.0f, mean_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      mean_a += get(a.samples[i]);
      mean_b += get(b.samples[i]);
    }
    mean_a /= static_cast<float>(n);
    mean_b /= static_cast<float>(n);

    // Pearson correlation
    float cov = 0.0f, var_a = 0.0f, var_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float da = get(a.samples[i]) - mean_a;
      float db = get(b.samples[i]) - mean_b;
      cov += da * db;
      var_a += da * da;
      var_b += db * db;
    }

    float denom = std::sqrt(var_a * var_b);
    if (denom < 1e-12f) return 1.0f;  // both constant = identical
    return cov / denom;
  }
};

}  // namespace mechabrain

#endif  // FWMC_BEHAVIORAL_FINGERPRINT_H_
