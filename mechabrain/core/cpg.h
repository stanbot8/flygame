#ifndef FWMC_CPG_H_
#define FWMC_CPG_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"
#include "core/platform.h"

namespace mechabrain {

// Central Pattern Generator: injects oscillatory current into VNC neurons
// to produce spontaneous rhythmic locomotion without keyboard input.
//
// In Drosophila, the VNC contains CPG circuits that generate tripod gait
// (3 legs swing while 3 legs stance, alternating). Descending commands
// from the brain modulate CPG frequency and amplitude, but the rhythm
// is intrinsic to the VNC (Bidaye et al. 2018, Mendes et al. 2013).
//
// This module provides tonic + oscillatory drive to VNC motor neurons,
// split into two anti-phase groups (left-forward + right-hind vs
// right-forward + left-hind) to produce tripod-like alternation.
struct CPGOscillator {
  // Two anti-phase neuron groups (tripod gait pattern)
  std::vector<uint32_t> group_a;  // L-fore + R-mid + L-hind
  std::vector<uint32_t> group_b;  // R-fore + L-mid + R-hind

  float frequency_hz = 8.0f;     // stepping frequency (Drosophila: 5-15 Hz)
  float amplitude = 6.0f;        // oscillation amplitude (current units)
  float tonic_drive = 3.0f;      // constant baseline drive (keeps neurons near threshold)
  float phase = 0.0f;            // current phase (radians)
  float drive_scale = 0.0f;      // modulation from descending commands [0,1]
                                  // 0 = CPG silent, 1 = full amplitude

  // Proprioceptive feedback gains (Drosophila leg sensory feedback)
  // Ground contact from stance legs reinforces the current phase,
  // while loss of contact signals swing completion and advances phase.
  // This entrains the CPG to actual limb dynamics (Mendes et al. 2013).
  float contact_phase_gain = 0.3f;  // phase correction from ground contact
  float freq_modulation_gain = 0.5f; // frequency adjustment from load

  bool initialized = false;

  // Auto-assign VNC motor neurons to tripod groups.
  // Uses spatial position: neurons below midline-y go to group A,
  // above to group B (rough L/R alternation proxy).
  // vnc_region: region index for VNC (5 in drosophila_full.brain)
  // sensory_fraction: skip first N% of VNC neurons (they're sensory)
  void Init(const NeuronArray& neurons, uint8_t vnc_region,
            float midline_x = 250.0f, float sensory_fraction = 0.3f) {
    group_a.clear();
    group_b.clear();

    // Collect VNC neurons
    std::vector<uint32_t> vnc;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == vnc_region)
        vnc.push_back(static_cast<uint32_t>(i));
    }
    if (vnc.empty()) return;

    // Skip sensory neurons (first fraction, used by proprioception)
    size_t motor_start = static_cast<size_t>(vnc.size() * sensory_fraction);

    // Split remaining VNC neurons into two anti-phase groups by x-position.
    // Neurons left of midline go to group A, right to group B.
    // This creates L/R alternation that maps to tripod gait through
    // the existing motor output asymmetry decoding.
    for (size_t idx = motor_start; idx < vnc.size(); ++idx) {
      uint32_t ni = vnc[idx];
      if (neurons.x[ni] < midline_x)
        group_a.push_back(ni);
      else
        group_b.push_back(ni);
    }

    initialized = !group_a.empty() && !group_b.empty();
  }

  // Advance CPG phase and inject oscillatory current.
  // Call once per brain timestep (1ms).
  // descending_drive: modulation from brain [0,1]. 0 = CPG off, 1 = full.
  void Step(NeuronArray& neurons, float dt_ms, float descending_drive) {
    if (!initialized) return;

    // Smooth drive transitions (don't snap CPG on/off)
    float alpha = 1.0f - std::exp(-dt_ms / 50.0f);  // ~50ms time constant
    drive_scale += alpha * (descending_drive - drive_scale);

    if (drive_scale < 0.01f) return;  // CPG effectively off

    // Advance phase
    phase += kTwoPi * frequency_hz * dt_ms / 1000.0f;
    if (phase > kTwoPi) phase -= kTwoPi;

    // Oscillatory current: group A gets sin(phase), group B gets sin(phase + pi)
    float osc_a = std::sin(phase);
    float osc_b = std::sin(phase + kPi);  // anti-phase

    // Current = tonic + oscillation * amplitude * drive
    float current_a = (tonic_drive + osc_a * amplitude) * drive_scale;
    float current_b = (tonic_drive + osc_b * amplitude) * drive_scale;

    // Only inject positive current (negative doesn't drive spiking neurons)
    current_a = std::max(0.0f, current_a);
    current_b = std::max(0.0f, current_b);

    for (uint32_t i : group_a) neurons.i_ext[i] += current_a;
    for (uint32_t i : group_b) neurons.i_ext[i] += current_b;
  }

  // Step with proprioceptive feedback from ground contacts.
  // contacts[6]: per-leg ground contact [0,1] from ProprioState.
  // Ground contacts entrain the CPG rhythm:
  //   - Stance legs (contact > 0.5) slow phase advance (hold stance)
  //   - All legs in swing (no contact) speed up phase (faster recovery)
  //   - Total contact load modulates frequency (faster under load)
  // This couples the oscillator to actual body dynamics, producing
  // more robust locomotion than a free-running oscillator.
  void StepWithFeedback(NeuronArray& neurons, float dt_ms,
                        float descending_drive, const float contacts[6]) {
    if (!initialized) return;

    // Smooth drive transitions
    float alpha = 1.0f - std::exp(-dt_ms / 50.0f);
    drive_scale += alpha * (descending_drive - drive_scale);
    if (drive_scale < 0.01f) return;

    // Count legs in stance (contact > 0.5)
    float total_contact = 0.0f;
    for (int l = 0; l < 6; ++l) total_contact += contacts[l];

    // Phase correction: stance legs resist phase advance,
    // creating a "hold" that lets the CPG lock to footfall timing.
    // group_a legs: 0,2,4 (L-fore, L-hind, R-mid in tripod)
    // group_b legs: 1,3,5 (R-fore, R-hind, L-mid)
    float contact_a = (contacts[0] + contacts[2] + contacts[4]) / 3.0f;
    float contact_b = (contacts[1] + contacts[3] + contacts[5]) / 3.0f;

    // During stance phase (sin > 0), ground contact reinforces;
    // during swing phase (sin < 0), contact signals early touchdown.
    float osc_a = std::sin(phase);
    float phase_correction = contact_phase_gain * (contact_a * osc_a - contact_b * osc_a);

    // Frequency modulation: higher load -> slightly faster stepping
    // (flies increase step frequency with forward velocity)
    float freq_mod = 1.0f + freq_modulation_gain * (total_contact / 6.0f - 0.5f);
    float effective_freq = frequency_hz * std::clamp(freq_mod, 0.7f, 1.5f);

    // Advance phase with feedback
    phase += kTwoPi * effective_freq * dt_ms / 1000.0f + phase_correction * dt_ms / 1000.0f;
    if (phase > kTwoPi) phase -= kTwoPi;
    if (phase < 0.0f) phase += kTwoPi;

    float osc_a2 = std::sin(phase);
    float osc_b2 = std::sin(phase + kPi);

    float current_a = (tonic_drive + osc_a2 * amplitude) * drive_scale;
    float current_b = (tonic_drive + osc_b2 * amplitude) * drive_scale;
    current_a = std::max(0.0f, current_a);
    current_b = std::max(0.0f, current_b);

    for (uint32_t i : group_a) neurons.i_ext[i] += current_a;
    for (uint32_t i : group_b) neurons.i_ext[i] += current_b;
  }

  // Set drive parameters from descending commands.
  // magnitude: overall locomotion drive [0,1]
  // turn_bias: left/right turning (-1 to +1)
  void SetDrive(float magnitude, float /*turn_bias*/) {
    drive_scale = magnitude;
    // Frequency modulation from descending commands (brain can speed up gait)
    frequency_hz = 8.0f + 4.0f * magnitude;  // 8-12 Hz range
    frequency_hz = std::clamp(frequency_hz, 5.0f, 15.0f);
  }
};

}  // namespace mechabrain

#endif  // FWMC_CPG_H_
