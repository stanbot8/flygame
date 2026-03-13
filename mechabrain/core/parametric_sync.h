#ifndef FWMC_PARAMETRIC_SYNC_H_
#define FWMC_PARAMETRIC_SYNC_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/cell_types.h"
#include "core/izhikevich.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Sync mode for the parametric brain: runs the parametric model alongside a
// reference brain (real hardware readings or a second simulation) and
// adaptively adjusts synaptic weights, external currents, and neuron
// parameters to minimize divergence.
//
// Three adaptation mechanisms run at different timescales:
//   FAST:   per-step external current injection to match instantaneous voltage
//   MEDIUM: periodic synaptic weight updates via error attribution
//   SLOW:   periodic Izhikevich parameter nudges toward better dynamics
//
// The reference can be:
//   (a) A NeuronArray from another simulation (sim-to-sim sync)
//   (b) Biological readings from a ReadChannel (bio-to-sim sync)
//
// This enables tuning a parametric model to match a specific fly's dynamics
// before using it as the digital twin in the bridge protocol.

// Per-neuron sync state.
struct NeuronSyncState {
  float correlation = 0.0f;           // exponential moving average
  float voltage_error_ema = 0.0f;     // EMA of |v_model - v_ref|
  float firing_rate_model = 0.0f;     // EMA firing rate (Hz) for model
  float firing_rate_ref = 0.0f;       // EMA firing rate (Hz) for reference
  int spikes_model = 0;               // spike count in current window
  int spikes_ref = 0;                 // spike count in current window
  bool converged = false;             // correlation >= converge_threshold
};

// Aggregated sync metrics for logging and convergence checking.
struct SyncSnapshot {
  float time_ms = 0.0f;
  float global_correlation = 0.0f;    // mean per-neuron correlation
  float global_rmse = 0.0f;           // population voltage RMSE
  float mean_firing_rate_error = 0.0f;// mean |rate_model - rate_ref|
  float fraction_converged = 0.0f;    // fraction of neurons converged
  int n_weight_updates = 0;
  int n_param_nudges = 0;
};

struct ParametricSync {
  // Configuration
  float dt_ms = 0.1f;
  float weight_scale = 1.0f;

  // Adaptation rates
  float current_injection_gain = 0.5f;     // FAST: scale of corrective i_ext
  float weight_lr = 0.0005f;               // MEDIUM: synaptic weight learning rate
  float weight_momentum = 0.9f;            // MEDIUM: momentum for weight updates
  float weight_decay = 1e-5f;              // MEDIUM: L2 regularization
  float param_nudge_rate = 0.001f;         // SLOW: Izhikevich param adjustment rate

  // Timescales (in steps)
  int weight_update_interval = 100;        // steps between weight updates
  int param_update_interval = 1000;        // steps between param nudges
  int metric_interval = 500;               // steps between metric snapshots

  // Convergence
  float converge_threshold = 0.85f;        // per-neuron correlation to count as converged
  float ema_alpha = 0.01f;                 // EMA decay for per-neuron stats
  float target_convergence = 0.95f;        // stop when this fraction converged

  // State
  std::vector<NeuronSyncState> neuron_state;
  std::vector<float> weight_velocity;      // momentum for weight updates
  std::vector<float> weight_error_accum;   // per-synapse error accumulator
  std::vector<float> corrective_buf;       // reusable buffer for corrective currents
  int weight_samples = 0;
  std::vector<SyncSnapshot> history;

  float sim_time_ms = 0.0f;
  int total_steps = 0;

  void Init(size_t n_neurons, size_t n_synapses) {
    neuron_state.resize(n_neurons);
    weight_velocity.assign(n_synapses, 0.0f);
    weight_error_accum.assign(n_synapses, 0.0f);
    corrective_buf.resize(n_neurons);
    weight_samples = 0;
  }

  // Run one sync step: advance model, compare with reference, adapt.
  // `model` is the parametric brain being tuned.
  // `ref` is the reference brain (ground truth).
  // Both must have the same neuron count.
  void Step(NeuronArray& model, SynapseTable& synapses,
            const NeuronArray& ref, CellTypeManager& types) {
    size_t n = model.n;

    // --- FAST: corrective current injection ---
    // Push model voltages toward reference by injecting a small corrective current.
    // This prevents runaway divergence while the slower mechanisms learn.
    // Save injected values so we can remove the exact amount after dynamics step.
    corrective_buf.resize(n);
    for (size_t i = 0; i < n; ++i) {
      corrective_buf[i] = current_injection_gain * (ref.v[i] - model.v[i]);
      model.i_ext[i] += corrective_buf[i];
    }

    // --- Dynamics step ---
    model.ClearSynapticInput();
    synapses.PropagateSpikes(model.spiked.data(), model.i_syn.data(),
                             weight_scale);
    IzhikevichStepHeterogeneous(model, dt_ms, sim_time_ms, types);

    // --- Per-neuron tracking ---
    for (size_t i = 0; i < n; ++i) {
      auto& ns = neuron_state[i];

      // Update spike counts
      if (model.spiked[i]) ns.spikes_model++;
      if (ref.spiked[i]) ns.spikes_ref++;

      // EMA voltage error
      float v_err = std::abs(model.v[i] - ref.v[i]);
      ns.voltage_error_ema = ema_alpha * v_err + (1.0f - ema_alpha) * ns.voltage_error_ema;

      // EMA spike correlation (binary match)
      float match = (model.spiked[i] == ref.spiked[i]) ? 1.0f : 0.0f;
      ns.correlation = ema_alpha * match + (1.0f - ema_alpha) * ns.correlation;
      ns.converged = (ns.correlation >= converge_threshold);
    }

    // --- MEDIUM: synaptic weight error accumulation ---
    AccumulateWeightError(synapses, model, ref);

    if (total_steps > 0 && total_steps % weight_update_interval == 0) {
      ApplyWeightUpdates(synapses);
    }

    // --- SLOW: parameter nudges ---
    if (total_steps > 0 && total_steps % param_update_interval == 0) {
      NudgeParams(model, types);
    }

    // --- Metrics ---
    if (total_steps % metric_interval == 0) {
      RecordMetrics(model, ref, n);
    }

    // Remove corrective current (exact amount that was injected)
    for (size_t i = 0; i < n; ++i) {
      model.i_ext[i] -= corrective_buf[i];
    }

    sim_time_ms += dt_ms;
    total_steps++;
  }

  // Check if the model has converged to the reference.
  bool HasConverged() const {
    if (history.empty()) return false;
    return history.back().fraction_converged >= target_convergence;
  }

  // Get the latest sync snapshot.
  const SyncSnapshot& Latest() const {
    return history.back();
  }

 private:
  void AccumulateWeightError(const SynapseTable& synapses,
                              const NeuronArray& model,
                              const NeuronArray& ref) {
    for (size_t pre = 0; pre < synapses.n_neurons; ++pre) {
      if (!model.spiked[pre] && !ref.spiked[pre]) continue;

      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];

      for (uint32_t s = start; s < end; ++s) {
        uint32_t post = synapses.post[s];
        // Error: model is too active -> reduce weight, model is too quiet -> increase
        float model_spike = model.spiked[post] ? 1.0f : 0.0f;
        float ref_spike = ref.spiked[post] ? 1.0f : 0.0f;
        weight_error_accum[s] += (model_spike - ref_spike);
      }
    }
    weight_samples++;
  }

  void ApplyWeightUpdates(SynapseTable& synapses) {
    if (weight_samples == 0) return;

    float inv_n = 1.0f / static_cast<float>(weight_samples);
    int n_updated = 0;

    for (size_t s = 0; s < synapses.Size(); ++s) {
      float grad = weight_error_accum[s] * inv_n;

      // Momentum SGD
      weight_velocity[s] = weight_momentum * weight_velocity[s]
                           - weight_lr * grad;

      // L2 weight decay
      weight_velocity[s] -= weight_decay * synapses.weight[s];

      float new_w = synapses.weight[s] + weight_velocity[s];
      new_w = std::clamp(new_w, 0.01f, 20.0f);

      if (new_w != synapses.weight[s]) n_updated++;
      synapses.weight[s] = new_w;
    }

    if (!history.empty()) {
      history.back().n_weight_updates = n_updated;
    }

    std::fill(weight_error_accum.begin(), weight_error_accum.end(), 0.0f);
    weight_samples = 0;
  }

  void NudgeParams(const NeuronArray& model, CellTypeManager& types) {
    // For each neuron, nudge Izhikevich params based on firing rate mismatch.
    // If model fires too fast -> increase 'a' (faster recovery), decrease 'd'
    // If model fires too slow -> decrease 'a', increase 'd'
    size_t n = model.n;
    int n_nudged = 0;

    float window_s = (param_update_interval * dt_ms) / 1000.0f;
    if (window_s <= 0.0f) return;

    for (size_t i = 0; i < n; ++i) {
      auto& ns = neuron_state[i];
      float rate_model = static_cast<float>(ns.spikes_model) / window_s;
      float rate_ref = static_cast<float>(ns.spikes_ref) / window_s;

      // Update EMA firing rates
      ns.firing_rate_model = 0.1f * rate_model + 0.9f * ns.firing_rate_model;
      ns.firing_rate_ref = 0.1f * rate_ref + 0.9f * ns.firing_rate_ref;

      // Reset window counters
      ns.spikes_model = 0;
      ns.spikes_ref = 0;

      // Skip converged neurons
      if (ns.converged) continue;

      float rate_error = rate_model - rate_ref;
      if (std::abs(rate_error) < 1.0f) continue;  // close enough

      auto& p = types.neuron_params[i];
      float sign = (rate_error > 0) ? 1.0f : -1.0f;

      // Nudge 'a' (recovery rate): higher = faster recovery = fewer spikes
      p.a += sign * param_nudge_rate * 0.01f;
      p.a = std::clamp(p.a, 0.01f, 0.15f);

      // Nudge 'd' (after-spike reset): higher = stronger reset = fewer spikes
      p.d += sign * param_nudge_rate * 1.0f;
      p.d = std::clamp(p.d, 1.0f, 10.0f);

      // Nudge 'c' (reset voltage): lower = more inhibited after spike
      p.c -= sign * param_nudge_rate * 2.0f;
      p.c = std::clamp(p.c, -70.0f, -45.0f);

      n_nudged++;
    }

    if (!history.empty()) {
      history.back().n_param_nudges = n_nudged;
    }

    Log(LogLevel::kInfo, "ParamSync: nudged %d/%zu neuron params (t=%.1fms)",
        n_nudged, n, sim_time_ms);
  }

  void RecordMetrics(const NeuronArray& model, const NeuronArray& ref,
                      size_t n) {
    SyncSnapshot snap;
    snap.time_ms = sim_time_ms;

    float corr_sum = 0.0f;
    float rmse_sum = 0.0f;
    float rate_err_sum = 0.0f;
    int n_converged = 0;

    for (size_t i = 0; i < n; ++i) {
      corr_sum += neuron_state[i].correlation;
      float v_diff = model.v[i] - ref.v[i];
      rmse_sum += v_diff * v_diff;
      rate_err_sum += std::abs(neuron_state[i].firing_rate_model -
                               neuron_state[i].firing_rate_ref);
      if (neuron_state[i].converged) n_converged++;
    }

    float fn = static_cast<float>(n);
    if (fn < 1.0f) fn = 1.0f;  // prevent division by zero for empty arrays
    snap.global_correlation = corr_sum / fn;
    snap.global_rmse = std::sqrt(rmse_sum / fn);
    snap.mean_firing_rate_error = rate_err_sum / fn;
    snap.fraction_converged = static_cast<float>(n_converged) / fn;

    history.push_back(snap);

    Log(LogLevel::kInfo,
        "ParamSync: t=%.1fms  corr=%.3f  rmse=%.2f  rate_err=%.1fHz  converged=%.1f%%",
        sim_time_ms, snap.global_correlation, snap.global_rmse,
        snap.mean_firing_rate_error, snap.fraction_converged * 100.0f);
  }
};

}  // namespace mechabrain

#endif  // FWMC_PARAMETRIC_SYNC_H_
