#ifndef FWMC_NEURON_REPLACER_H_
#define FWMC_NEURON_REPLACER_H_

#include <vector>

namespace mechabrain {

// Manages gradual replacement of biological neurons with digital twins.
// State machine per neuron:
//   BIOLOGICAL -> MONITORED -> BRIDGED -> REPLACED
// Each transition requires sustained correlation above threshold
// for a minimum observation period.
//
// Safety features:
//   - Hysteresis: promotion requires threshold + margin, demotion at threshold - margin
//   - Automatic rollback: BRIDGED neurons that diverge get demoted to MONITORED
//   - Confidence tracking: requires high correlation, not just above threshold
struct NeuronReplacer {
  enum class State : uint8_t {
    kBiological = 0,  // running on tissue, not monitored
    kMonitored = 1,   // running on tissue, read channel active
    kBridged = 2,     // digital twin running in parallel with write
    kReplaced = 3,    // biological silenced, digital drives output
  };

  std::vector<State> state;
  std::vector<float> time_in_state;
  std::vector<float> running_correlation;
  std::vector<float> min_correlation;     // worst correlation seen in current state
  std::vector<int> rollback_count;        // how many times this neuron was rolled back

  float monitor_threshold = 0.6f;
  float bridge_threshold = 0.8f;
  float min_observation_ms = 10000.0f;

  // Hysteresis: need threshold + margin to promote, demote at threshold - margin
  float hysteresis_margin = 0.1f;

  // Automatic rollback threshold: if correlation drops below this, demote
  float rollback_threshold = 0.3f;

  // Max rollbacks before giving up on a neuron
  int max_rollbacks = 3;

  void Init(size_t n) {
    state.assign(n, State::kBiological);
    time_in_state.assign(n, 0.0f);
    running_correlation.assign(n, 0.0f);
    min_correlation.assign(n, 1.0f);
    rollback_count.assign(n, 0);
  }

  void BeginMonitoring(const std::vector<uint32_t>& indices) {
    for (auto idx : indices) {
      if (idx < state.size() && state[idx] == State::kBiological) {
        state[idx] = State::kMonitored;
        time_in_state[idx] = 0.0f;
        running_correlation[idx] = 0.0f;
        min_correlation[idx] = 1.0f;
      }
    }
  }

  void UpdateCorrelation(uint32_t idx, float correlation, float dt_ms) {
    if (idx >= state.size()) return;
    time_in_state[idx] += dt_ms;
    float alpha = std::min(1.0f, dt_ms / 1000.0f);
    running_correlation[idx] =
        (1.0f - alpha) * running_correlation[idx] + alpha * correlation;

    // Track worst-case correlation in this state
    if (running_correlation[idx] < min_correlation[idx]) {
      min_correlation[idx] = running_correlation[idx];
    }
  }

  std::vector<uint32_t> TryAdvance() {
    std::vector<uint32_t> promoted;
    for (size_t i = 0; i < state.size(); ++i) {
      if (time_in_state[i] < min_observation_ms) continue;

      // Don't promote neurons that have been rolled back too many times
      if (rollback_count[i] >= max_rollbacks) continue;

      switch (state[i]) {
        case State::kMonitored:
          // Hysteresis: require threshold + margin to promote
          if (running_correlation[i] >= monitor_threshold + hysteresis_margin &&
              min_correlation[i] >= monitor_threshold * 0.8f) {
            state[i] = State::kBridged;
            time_in_state[i] = 0.0f;
            min_correlation[i] = 1.0f;
            promoted.push_back(static_cast<uint32_t>(i));
          }
          break;
        case State::kBridged:
          if (running_correlation[i] >= bridge_threshold + hysteresis_margin &&
              min_correlation[i] >= bridge_threshold * 0.8f) {
            state[i] = State::kReplaced;
            time_in_state[i] = 0.0f;
            min_correlation[i] = 1.0f;
            promoted.push_back(static_cast<uint32_t>(i));
          }
          break;
        default: break;
      }
    }
    return promoted;
  }

  // Automatic rollback for neurons whose correlation drops too low.
  // Called after resync events to demote unstable neurons.
  std::vector<uint32_t> RollbackDiverged(float threshold) {
    std::vector<uint32_t> demoted;
    for (size_t i = 0; i < state.size(); ++i) {
      if ((state[i] == State::kBridged || state[i] == State::kReplaced) &&
          running_correlation[i] < threshold) {
        // Demote to MONITORED (kReplaced neurons are most dangerous to
        // leave unchecked since the bio neuron is silenced)
        state[i] = State::kMonitored;
        time_in_state[i] = 0.0f;
        running_correlation[i] = 0.0f;
        min_correlation[i] = 1.0f;
        rollback_count[i]++;
        demoted.push_back(static_cast<uint32_t>(i));
      }
    }
    return demoted;
  }

  void Rollback(uint32_t idx) {
    if (idx < state.size()) {
      state[idx] = State::kBiological;
      time_in_state[idx] = 0.0f;
      running_correlation[idx] = 0.0f;
      min_correlation[idx] = 1.0f;
      rollback_count[idx]++;
    }
  }

  size_t CountInState(State s) const {
    size_t c = 0;
    for (auto st : state) if (st == s) c++;
    return c;
  }

  float ReplacementFraction() const {
    if (state.empty()) return 0;
    return static_cast<float>(CountInState(State::kReplaced)) / state.size();
  }

  std::vector<uint32_t> GetIndicesInState(State s) const {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < state.size(); ++i)
      if (state[i] == s) result.push_back(static_cast<uint32_t>(i));
    return result;
  }

  // Adaptive boundary refinement: when a MONITORED/BRIDGED neuron has
  // high drift, auto-promote its BIOLOGICAL neighbors to MONITORED so
  // the read channel can track the spreading divergence front.
  // Inspired by SkiBiDy's hybrid agent-continuum boundary detection.
  //
  // Bidirectional: promotes both post-synaptic targets (forward, output side)
  // and pre-synaptic sources (backward, input side). Divergence propagates
  // both ways: a drifting neuron sends wrong spikes downstream, AND a neuron
  // may drift because its upstream inputs have changed.
  //
  // forward[i] = post-synaptic neuron indices (outputs).
  // reverse[i] = pre-synaptic neuron indices (inputs).
  // drift_threshold: neurons with running_correlation below this trigger
  //                  neighbor promotion.
  std::vector<uint32_t> AutoPromoteNeighbors(
      const std::vector<std::vector<uint32_t>>& forward,
      float drift_threshold = 0.5f,
      const std::vector<std::vector<uint32_t>>& reverse = {}) {
    std::vector<uint32_t> promoted;
    for (size_t i = 0; i < state.size(); ++i) {
      // Only look at actively tracked neurons with poor correlation
      if (state[i] != State::kMonitored && state[i] != State::kBridged)
        continue;
      if (running_correlation[i] >= drift_threshold) continue;

      // This neuron is drifting; promote its post-synaptic targets (forward)
      if (i < forward.size()) {
        for (uint32_t nb : forward[i]) {
          if (nb < state.size() && state[nb] == State::kBiological) {
            state[nb] = State::kMonitored;
            time_in_state[nb] = 0.0f;
            running_correlation[nb] = 0.0f;
            min_correlation[nb] = 1.0f;
            promoted.push_back(nb);
          }
        }
      }

      // Also promote its pre-synaptic sources (reverse) to catch upstream causes
      if (i < reverse.size()) {
        for (uint32_t nb : reverse[i]) {
          if (nb < state.size() && state[nb] == State::kBiological) {
            state[nb] = State::kMonitored;
            time_in_state[nb] = 0.0f;
            running_correlation[nb] = 0.0f;
            min_correlation[nb] = 1.0f;
            promoted.push_back(nb);
          }
        }
      }
    }
    return promoted;
  }
};

}  // namespace mechabrain

#endif  // FWMC_NEURON_REPLACER_H_
