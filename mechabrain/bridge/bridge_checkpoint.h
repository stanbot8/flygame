#pragma once
// bridge_checkpoint.h — Serialize/deserialize bridge state (NeuronReplacer +
// ShadowTracker) into an opaque byte blob for embedding in a core Checkpoint.

#include "neuron_replacer.h"
#include "shadow_tracker.h"
#include <cstring>
#include <vector>

namespace mechabrain {

struct BridgeCheckpoint {
  static std::vector<uint8_t> Serialize(const NeuronReplacer& replacer,
                                         const ShadowTracker& shadow) {
    std::vector<uint8_t> buf;

    auto push = [&](const void* data, size_t bytes) {
      const auto* p = static_cast<const uint8_t*>(data);
      buf.insert(buf.end(), p, p + bytes);
    };

    // Replacer state
    uint32_t n = static_cast<uint32_t>(replacer.state.size());
    push(&n, 4);
    push(replacer.state.data(), n * sizeof(NeuronReplacer::State));
    push(replacer.time_in_state.data(), n * sizeof(float));
    push(replacer.running_correlation.data(), n * sizeof(float));
    push(replacer.min_correlation.data(), n * sizeof(float));
    push(replacer.rollback_count.data(), n * sizeof(int));

    // Shadow tracker
    push(&shadow.last_resync_time, sizeof(float));
    uint32_t nh = static_cast<uint32_t>(shadow.history.size());
    push(&nh, 4);
    for (const auto& snap : shadow.history) {
      push(&snap, sizeof(ShadowTracker::DriftSnapshot));
    }

    return buf;
  }

  static void Deserialize(const std::vector<uint8_t>& buf,
                           NeuronReplacer& replacer,
                           ShadowTracker& shadow) {
    if (buf.empty()) return;

    size_t pos = 0;
    auto pull = [&](void* data, size_t bytes) {
      if (pos + bytes <= buf.size()) {
        std::memcpy(data, buf.data() + pos, bytes);
        pos += bytes;
      }
    };

    // Replacer
    uint32_t n = 0;
    pull(&n, 4);
    if (n > 0 && n <= replacer.state.size()) {
      pull(replacer.state.data(), n * sizeof(NeuronReplacer::State));
      pull(replacer.time_in_state.data(), n * sizeof(float));
      pull(replacer.running_correlation.data(), n * sizeof(float));
      pull(replacer.min_correlation.data(), n * sizeof(float));
      pull(replacer.rollback_count.data(), n * sizeof(int));
    }

    // Shadow tracker
    pull(&shadow.last_resync_time, sizeof(float));
    uint32_t nh = 0;
    pull(&nh, 4);
    shadow.history.clear();
    for (uint32_t i = 0; i < nh; ++i) {
      ShadowTracker::DriftSnapshot snap{};
      pull(&snap, sizeof(ShadowTracker::DriftSnapshot));
      shadow.history.push_back(snap);
    }
  }
};

}  // namespace mechabrain
