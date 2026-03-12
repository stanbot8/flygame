#ifndef FWMC_GAP_JUNCTIONS_H_
#define FWMC_GAP_JUNCTIONS_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/neuron_array.h"

namespace mechabrain {

// Electrical gap junction storage.
// Gap junctions pass current proportional to the voltage difference
// between two coupled neurons: I_gap = g * (V_post - V_pre).
// They are bidirectional (symmetric): neuron_a receives +I, neuron_b
// receives -I, so current flows from the higher-voltage cell to the
// lower-voltage cell.
//
// Important in Drosophila:
//   - Giant fiber system (GF1/GF2 escape response, ~1ms latency)
//   - Clock neurons (LNv synchronization via Inx6/Inx7 innexins)
//   - Antennal lobe local interneurons (fast oscillatory coupling)
struct GapJunctionTable {
  // Parallel arrays: junction j connects neuron_a[j] <-> neuron_b[j]
  // with conductance[j]. All three arrays have the same length.
  std::vector<uint32_t> neuron_a;
  std::vector<uint32_t> neuron_b;
  std::vector<float>    conductance;  // gap junction conductance (nS)
  std::vector<float>    rectification; // voltage-dependent rectification factor [0,1]
                                       // 0 = fully rectifying (one-way), 1 = symmetric (default)
                                       // Drosophila innexin channels show VDR (Phelan 2005)

  size_t Size() const { return neuron_a.size(); }

  // Add a gap junction. rectify=1.0 for symmetric, <1.0 for rectifying.
  // Rectifying junctions preferentially pass current in one direction
  // (from a to b when rectify < 1), modeling innexin channel asymmetry.
  void AddJunction(uint32_t a, uint32_t b, float g, float rectify = 1.0f) {
    neuron_a.push_back(a);
    neuron_b.push_back(b);
    conductance.push_back(g);
    rectification.push_back(rectify);
  }

  // Persistent buffer for thread-local reduction in PropagateGapCurrents.
  // Allocated once, reused across timesteps. Avoids per-step heap churn.
  mutable std::vector<float> omp_reduce_buf_;

  // Propagate gap junction currents into neurons.i_ext.
  // For each junction j:
  //   I = g * (Vb - Va)
  //   neuron_a.i_ext += I   (current into a)
  //   neuron_b.i_ext -= I   (current into b, equal and opposite)
  //
  // Uses thread-local accumulation buffers with a final reduction pass
  // to avoid atomic writes (same strategy as SynapseTable::PropagateSpikes).
  // Each thread accumulates into its own buffer, then buffers are summed.
  // Eliminates the lock-cmpxchg loop that atomic float adds compile to.
  void PropagateGapCurrents(NeuronArray& neurons) const {
    const int n = static_cast<int>(Size());
    if (n == 0) return;

    float* i_ext = neurons.i_ext.data();
    const float* v = neurons.v.data();

    #ifdef _OPENMP
    if (n > 10000) {
      int n_threads = 1;
      #pragma omp parallel
      {
        #pragma omp single
        n_threads = omp_get_num_threads();
      }

      const size_t nn = neurons.n;
      size_t buf_size = static_cast<size_t>(n_threads) * nn;
      if (omp_reduce_buf_.size() < buf_size) {
        omp_reduce_buf_.resize(buf_size);
      }
      std::fill(omp_reduce_buf_.begin(),
                omp_reduce_buf_.begin() + static_cast<ptrdiff_t>(buf_size),
                0.0f);

      #pragma omp parallel num_threads(n_threads)
      {
        int tid = omp_get_thread_num();
        float* local = omp_reduce_buf_.data() + static_cast<size_t>(tid) * nn;

        #pragma omp for schedule(static)
        for (int j = 0; j < n; ++j) {
          uint32_t a = neuron_a[j];
          uint32_t b = neuron_b[j];
          float dv = v[b] - v[a];
          float g = conductance[j];

          float r = rectification[j];
          if (r < 1.0f && dv < 0.0f) {
            g *= r;
          }

          float I = g * dv;
          local[a] += I;
          local[b] -= I;
        }

        // Reduction: each thread adds its buffer into i_ext
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(nn); ++i) {
          for (int t = 0; t < n_threads; ++t) {
            i_ext[i] += omp_reduce_buf_[static_cast<size_t>(t) * nn + static_cast<size_t>(i)];
          }
        }
      }
      return;
    }
    #endif

    // Scalar fallback (single-threaded or small junction count)
    for (int j = 0; j < n; ++j) {
      uint32_t a = neuron_a[j];
      uint32_t b = neuron_b[j];
      float dv = v[b] - v[a];
      float g = conductance[j];

      float r = rectification[j];
      if (r < 1.0f && dv < 0.0f) {
        g *= r;
      }

      float I = g * dv;
      i_ext[a] += I;
      i_ext[b] -= I;
    }
  }

  // Connect neurons within a region with gap junctions at a given
  // probability (density). Useful for building clock neuron networks
  // or antennal lobe coupling.
  //
  // For each pair (i, j) with i < j in the specified region, a junction
  // is created with probability `density` and conductance `g_default`.
  void BuildFromRegion(const NeuronArray& neurons, uint8_t region,
                       float density, float g_default,
                       uint32_t seed = 42) {
    // Collect neuron indices in the target region
    std::vector<uint32_t> members;
    members.reserve(256);
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == region) {
        members.push_back(static_cast<uint32_t>(i));
      }
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    for (size_t i = 0; i < members.size(); ++i) {
      for (size_t j = i + 1; j < members.size(); ++j) {
        if (coin(rng) < density) {
          AddJunction(members[i], members[j], g_default);
        }
      }
    }
  }

  void Clear() {
    neuron_a.clear();
    neuron_b.clear();
    conductance.clear();
    rectification.clear();
  }
};

}  // namespace mechabrain

#endif  // FWMC_GAP_JUNCTIONS_H_
