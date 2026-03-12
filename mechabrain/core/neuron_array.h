#ifndef FWMC_NEURON_ARRAY_H_
#define FWMC_NEURON_ARRAY_H_

#include <cmath>
#include <cstdint>
#include <vector>
#include "core/platform.h"

#ifdef FWMC_HAS_AVX2
  #include <immintrin.h>
#elif defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace mechabrain {

// Structure-of-arrays neuron storage.
// No virtual dispatch, no agent overhead. Just flat arrays
// that map directly to GPU memory or SIMD lanes.
//
// Index i in every array refers to the same neuron.
struct NeuronArray {
  size_t n = 0;

  // Izhikevich state
  std::vector<float> v;       // membrane potential (mV)
  std::vector<float> u;       // recovery variable
  std::vector<float> i_syn;   // accumulated synaptic input
  std::vector<float> i_ext;   // external stimulus current
  std::vector<uint8_t> spiked; // 1 if fired this step

  // Metadata
  std::vector<uint64_t> root_id;  // FlyWire root_id
  std::vector<float> x, y, z;     // position in brain (nm)
  std::vector<uint8_t> type;      // cell type index
  std::vector<uint8_t> region;    // neuropil region index

  // Neuromodulator concentrations (normalized [0, 1])
  std::vector<float> dopamine;     // DA: reward/punishment signals (DANs)
  std::vector<float> serotonin;    // 5-HT: arousal/state modulation
  std::vector<float> octopamine;   // OA: Drosophila "norepinephrine"

  // Per-neuron synaptic time constant (ms).
  // Assigned from dominant input NT type: ACh ~2ms, GABA ~5ms, Glut ~3ms.
  // Empty = use uniform tau (backward compatible).
  std::vector<float> tau_syn;

  // Cached per-neuron synaptic decay factor: exp(-dt/tau_syn[i]).
  // Recomputed only when dt changes. Eliminates N exp() calls per timestep.
  std::vector<float> decay_syn_cached;
  float decay_syn_cached_dt = -1.0f;

  // STDP
  std::vector<float> last_spike_time;

  void Resize(size_t count) {
    n = count;
    v.assign(count, -65.0f);
    u.assign(count, -13.0f);
    i_syn.assign(count, 0.0f);
    i_ext.assign(count, 0.0f);
    spiked.assign(count, 0);
    root_id.assign(count, 0);
    x.assign(count, 0.0f);
    y.assign(count, 0.0f);
    z.assign(count, 0.0f);
    type.assign(count, 0);
    region.assign(count, 0);
    dopamine.assign(count, 0.0f);
    serotonin.assign(count, 0.0f);
    octopamine.assign(count, 0.0f);
    last_spike_time.assign(count, -1e9f);
  }

  void ClearSynapticInput() {
    std::fill(i_syn.begin(), i_syn.end(), 0.0f);
  }

  // Exponential synaptic current decay: i_syn *= exp(-dt/tau).
  // Call this instead of ClearSynapticInput() for realistic temporal
  // integration. New spikes add on top of the decaying residual,
  // producing alpha-function-like postsynaptic currents.
  // If per-neuron tau_syn is populated, uses per-neuron time constants.
  // Otherwise falls back to the uniform tau_syn_ms.
  void DecaySynapticInput(float dt_ms, float tau_syn_ms) {
    if (!tau_syn.empty()) {
      // Per-neuron tau: use cached decay factors (recompute only on dt change)
      if (dt_ms != decay_syn_cached_dt) {
        decay_syn_cached.resize(tau_syn.size());
        for (size_t i = 0; i < tau_syn.size(); ++i) {
          decay_syn_cached[i] = std::exp(-dt_ms / tau_syn[i]);
        }
        decay_syn_cached_dt = dt_ms;
      }
      float* FWMC_RESTRICT syn = i_syn.data();
      const float* FWMC_RESTRICT decay = decay_syn_cached.data();
#ifdef FWMC_HAS_AVX2
      size_t i = 0;
      for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_loadu_ps(syn + i);
        __m256 dec = _mm256_loadu_ps(decay + i);
        _mm256_storeu_ps(syn + i, _mm256_mul_ps(val, dec));
      }
      for (; i < n; ++i) syn[i] *= decay[i];
#else
      for (size_t i = 0; i < n; ++i) {
        syn[i] *= decay[i];
      }
#endif
    } else {
      // Uniform tau: single exp, then vectorized multiply
      float decay = std::exp(-dt_ms / tau_syn_ms);
#ifdef FWMC_HAS_AVX2
      __m256 vdecay = _mm256_set1_ps(decay);
      size_t i = 0;
      float* FWMC_RESTRICT syn = i_syn.data();
      for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_loadu_ps(syn + i);
        _mm256_storeu_ps(syn + i, _mm256_mul_ps(val, vdecay));
      }
      for (; i < n; ++i) syn[i] *= decay;
#else
      for (size_t i = 0; i < n; ++i) {
        i_syn[i] *= decay;
      }
#endif
    }
  }

  void ClearExternalInput() {
    std::fill(i_ext.begin(), i_ext.end(), 0.0f);
  }

  int CountSpikes() const {
    int count = 0;
    const int nn = static_cast<int>(n);
    #pragma omp parallel for reduction(+:count) schedule(static) if(nn > 10000)
    for (int i = 0; i < nn; ++i) {
      count += spiked[i];
    }
    return count;
  }

  // Collect indices of spiking neurons into a compact list.
  // Use for sparse iteration: at 5% spike rate with 140K neurons,
  // only ~7K entries vs iterating all 140K.
  // Reuses a persistent buffer to avoid per-step allocation.
  mutable std::vector<uint32_t> spiked_indices_;

  int CollectSpikedIndices() const {
    spiked_indices_.clear();
    for (size_t i = 0; i < n; ++i) {
      if (spiked[i]) spiked_indices_.push_back(static_cast<uint32_t>(i));
    }
    return static_cast<int>(spiked_indices_.size());
  }

  const std::vector<uint32_t>& SpikedIndices() const { return spiked_indices_; }
};

}  // namespace mechabrain

#endif  // FWMC_NEURON_ARRAY_H_
