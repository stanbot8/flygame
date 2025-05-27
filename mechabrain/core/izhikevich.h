#ifndef FWMC_IZHIKEVICH_H_
#define FWMC_IZHIKEVICH_H_

#include <cmath>
#include "core/neuron_array.h"
#include "core/platform.h"

#ifdef FWMC_HAS_AVX2
  #include <immintrin.h>
#elif defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace mechabrain {

// Izhikevich parameters per neuron type.
// Different cell types in the fly brain have different dynamics:
//   Regular spiking (most Kenyon cells): a=0.02, b=0.2, c=-65, d=8
//   Fast spiking (GABAergic interneurons): a=0.1, b=0.2, c=-65, d=2
//   Bursting (some projection neurons): a=0.02, b=0.2, c=-50, d=2
struct IzhikevichParams {
  float a = 0.02f, b = 0.2f, c = -65.0f, d = 8.0f;
  float v_thresh = 30.0f;
  float refractory_ms = 2.0f;  // absolute refractory period
  float tau_syn_ms = 3.0f;     // synaptic current decay time constant
                                // 0 = delta synapses (instantaneous, clear each step)
                                // >0 = exponential decay (e.g. 3ms ACh, 5ms GABA)
                                // Default 3ms matches fast cholinergic (ACh) kinetics
                                // in Drosophila (Wilson & Laurent 2005)
};

// Step all neurons forward by dt_ms using Izhikevich dynamics.
// Pure function over flat arrays. No virtual dispatch, no indirection.
// Suitable for SIMD vectorization or direct port to CUDA kernel.
inline void IzhikevichStep(NeuronArray& neurons, float dt_ms,
                           float sim_time_ms, const IzhikevichParams& p) {
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  float* FWMC_RESTRICT u = neurons.u.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();

  // Each neuron is independent (embarrassingly parallel)
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(n > 10000)
  #endif
  for (int i = 0; i < n; ++i) {
    float vi = v[i];
    float ui = u[i];
    float I = i_syn[i] + i_ext[i];

    // Two half-steps for numerical stability (Izhikevich 2003)
    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    ui += dt_ms * p.a * (p.b * vi - ui);

    // Clamp runaway voltages and reset divergent neurons
    if (!std::isfinite(vi) || vi > 100.0f) vi = p.c;
    if (!std::isfinite(ui) || std::abs(ui) > 1e6f) ui = p.b * p.c;

    // Absolute refractory period: neuron cannot fire again too soon
    bool in_refractory = (sim_time_ms - last_spike[i]) < p.refractory_ms;
    uint8_t fired = (!in_refractory && vi >= p.v_thresh) ? 1 : 0;
    if (fired) {
      vi = p.c;
      ui += p.d;
      last_spike[i] = sim_time_ms;
    }

    v[i] = vi;
    u[i] = ui;
    spiked[i] = fired;
  }
}

// LIF alternative for large-scale runs where individual dynamics matter less
struct LIFParams {
  float tau_ms = 20.0f;
  float v_rest = -70.0f;
  float v_thresh = -55.0f;
  float v_reset = -70.0f;
  float r_membrane = 10.0f;
  float refractory_ms = 2.0f;  // absolute refractory period
  float tau_syn_ms = 0.0f;     // synaptic current decay (0 = delta)
};

inline void LIFStep(NeuronArray& neurons, float dt_ms,
                    float sim_time_ms, const LIFParams& p) {
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(n > 10000)
  #endif
  for (int i = 0; i < n; ++i) {
    float vi = v[i];
    float I = i_syn[i] + i_ext[i];
    vi += dt_ms * (-(vi - p.v_rest) + p.r_membrane * I) / p.tau_ms;

    // Clamp runaway voltages
    if (!std::isfinite(vi) || vi > 100.0f) vi = p.v_reset;

    bool in_refractory = (sim_time_ms - last_spike[i]) < p.refractory_ms;
    uint8_t fired = (!in_refractory && vi >= p.v_thresh) ? 1 : 0;
    if (fired) {
      vi = p.v_reset;
      last_spike[i] = sim_time_ms;
    }

    v[i] = vi;
    spiked[i] = fired;
  }
}

// AVX2-vectorized Izhikevich step: processes 8 neurons per iteration.
// Falls back to scalar IzhikevichStep when AVX2 is not available.
// ~4-6x faster than scalar on the neuron update (not including spike propagation).
#ifdef FWMC_HAS_AVX2
inline void IzhikevichStepAVX2(NeuronArray& neurons, float dt_ms,
                                float sim_time_ms, const IzhikevichParams& p) {
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  float* FWMC_RESTRICT u = neurons.u.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();

  // Broadcast constants
  const __m256 v_dt = _mm256_set1_ps(dt_ms);
  const __m256 v_half_dt = _mm256_set1_ps(0.5f * dt_ms);
  const __m256 v_004 = _mm256_set1_ps(0.04f);
  const __m256 v_5 = _mm256_set1_ps(5.0f);
  const __m256 v_140 = _mm256_set1_ps(140.0f);
  const __m256 v_a = _mm256_set1_ps(p.a);
  const __m256 v_b = _mm256_set1_ps(p.b);
  const __m256 v_c = _mm256_set1_ps(p.c);
  const __m256 v_d = _mm256_set1_ps(p.d);
  const __m256 v_thresh = _mm256_set1_ps(p.v_thresh);
  const __m256 v_clamp = _mm256_set1_ps(100.0f);
  const __m256 v_sim_time = _mm256_set1_ps(sim_time_ms);
  const __m256 v_refract = _mm256_set1_ps(p.refractory_ms);

  // Process 8 neurons at a time
  int i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 vi = _mm256_loadu_ps(v + i);
    __m256 ui = _mm256_loadu_ps(u + i);
    __m256 isyn = _mm256_loadu_ps(i_syn + i);
    __m256 iext = _mm256_loadu_ps(i_ext + i);
    __m256 ls = _mm256_loadu_ps(last_spike + i);

    __m256 I = _mm256_add_ps(isyn, iext);

    // Two half-steps: vi += 0.5*dt*(0.04*vi*vi + 5*vi + 140 - ui + I)
    auto half_step = [&]() {
      __m256 vi2 = _mm256_mul_ps(vi, vi);
      __m256 dv = _mm256_mul_ps(v_004, vi2);           // 0.04*vi*vi
      dv = _mm256_fmadd_ps(v_5, vi, dv);               // + 5*vi
      dv = _mm256_add_ps(dv, v_140);                    // + 140
      dv = _mm256_sub_ps(dv, ui);                       // - ui
      dv = _mm256_add_ps(dv, I);                        // + I
      vi = _mm256_fmadd_ps(v_half_dt, dv, vi);          // vi += 0.5*dt*dv
    };
    half_step();
    half_step();

    // ui += dt * a * (b*vi - ui)
    __m256 bv = _mm256_mul_ps(v_b, vi);
    __m256 du = _mm256_sub_ps(bv, ui);
    du = _mm256_mul_ps(v_a, du);
    ui = _mm256_fmadd_ps(v_dt, du, ui);

    // Clamp: if vi > 100 or NaN/Inf, reset to c
    // NaN comparison: vi != vi is true for NaN (unordered comparison)
    __m256 nan_v = _mm256_cmp_ps(vi, vi, _CMP_UNORD_Q);
    __m256 clamp_mask = _mm256_or_ps(
        _mm256_cmp_ps(vi, v_clamp, _CMP_GT_OQ), nan_v);
    vi = _mm256_blendv_ps(vi, v_c, clamp_mask);
    // Also reset ui if vi was clamped, or if ui itself is NaN
    __m256 nan_u = _mm256_cmp_ps(ui, ui, _CMP_UNORD_Q);
    __m256 u_reset_mask = _mm256_or_ps(clamp_mask, nan_u);
    ui = _mm256_blendv_ps(ui, _mm256_mul_ps(v_b, v_c), u_reset_mask);

    // Refractory check: (sim_time - last_spike) >= refractory_ms
    __m256 elapsed = _mm256_sub_ps(v_sim_time, ls);
    __m256 not_refractory = _mm256_cmp_ps(elapsed, v_refract, _CMP_GE_OQ);

    // Spike check: vi >= v_thresh AND not refractory
    __m256 above_thresh = _mm256_cmp_ps(vi, v_thresh, _CMP_GE_OQ);
    __m256 fire_mask = _mm256_and_ps(above_thresh, not_refractory);

    // Apply spike reset: if fired, vi = c, ui += d, last_spike = sim_time
    vi = _mm256_blendv_ps(vi, v_c, fire_mask);
    ui = _mm256_blendv_ps(ui, _mm256_add_ps(ui, v_d), fire_mask);
    ls = _mm256_blendv_ps(ls, v_sim_time, fire_mask);

    _mm256_storeu_ps(v + i, vi);
    _mm256_storeu_ps(u + i, ui);
    _mm256_storeu_ps(last_spike + i, ls);

    // Extract spike flags to uint8_t (movemask gives 8 bits)
    int mask = _mm256_movemask_ps(fire_mask);
    for (int k = 0; k < 8; ++k)
      spiked[i + k] = (mask >> k) & 1;
  }

  // Scalar tail for remaining neurons
  for (; i < n; ++i) {
    float vi_s = v[i];
    float ui_s = u[i];
    float I_s = i_syn[i] + i_ext[i];
    vi_s += 0.5f * dt_ms * (0.04f * vi_s * vi_s + 5.0f * vi_s + 140.0f - ui_s + I_s);
    vi_s += 0.5f * dt_ms * (0.04f * vi_s * vi_s + 5.0f * vi_s + 140.0f - ui_s + I_s);
    ui_s += dt_ms * p.a * (p.b * vi_s - ui_s);
    if (!std::isfinite(vi_s) || vi_s > 100.0f) vi_s = p.c;
    if (!std::isfinite(ui_s) || std::abs(ui_s) > 1e6f) ui_s = p.b * p.c;
    bool in_refract = (sim_time_ms - last_spike[i]) < p.refractory_ms;
    uint8_t fired = (!in_refract && vi_s >= p.v_thresh) ? 1 : 0;
    if (fired) { vi_s = p.c; ui_s += p.d; last_spike[i] = sim_time_ms; }
    v[i] = vi_s; u[i] = ui_s; spiked[i] = fired;
  }
}
#endif  // FWMC_HAS_AVX2

// Dispatch: use AVX2 if available, otherwise scalar.
inline void IzhikevichStepFast(NeuronArray& neurons, float dt_ms,
                                float sim_time_ms, const IzhikevichParams& p) {
#ifdef FWMC_HAS_AVX2
  IzhikevichStepAVX2(neurons, dt_ms, sim_time_ms, p);
#else
  IzhikevichStep(neurons, dt_ms, sim_time_ms, p);
#endif
}

}  // namespace mechabrain

#endif  // FWMC_IZHIKEVICH_H_
