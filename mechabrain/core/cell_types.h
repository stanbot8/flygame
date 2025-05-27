#ifndef FWMC_CELL_TYPES_H_
#define FWMC_CELL_TYPES_H_

#include <unordered_map>
#include <vector>
#include "core/cell_type_defs.h"
#include "core/izhikevich.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Manages per-neuron Izhikevich parameters based on cell type assignments.
// Supports config-driven overrides: load defaults from ParamsForCellType(),
// then override individual params from a config map.
struct CellTypeManager {
  // Per-neuron params (indexed by neuron index) -- AoS for convenience
  std::vector<IzhikevichParams> neuron_params;

  // SoA layout for SIMD: separate arrays for each parameter.
  // Populated by AssignFromTypes(). Enables AVX2 heterogeneous stepping.
  std::vector<float> pa, pb, pc, pd, pv_thresh, prefract;

  // Custom overrides: CellType -> IzhikevichParams
  std::unordered_map<uint8_t, IzhikevichParams> overrides;

  // Assign default params to all neurons based on their type field.
  // Populates both AoS and SoA layouts.
  void AssignFromTypes(const NeuronArray& neurons) {
    neuron_params.resize(neurons.n);
    pa.resize(neurons.n);
    pb.resize(neurons.n);
    pc.resize(neurons.n);
    pd.resize(neurons.n);
    pv_thresh.resize(neurons.n);
    prefract.resize(neurons.n);
    for (size_t i = 0; i < neurons.n; ++i) {
      auto ct = static_cast<CellType>(neurons.type[i]);
      auto it = overrides.find(neurons.type[i]);
      if (it != overrides.end()) {
        neuron_params[i] = it->second;
      } else {
        neuron_params[i] = ParamsForCellType(ct);
      }
      pa[i] = neuron_params[i].a;
      pb[i] = neuron_params[i].b;
      pc[i] = neuron_params[i].c;
      pd[i] = neuron_params[i].d;
      pv_thresh[i] = neuron_params[i].v_thresh;
      prefract[i] = neuron_params[i].refractory_ms;
    }
  }

  // Get params for a single neuron.
  const IzhikevichParams& Get(size_t idx) const {
    return neuron_params[idx];
  }

  // Override params for a cell type. Call AssignFromTypes() after to apply.
  void SetOverride(CellType ct, const IzhikevichParams& p) {
    overrides[static_cast<uint8_t>(ct)] = p;
  }
};

// Step neurons with per-neuron params (heterogeneous dynamics).
// Scalar fallback version. Biologically richer than uniform-param stepping.
inline void IzhikevichStepHeterogeneous(NeuronArray& neurons, float dt_ms,
                                         float sim_time_ms,
                                         const CellTypeManager& types) {
  if (types.neuron_params.size() < neurons.n) return;
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  float* FWMC_RESTRICT u = neurons.u.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(n > 10000)
  #endif
  for (int i = 0; i < n; ++i) {
    const auto& p = types.neuron_params[static_cast<size_t>(i)];
    float vi = v[i];
    float ui = u[i];
    float I = i_syn[i] + i_ext[i];

    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    ui += dt_ms * p.a * (p.b * vi - ui);

    if (!std::isfinite(vi) || vi > 100.0f) vi = p.c;
    if (!std::isfinite(ui) || std::abs(ui) > 1e6f) ui = p.b * p.c;

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

// AVX2-vectorized heterogeneous Izhikevich step: per-neuron a,b,c,d from SoA.
// Processes 8 neurons per iteration using the CellTypeManager SoA arrays.
// ~4-6x faster than scalar IzhikevichStepHeterogeneous.
#ifdef FWMC_HAS_AVX2
inline void IzhikevichStepHeterogeneousAVX2(NeuronArray& neurons, float dt_ms,
                                              float sim_time_ms,
                                              const CellTypeManager& types) {
  if (types.pa.size() < neurons.n) return;
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  float* FWMC_RESTRICT u = neurons.u.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();
  const float* FWMC_RESTRICT a_arr = types.pa.data();
  const float* FWMC_RESTRICT b_arr = types.pb.data();
  const float* FWMC_RESTRICT c_arr = types.pc.data();
  const float* FWMC_RESTRICT d_arr = types.pd.data();
  const float* FWMC_RESTRICT thresh_arr = types.pv_thresh.data();
  const float* FWMC_RESTRICT refract_arr = types.prefract.data();

  const __m256 v_half_dt = _mm256_set1_ps(0.5f * dt_ms);
  const __m256 v_dt = _mm256_set1_ps(dt_ms);
  const __m256 v_004 = _mm256_set1_ps(0.04f);
  const __m256 v_5 = _mm256_set1_ps(5.0f);
  const __m256 v_140 = _mm256_set1_ps(140.0f);
  const __m256 v_clamp = _mm256_set1_ps(100.0f);
  const __m256 v_sim_time = _mm256_set1_ps(sim_time_ms);

  int i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 vi = _mm256_loadu_ps(v + i);
    __m256 ui = _mm256_loadu_ps(u + i);
    __m256 isyn = _mm256_loadu_ps(i_syn + i);
    __m256 iext = _mm256_loadu_ps(i_ext + i);
    __m256 ls = _mm256_loadu_ps(last_spike + i);

    // Load per-neuron params from SoA
    __m256 va = _mm256_loadu_ps(a_arr + i);
    __m256 vb = _mm256_loadu_ps(b_arr + i);
    __m256 vc = _mm256_loadu_ps(c_arr + i);
    __m256 vd = _mm256_loadu_ps(d_arr + i);
    __m256 vthresh = _mm256_loadu_ps(thresh_arr + i);
    __m256 vrefract = _mm256_loadu_ps(refract_arr + i);

    __m256 I = _mm256_add_ps(isyn, iext);

    // Two half-steps: vi += 0.5*dt*(0.04*vi*vi + 5*vi + 140 - ui + I)
    for (int step = 0; step < 2; ++step) {
      __m256 vi2 = _mm256_mul_ps(vi, vi);
      __m256 dv = _mm256_mul_ps(v_004, vi2);
      dv = _mm256_fmadd_ps(v_5, vi, dv);
      dv = _mm256_add_ps(dv, v_140);
      dv = _mm256_sub_ps(dv, ui);
      dv = _mm256_add_ps(dv, I);
      vi = _mm256_fmadd_ps(v_half_dt, dv, vi);
    }

    // ui += dt * a * (b*vi - ui)
    __m256 bv = _mm256_mul_ps(vb, vi);
    __m256 du = _mm256_sub_ps(bv, ui);
    du = _mm256_mul_ps(va, du);
    ui = _mm256_fmadd_ps(v_dt, du, ui);

    // Clamp: if vi > 100, reset to per-neuron c
    __m256 clamp_mask = _mm256_cmp_ps(vi, v_clamp, _CMP_GT_OQ);
    vi = _mm256_blendv_ps(vi, vc, clamp_mask);
    ui = _mm256_blendv_ps(ui, _mm256_mul_ps(vb, vc), clamp_mask);

    // Refractory check
    __m256 elapsed = _mm256_sub_ps(v_sim_time, ls);
    __m256 not_refractory = _mm256_cmp_ps(elapsed, vrefract, _CMP_GE_OQ);

    // Spike check
    __m256 above_thresh = _mm256_cmp_ps(vi, vthresh, _CMP_GE_OQ);
    __m256 fire_mask = _mm256_and_ps(above_thresh, not_refractory);

    // Apply spike reset with per-neuron c, d
    vi = _mm256_blendv_ps(vi, vc, fire_mask);
    ui = _mm256_blendv_ps(ui, _mm256_add_ps(ui, vd), fire_mask);
    ls = _mm256_blendv_ps(ls, v_sim_time, fire_mask);

    _mm256_storeu_ps(v + i, vi);
    _mm256_storeu_ps(u + i, ui);
    _mm256_storeu_ps(last_spike + i, ls);

    int mask = _mm256_movemask_ps(fire_mask);
    for (int k = 0; k < 8; ++k)
      spiked[i + k] = (mask >> k) & 1;
  }

  // Scalar tail
  for (; i < n; ++i) {
    const auto& p = types.neuron_params[static_cast<size_t>(i)];
    float vi_s = v[i], ui_s = u[i];
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

// Dispatch: AVX2 heterogeneous if available, otherwise scalar.
inline void IzhikevichStepHeterogeneousFast(NeuronArray& neurons, float dt_ms,
                                              float sim_time_ms,
                                              const CellTypeManager& types) {
#ifdef FWMC_HAS_AVX2
  IzhikevichStepHeterogeneousAVX2(neurons, dt_ms, sim_time_ms, types);
#else
  IzhikevichStepHeterogeneous(neurons, dt_ms, sim_time_ms, types);
#endif
}

}  // namespace mechabrain

#endif  // FWMC_CELL_TYPES_H_
