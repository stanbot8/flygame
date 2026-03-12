#ifndef FWMC_SYNAPSE_TABLE_H_
#define FWMC_SYNAPSE_TABLE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef FWMC_HAS_AVX2
#include <immintrin.h>
#endif

#include "core/neuron_array.h"

namespace mechabrain {

// Neurotransmitter types (from FlyWire classification, Eckstein et al. 2024)
enum NTType : uint8_t {
  kACh = 0,    // acetylcholine (excitatory)
  kGABA = 1,   // GABA (inhibitory)
  kGlut = 2,   // glutamate (inhibitory via GluCl in Drosophila; Liu & Wilson 2013)
  kDA = 3,     // dopamine (modulatory)
  k5HT = 4,    // serotonin (modulatory)
  kOA = 5,     // octopamine (modulatory)
  kUnknown = 255
};

// Tsodyks-Markram short-term plasticity parameters (per synapse class).
// u: utilization variable (fraction of available resources used per spike)
// x: available resources (fraction of vesicle pool that is ready)
// U_se: baseline utilization (release probability at rest)
// tau_d: depression recovery time constant (ms)
// tau_f: facilitation decay time constant (ms)
//
// Default values from Drosophila NMJ electrophysiology:
//   U_se ~0.5 at 1mM Ca2+ (Hallermann et al. 2010; Kittel et al. 2006)
//   tau_d ~40ms rapid vesicle replenishment (Hallermann et al. 2010)
//   tau_f ~50ms residual Ca2+ clearance
// Mammalian cortex typically uses tau_d=200ms, tau_f=500-1500ms.
struct STPParams {
  float U_se = 0.5f;    // baseline release probability (Drosophila NMJ ~0.5)
  float tau_d = 40.0f;  // depression recovery (ms), Drosophila vesicle replenishment
  float tau_f = 50.0f;  // facilitation decay (ms), residual calcium clearance
};

// CSR (Compressed Sparse Row) synapse storage.
// Sorted by pre-synaptic neuron for cache-friendly spike propagation.
// 50M synapses at 9 bytes each = ~450MB (plus optional STP/release state).
struct SynapseTable {
  // CSR index: neuron i's outgoing synapses are in [row_ptr[i], row_ptr[i+1])
  std::vector<uint32_t> row_ptr;

  // Synapse data (sorted by pre-synaptic neuron)
  std::vector<uint32_t> post;     // post-synaptic neuron index
  std::vector<float> weight;      // synaptic weight
  std::vector<uint8_t> nt_type;   // neurotransmitter type

  // Stochastic release: per-synapse release probability [0,1].
  // If empty, all synapses transmit deterministically (p=1).
  // Biological basis: vesicle fusion probability depends on presynaptic
  // calcium concentration and number of docked vesicles per active zone.
  // KC-PN synapses: p_rel ~ 0.3 (sparse coding mechanism).
  std::vector<float> p_release;

  // Short-term plasticity state (Tsodyks-Markram model).
  // If empty, STP is disabled. When enabled, effective weight is
  // weight * u * x, where u tracks facilitation (residual calcium)
  // and x tracks depression (vesicle pool depletion).
  std::vector<float> stp_u;      // utilization variable (facilitation)
  std::vector<float> stp_x;      // available resources (depression)
  std::vector<float> stp_U_se;   // baseline utilization per synapse
  std::vector<float> stp_tau_d;  // depression recovery time constant
  std::vector<float> stp_tau_f;  // facilitation decay time constant

  // Synaptic delay: per-synapse delay in timesteps.
  // If empty, all synapses transmit instantaneously (delay=0).
  // uint16_t supports up to 65535 steps (6553ms at 0.1ms dt),
  // sufficient for long-range human cortico-cortical projections.
  // Drosophila default: ~1.8ms uniform delay.
  std::vector<uint16_t> delay_steps;

  // Delay ring buffer: per-neuron circular buffer for incoming current.
  // delay_buffer[neuron * ring_size + slot] holds pending current.
  std::vector<float> delay_buffer;
  size_t ring_size = 0;       // slots in ring buffer (max_delay_steps + 1)
  size_t ring_head = 0;       // current read position

  // Cached STP decay alphas: alpha = 1 - exp(-dt/tau), precomputed once
  // per dt change rather than per synapse per timestep.
  std::vector<float> stp_alpha_f;  // cached facilitation decay
  std::vector<float> stp_alpha_d;  // cached depression recovery
  float stp_cached_dt = -1.0f;     // dt for which alphas are cached
  bool stp_uniform_alpha = false;  // true when all synapses share same tau_f/tau_d

  // Eligibility traces for three-factor learning (Izhikevich 2007).
  // Each synapse accumulates a trace from STDP spike pairs. The trace
  // decays exponentially and is converted to weight change when
  // dopamine is present at the postsynaptic neuron.
  std::vector<float> eligibility_trace;

  size_t n_neurons = 0;

  // Persistent thread-local buffer for PropagateSpikes OpenMP reduction.
  // Allocated once (on first use or resize), reused across timesteps.
  // Avoids a large heap allocation (n_threads * n_neurons * 4 bytes)
  // every simulation step.
  mutable std::vector<float> omp_reduce_buf_;
  mutable int omp_reduce_threads_ = 0;

  size_t Size() const { return post.size(); }

  bool HasStochasticRelease() const { return !p_release.empty(); }
  bool HasSTP() const { return !stp_u.empty(); }
  bool HasDelays() const { return !delay_steps.empty(); }
  bool HasEligibilityTraces() const { return !eligibility_trace.empty(); }

  void InitEligibilityTraces() {
    eligibility_trace.assign(post.size(), 0.0f);
  }

  // Sign of synapse based on neurotransmitter.
  // In Drosophila, glutamate acts on GluCl receptors (inhibitory chloride
  // channels) on many postsynaptic targets, making it functionally inhibitory.
  static float Sign(uint8_t nt) {
    return (nt == kGABA || nt == kGlut) ? -1.0f : 1.0f;
  }

  // Initialize stochastic release with a uniform probability for all synapses.
  void InitReleaseProbability(float p) {
    p_release.assign(post.size(), p);
  }

  // Initialize short-term plasticity state for all synapses.
  void InitSTP(const STPParams& params) {
    size_t n = post.size();
    stp_u.assign(n, params.U_se);
    stp_x.assign(n, 1.0f);           // fully recovered at start
    stp_U_se.assign(n, params.U_se);
    stp_tau_d.assign(n, params.tau_d);
    stp_tau_f.assign(n, params.tau_f);
  }

  // Initialize uniform synaptic delay for all synapses.
  // delay_ms: desired delay in milliseconds
  // dt_ms: simulation timestep in milliseconds
  // Drosophila default: 1.8ms (comparable to membrane time constants).
  void InitDelay(float delay_ms, float dt_ms) {
    uint16_t steps = static_cast<uint16_t>(std::max(1.0f, delay_ms / dt_ms));
    delay_steps.assign(post.size(), steps);
    ring_size = static_cast<size_t>(steps) + 1;
    delay_buffer.assign(n_neurons * ring_size, 0.0f);
    ring_head = 0;
  }

  // Axonal conduction velocity constants (in m/s).
  // Convert to position_units/ms with: vel_m_s * scale, where
  // scale = 1e6 for nm positions, 1e3 for um positions.
  //
  // Drosophila unmyelinated: 0.2-1.0 m/s (Tanouye & Wyman 1980, giant fiber ~2 m/s)
  // Mammalian unmyelinated C-fibers: 0.5-2.0 m/s (Waxman & Bennett 1972)
  // Mammalian thin myelinated (intracortical): 1-10 m/s (Swadlow 2000)
  // Mammalian corpus callosum: 5-20 m/s (Ringo et al. 1994)
  // Mammalian thick myelinated (motor): 30-120 m/s (Hursh 1939)
  static constexpr float kVelocityDrosophilaUnmyel = 0.5f;   // m/s
  static constexpr float kVelocityMammalUnmyel     = 1.0f;   // m/s
  static constexpr float kVelocityMammalThinMyel   = 5.0f;   // m/s
  static constexpr float kVelocityCorpusCallosum   = 10.0f;  // m/s
  static constexpr float kVelocityMammalThickMyel  = 50.0f;  // m/s

  // Initialize distance-dependent axonal delays from neuron positions.
  //
  // For each synapse (pre -> post), computes Euclidean distance between
  // pre and post neuron positions, then delay = distance / velocity.
  //
  // velocity: conduction velocity in position_units per millisecond.
  //   To convert from m/s with positions in micrometers: vel_m_s * 1000.0f
  //   To convert from m/s with positions in nanometers:  vel_m_s * 1e6f
  // dt_ms: simulation timestep in milliseconds.
  // min_delay_ms: minimum delay (biological minimum ~0.1ms for adjacent neurons).
  //
  // Biological basis: axonal conduction delay is proportional to distance
  // and inversely proportional to conduction velocity, which depends on
  // axon diameter and myelination (Rushton 1951, Waxman 1980).
  // In Drosophila (~500um brain), delays are ~0.5-2ms.
  // In human brain (~15cm interhemispheric), delays reach 10-100ms.
  void InitDistanceDelay(const float* pre_x, const float* pre_y,
                         const float* pre_z, const float* post_x,
                         const float* post_y, const float* post_z,
                         float velocity, float dt_ms,
                         float min_delay_ms = 0.1f) {
    size_t n_syn = post.size();
    delay_steps.resize(n_syn);
    uint16_t max_steps = 0;

    float inv_vel = (velocity > 0.0f) ? (1.0f / velocity) : 0.0f;
    uint16_t min_steps = static_cast<uint16_t>(
        std::max(1.0f, min_delay_ms / dt_ms));

    // Walk CSR row pointers forward (O(n_syn + n_neurons)) instead of
    // binary searching per synapse (O(n_syn * log(n_neurons))).
    uint32_t pre_idx = 0;
    for (size_t s = 0; s < n_syn; ++s) {
      // Advance pre_idx until synapse s falls within [row_ptr[pre_idx], row_ptr[pre_idx+1])
      while (pre_idx < n_neurons && row_ptr[pre_idx + 1] <= static_cast<uint32_t>(s))
        ++pre_idx;
      uint32_t post_idx = post[s];

      float dx = post_x[post_idx] - pre_x[pre_idx];
      float dy = post_y[post_idx] - pre_y[pre_idx];
      float dz = post_z[post_idx] - pre_z[pre_idx];
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      float delay_ms = dist * inv_vel;

      uint16_t steps = static_cast<uint16_t>(
          std::max(static_cast<float>(min_steps), delay_ms / dt_ms));
      delay_steps[s] = steps;
      if (steps > max_steps) max_steps = steps;
    }

    ring_size = static_cast<size_t>(max_steps) + 1;
    delay_buffer.assign(n_neurons * ring_size, 0.0f);
    ring_head = 0;
  }

  // Convenience overload that reads positions from NeuronArray directly.
  void InitDistanceDelay(const NeuronArray& neurons,
                         float velocity, float dt_ms,
                         float min_delay_ms = 0.1f) {
    InitDistanceDelay(neurons.x.data(), neurons.y.data(), neurons.z.data(),
                      neurons.x.data(), neurons.y.data(), neurons.z.data(),
                      velocity, dt_ms, min_delay_ms);
  }

  // Deliver delayed current: read the current slot, add to i_syn, then zero it.
  // Call this once per timestep BEFORE PropagateSpikes.
  void DeliverDelayed(float* i_syn) {
    if (!HasDelays()) return;
    for (size_t i = 0; i < n_neurons; ++i) {
      size_t slot = i * ring_size + ring_head;
      i_syn[i] += delay_buffer[slot];
      delay_buffer[slot] = 0.0f;
    }
  }

  // Advance the ring buffer head. Call once per timestep AFTER PropagateSpikes.
  void AdvanceDelayRing() {
    if (!HasDelays()) return;
    ring_head = (ring_head + 1) % ring_size;
  }

  // Update STP state for a single synapse when a presynaptic spike arrives.
  // Returns the effective release fraction (u * x) for this spike.
  // Called before delivery; updates u (facilitation) then depletes x (depression).
  float UpdateSTP(size_t s) {
    // Facilitation: residual calcium adds to utilization
    stp_u[s] += stp_U_se[s] * (1.0f - stp_u[s]);
    stp_u[s] = std::clamp(stp_u[s], 0.0f, 1.0f);
    // Effective transmission = u * x (fraction utilized * fraction available)
    float ux = stp_u[s] * stp_x[s];
    // Depression: deplete the vesicle pool
    stp_x[s] = std::max(0.0f, stp_x[s] - ux);
    return ux;
  }

  // Recover STP state between spikes (called once per timestep for all synapses).
  // Uses cached exponential decay alphas: alpha = 1 - exp(-dt/tau).
  // Alphas are precomputed once per dt change (not per synapse per step),
  // eliminating 2*N_synapses exp() calls from the hot path.
  void RecoverSTP(float dt_ms) {
    const size_t n = stp_u.size();
    if (n == 0) return;

    // Recompute alphas only when dt changes (typically never after first call)
    if (dt_ms != stp_cached_dt) {
      stp_alpha_f.resize(n);
      stp_alpha_d.resize(n);
      // Check if all tau values are uniform (common case: InitSTP with single STPParams)
      stp_uniform_alpha = (n > 0);
      for (size_t s = 1; s < n && stp_uniform_alpha; ++s) {
        if (stp_tau_f[s] != stp_tau_f[0] || stp_tau_d[s] != stp_tau_d[0])
          stp_uniform_alpha = false;
      }
      if (stp_uniform_alpha && n > 0) {
        float af = 1.0f - std::exp(-dt_ms / stp_tau_f[0]);
        float ad = 1.0f - std::exp(-dt_ms / stp_tau_d[0]);
        stp_alpha_f[0] = af;
        stp_alpha_d[0] = ad;
      } else {
        for (size_t s = 0; s < n; ++s) {
          stp_alpha_f[s] = 1.0f - std::exp(-dt_ms / stp_tau_f[s]);
          stp_alpha_d[s] = 1.0f - std::exp(-dt_ms / stp_tau_d[s]);
        }
      }
      stp_cached_dt = dt_ms;
    }

    // Hot loop: update all synapses' STP state.
    // Uniform alpha path: broadcast single alpha values (saves 2 array loads per iteration).
    // Non-uniform: load per-synapse alphas from arrays.
#ifdef FWMC_HAS_AVX2
    {
      float* FWMC_RESTRICT u = stp_u.data();
      float* FWMC_RESTRICT x = stp_x.data();
      const float* FWMC_RESTRICT U_se = stp_U_se.data();
      const __m256 ones = _mm256_set1_ps(1.0f);

      if (stp_uniform_alpha) {
        const float af_s = stp_alpha_f[0], ad_s = stp_alpha_d[0];
        const __m256 vaf = _mm256_set1_ps(af_s);
        const __m256 vad = _mm256_set1_ps(ad_s);
        size_t s = 0;
        for (; s + 7 < n; s += 8) {
          __m256 vu = _mm256_loadu_ps(u + s);
          __m256 vU = _mm256_loadu_ps(U_se + s);
          vu = _mm256_fmadd_ps(_mm256_sub_ps(vU, vu), vaf, vu);
          _mm256_storeu_ps(u + s, vu);
          __m256 vx = _mm256_loadu_ps(x + s);
          vx = _mm256_fmadd_ps(_mm256_sub_ps(ones, vx), vad, vx);
          _mm256_storeu_ps(x + s, vx);
        }
        for (; s < n; ++s) {
          u[s] += (U_se[s] - u[s]) * af_s;
          x[s] += (1.0f - x[s]) * ad_s;
        }
      } else {
        const float* FWMC_RESTRICT af = stp_alpha_f.data();
        const float* FWMC_RESTRICT ad = stp_alpha_d.data();
        size_t s = 0;
        for (; s + 7 < n; s += 8) {
          __m256 vu = _mm256_loadu_ps(u + s);
          __m256 vU = _mm256_loadu_ps(U_se + s);
          __m256 vaf = _mm256_loadu_ps(af + s);
          vu = _mm256_fmadd_ps(_mm256_sub_ps(vU, vu), vaf, vu);
          _mm256_storeu_ps(u + s, vu);
          __m256 vx = _mm256_loadu_ps(x + s);
          __m256 vad = _mm256_loadu_ps(ad + s);
          vx = _mm256_fmadd_ps(_mm256_sub_ps(ones, vx), vad, vx);
          _mm256_storeu_ps(x + s, vx);
        }
        for (; s < n; ++s) {
          u[s] += (U_se[s] - u[s]) * af[s];
          x[s] += (1.0f - x[s]) * ad[s];
        }
      }
    }
#else
    if (stp_uniform_alpha) {
      float af = stp_alpha_f[0], ad = stp_alpha_d[0];
      for (size_t s = 0; s < n; ++s) {
        stp_u[s] += (stp_U_se[s] - stp_u[s]) * af;
        stp_x[s] += (1.0f - stp_x[s]) * ad;
      }
    } else {
      for (size_t s = 0; s < n; ++s) {
        stp_u[s] += (stp_U_se[s] - stp_u[s]) * stp_alpha_f[s];
        stp_x[s] += (1.0f - stp_x[s]) * stp_alpha_d[s];
      }
    }
#endif
  }

  // Build CSR from unsorted COO (coordinate) input.
  // After this call, synapses are sorted by pre-neuron for fast traversal.
  void BuildFromCOO(size_t num_neurons,
                    const std::vector<uint32_t>& pre_in,
                    const std::vector<uint32_t>& post_in,
                    const std::vector<float>& weight_in,
                    const std::vector<uint8_t>& nt_in) {
    BuildFromCOOImpl(num_neurons, pre_in, post_in, weight_in, nt_in, {});
  }

  // Extended BuildFromCOO that also reorders per-synapse release probabilities.
  void BuildFromCOO(size_t num_neurons,
                    const std::vector<uint32_t>& pre_in,
                    const std::vector<uint32_t>& post_in,
                    const std::vector<float>& weight_in,
                    const std::vector<uint8_t>& nt_in,
                    const std::vector<float>& p_release_in) {
    BuildFromCOOImpl(num_neurons, pre_in, post_in, weight_in, nt_in, p_release_in);
  }

 private:
  // Shared implementation for both BuildFromCOO overloads.
  // Computes the sort order once and reuses it for all column arrays.
  void BuildFromCOOImpl(size_t num_neurons,
                        const std::vector<uint32_t>& pre_in,
                        const std::vector<uint32_t>& post_in,
                        const std::vector<float>& weight_in,
                        const std::vector<uint8_t>& nt_in,
                        const std::vector<float>& p_release_in) {
    n_neurons = num_neurons;
    size_t nnz = pre_in.size();

    // Validate indices before allocating anything
    for (size_t i = 0; i < nnz; ++i) {
      if (pre_in[i] >= num_neurons || post_in[i] >= num_neurons) {
        post.clear(); weight.clear(); nt_type.clear(); p_release.clear();
        row_ptr.assign(num_neurons + 1, 0);
        return;
      }
    }

    // Sort by pre-synaptic index (single O(n log n) pass)
    std::vector<size_t> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](size_t a, size_t b) { return pre_in[a] < pre_in[b]; });

    post.resize(nnz);
    weight.resize(nnz);
    nt_type.resize(nnz);
    for (size_t i = 0; i < nnz; ++i) {
      post[i]   = post_in[order[i]];
      weight[i] = weight_in[order[i]];
      nt_type[i] = nt_in[order[i]];
    }

    if (!p_release_in.empty()) {
      p_release.resize(nnz);
      for (size_t i = 0; i < nnz; ++i) {
        p_release[i] = p_release_in[order[i]];
      }
    }

    // Build row pointers
    row_ptr.assign(num_neurons + 1, 0);
    for (size_t i = 0; i < nnz; ++i) {
      row_ptr[pre_in[order[i]] + 1]++;
    }
    for (size_t i = 1; i <= num_neurons; ++i) {
      row_ptr[i] += row_ptr[i - 1];
    }
  }

 public:
  // Propagate spikes: for each neuron that spiked, deliver weighted
  // current to all post-synaptic targets.
  // This is the hot loop. CSR layout means sequential memory access
  // for each pre-neuron's outgoing synapses.
  //
  // OpenMP parallelization: each pre-neuron's outgoing synapses are
  // independent reads, but multiple pre-neurons can target the same
  // post-neuron (write conflict on i_syn). We use thread-local
  // accumulation buffers with a final reduction pass to avoid atomic
  // writes entirely. Each thread accumulates into its own buffer,
  // then buffers are summed into i_syn. This eliminates the lock
  // cmpxchg loop that atomic float adds compile to on x86.
  //
  // Memory cost: n_neurons * sizeof(float) per thread (~560KB for 140K
  // neurons). Fits easily in L3 cache per core.
  //
  // For delayed synapses, atomics are retained since the delay ring
  // buffer is too large for per-thread duplication.
  void PropagateSpikes(const uint8_t* spiked, float* i_syn,
                       float weight_scale) {
    const bool has_delay = HasDelays();
    const bool has_stp = HasSTP();
    const int n = static_cast<int>(n_neurons);
    // STP state (stp_u, stp_x) is per-synapse and each synapse belongs to
    // exactly one pre-neuron in CSR order, so UpdateSTP(s) has no write
    // conflicts when parallelized by pre-neuron.

    if (has_delay) {
      // Delay path: atomics on ring buffer (too large for thread-local copy)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
      #endif
      for (int pre = 0; pre < n; ++pre) {
        if (!spiked[pre]) continue;
        uint32_t start = row_ptr[pre];
        uint32_t end = row_ptr[pre + 1];
        for (uint32_t s = start; s < end; ++s) {
          float stp_factor = 1.0f;
          if (has_stp) stp_factor = UpdateSTP(s);
          float val = Sign(nt_type[s]) * weight[s] * weight_scale * stp_factor;
          size_t future = (ring_head + delay_steps[s]) % ring_size;
          size_t slot = post[s] * ring_size + future;
          #ifdef _OPENMP
          #pragma omp atomic
          #endif
          delay_buffer[slot] += val;
        }
      }
    } else {
      // No-delay path: thread-local reduction (no atomics)
      #ifdef _OPENMP
      if (n > 10000) {
        int n_threads = 1;
#pragma omp parallel
        {
#pragma omp single
          n_threads = omp_get_num_threads();
        }
        // Reuse persistent buffer (allocated once, zeroed per call)
        size_t buf_size = static_cast<size_t>(n_threads) * n_neurons;
        if (omp_reduce_threads_ != n_threads ||
            omp_reduce_buf_.size() < buf_size) {
          omp_reduce_buf_.resize(buf_size, 0.0f);
          omp_reduce_threads_ = n_threads;
        } else {
          std::fill(omp_reduce_buf_.begin(),
                    omp_reduce_buf_.begin() + static_cast<ptrdiff_t>(buf_size),
                    0.0f);
        }

        #pragma omp parallel num_threads(n_threads)
        {
          int tid = omp_get_thread_num();
          float* local = omp_reduce_buf_.data() + static_cast<size_t>(tid) * n_neurons;

          #pragma omp for schedule(dynamic, 64)
          for (int pre = 0; pre < n; ++pre) {
            if (!spiked[pre]) continue;
            uint32_t start = row_ptr[pre];
            uint32_t end = row_ptr[pre + 1];
            for (uint32_t s = start; s < end; ++s) {
              float stp_factor = 1.0f;
              if (has_stp) stp_factor = UpdateSTP(s);
              float val = Sign(nt_type[s]) * weight[s] * weight_scale * stp_factor;
              local[post[s]] += val;
            }
          }

          // Reduction: each thread adds its buffer into i_syn
          // Partition neurons across threads for parallel reduction
          #pragma omp for schedule(static)
          for (int i = 0; i < n; ++i) {
            for (int t = 0; t < n_threads; ++t) {
              i_syn[i] += omp_reduce_buf_[static_cast<size_t>(t) * n_neurons + i];
            }
          }
        }
      } else
      #endif
      {
        // Single-threaded fallback
        for (int pre = 0; pre < n; ++pre) {
          if (!spiked[pre]) continue;
          uint32_t start = row_ptr[pre];
          uint32_t end = row_ptr[pre + 1];
          for (uint32_t s = start; s < end; ++s) {
            float stp_factor = 1.0f;
            if (has_stp) stp_factor = UpdateSTP(s);
            float val = Sign(nt_type[s]) * weight[s] * weight_scale * stp_factor;
            i_syn[post[s]] += val;
          }
        }
      }
    }
  }

  // Stochastic spike propagation: each synapse transmits with probability
  // p_release[s]. If STP is enabled, the effective weight is further
  // modulated by the utilization-resource product (u * x).
  // STP state is updated for ALL synapses of a spiking neuron (the
  // presynaptic spike triggers facilitation/depression regardless of
  // whether the vesicle was released). The stochastic gate only controls
  // whether current is delivered to the postsynaptic neuron.
  // rng must be thread-local or externally synchronized.
  void PropagateSpikesMonteCarlo(const uint8_t* spiked, float* i_syn,
                                  float weight_scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    const bool has_stp = HasSTP();
    const int n = static_cast<int>(n_neurons);

    for (int pre = 0; pre < n; ++pre) {
      if (!spiked[pre]) continue;
      uint32_t start = row_ptr[pre];
      uint32_t end = row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        // STP state update happens for every synapse of a spiking neuron
        float stp_factor = 1.0f;
        if (has_stp) {
          stp_factor = UpdateSTP(s);
        }

        // Stochastic release gate (only gates current delivery)
        if (!p_release.empty() && coin(rng) >= p_release[s]) continue;

        float val = Sign(nt_type[s]) * weight[s] * weight_scale * stp_factor;
        i_syn[post[s]] += val;
      }
    }
  }

  // Per-NT synaptic time constants (ms).
  // ACh: ~2ms (fast nicotinic, Wilson & Laurent 2005)
  // GABA: ~5ms (GABAa Cl- channels, slower kinetics)
  // Glut: ~3ms (GluCl in Drosophila, moderate)
  // DA/5HT/OA: ~10ms (volume transmission, G-protein coupled, slower)
  static float TauForNT(uint8_t nt) {
    switch (nt) {
      case kACh:  return 2.0f;
      case kGABA: return 5.0f;
      case kGlut: return 3.0f;
      case kDA:   return 10.0f;
      case k5HT:  return 10.0f;
      case kOA:   return 10.0f;
      default:    return 3.0f;
    }
  }

  // Compute per-neuron synaptic time constants from dominant input NT type.
  // For each postsynaptic neuron, tallies incoming synapse count by NT,
  // picks the most frequent, and assigns that NT's tau.
  void AssignPerNeuronTau(NeuronArray& neurons) const {
    neurons.tau_syn.assign(neurons.n, 3.0f);  // default
    if (post.empty()) return;

    // Count input synapses per NT for each neuron
    // Using 6 NT types
    std::vector<std::array<int, 6>> counts(neurons.n);
    for (auto& c : counts) c.fill(0);

    for (size_t s = 0; s < post.size(); ++s) {
      uint32_t target = post[s];
      uint8_t nt = nt_type[s];
      if (target < neurons.n && nt < 6) {
        counts[target][nt]++;
      }
    }

    for (size_t i = 0; i < neurons.n; ++i) {
      int best_count = 0;
      uint8_t best_nt = kACh;
      for (uint8_t nt = 0; nt < 6; ++nt) {
        if (counts[i][nt] > best_count) {
          best_count = counts[i][nt];
          best_nt = nt;
        }
      }
      if (best_count > 0) {
        neurons.tau_syn[i] = TauForNT(best_nt);
      }
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_SYNAPSE_TABLE_H_
