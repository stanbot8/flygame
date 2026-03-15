#ifdef FWMC_CUDA

#include "cuda/kernels.cuh"
#include <cmath>

namespace mechabrain {

// ---------------------------------------------------------------------------
// GPU-side mirror of STDPParams (core/stdp.h).
// Kept as a plain struct so it can be passed by value to the kernel launch
// (ends up in constant memory / kernel arguments, not global).
// ---------------------------------------------------------------------------
struct STDPParamsGPU {
  float a_plus        = 0.01f;
  float a_minus       = 0.012f;
  float tau_plus       = 20.0f;
  float tau_minus      = 20.0f;
  float w_min          = 0.0f;
  float w_max          = 10.0f;
  bool  dopamine_gated = false;
  float da_scale       = 5.0f;
  float window_factor  = 5.0f;  // matches STDPParams::window_factor
};

// ---------------------------------------------------------------------------
// STDP weight update kernel, one thread per synapse.
//
// For every synapse (pre -> post) we check:
//   1. If pre spiked this step:
//      dt = sim_time - last_spike_time[post]
//      If post fired recently (dt > 0, dt < window_factor*tau_minus): depression
//   2. If post spiked this step:
//      dt = sim_time - last_spike_time[pre]
//      If pre fired recently (dt > 0, dt < window_factor*tau_plus): potentiation
//
// Dopamine gating (Izhikevich 2007):
//   dw_effective = dw * (1 + da_scale * dopamine[post])
//
// Weight is clamped to [w_min, w_max] after each update.
//
// To map synapse index → pre-neuron we walk the CSR row_ptr array with a
// binary search.  This is O(log n_neurons) per thread and cheaper than the
// alternative of storing an explicit pre[] array (which would double memory
// bandwidth for the synapse table).
// ---------------------------------------------------------------------------
__device__ int FindPreNeuron(const uint32_t* __restrict__ row_ptr,
                             int n_neurons,
                             uint32_t syn_idx) {
  // Binary search: find largest pre such that row_ptr[pre] <= syn_idx
  int lo = 0;
  int hi = n_neurons;  // row_ptr has n_neurons+1 entries
  while (lo < hi) {
    int mid = lo + (hi - lo + 1) / 2;
    if (row_ptr[mid] <= syn_idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

__global__ void STDPUpdateKernel(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ post,
    float*          __restrict__ weight,
    const uint8_t*  __restrict__ spiked,
    const float*    __restrict__ last_spike_time,
    const float*    __restrict__ dopamine,
    int             n_neurons,
    int             n_synapses,
    float           sim_time_ms,
    float           a_plus,
    float           a_minus,
    float           tau_plus,
    float           tau_minus,
    float           w_min,
    float           w_max,
    bool            dopamine_gated,
    float           da_scale,
    float           window_factor) {

  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_synapses) return;

  // Identify pre- and post-synaptic neurons for this synapse
  int pre      = FindPreNeuron(row_ptr, n_neurons, static_cast<uint32_t>(s));
  uint32_t post_idx = post[s];

  // Dopamine modulation factor
  float da_mod = 1.0f;
  if (dopamine_gated) {
    da_mod = 1.0f + da_scale * dopamine[post_idx];
  }

  float w = weight[s];

  // --- Case 1: pre spiked this step → check for depression ---
  if (spiked[pre]) {
    float dt = sim_time_ms - last_spike_time[post_idx];
    float window = window_factor * tau_minus;
    if (dt > 0.0f && dt < window) {
      // Post fired before pre → depression
      float dw = -a_minus * expf(-dt / tau_minus) * da_mod;
      w = fmaxf(w_min, w + dw);
    }
  }

  // --- Case 2: post spiked this step → check for potentiation ---
  if (spiked[post_idx]) {
    float dt = sim_time_ms - last_spike_time[pre];
    float window = window_factor * tau_plus;
    if (dt > 0.0f && dt < window) {
      // Pre fired before post → potentiation
      float dw = a_plus * expf(-dt / tau_plus) * da_mod;
      w = fminf(w_max, w + dw);
    }
  }

  weight[s] = w;
}

// ---------------------------------------------------------------------------
// Host-side wrapper.
// ---------------------------------------------------------------------------
void STDPUpdateGPU(SynapseTableGPU&       synapses,
                   const NeuronArrayGPU&  neurons,
                   float                  sim_time_ms,
                   const STDPParamsGPU&   p,
                   cudaStream_t           stream) {
  if (synapses.n_synapses == 0) return;

  int n_syn  = static_cast<int>(synapses.n_synapses);
  int blocks = (n_syn + kCudaBlockSize - 1) / kCudaBlockSize;

  STDPUpdateKernel<<<blocks, kCudaBlockSize, 0, stream>>>(
      synapses.d_row_ptr,
      synapses.d_post,
      synapses.d_weight,
      neurons.d_spiked,
      neurons.d_last_spike_time,
      neurons.d_dopamine,
      static_cast<int>(synapses.n_neurons),
      n_syn,
      sim_time_ms,
      p.a_plus,
      p.a_minus,
      p.tau_plus,
      p.tau_minus,
      p.w_min,
      p.w_max,
      p.dopamine_gated,
      p.da_scale,
      p.window_factor);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mechabrain

#endif  // FWMC_CUDA
