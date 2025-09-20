#ifdef FWMC_CUDA

#include "cuda/kernels.cuh"

namespace mechabrain {

// ---------------------------------------------------------------------------
// GPU spike propagation kernel: CSR sparse-matrix × spike-vector multiply.
//
// Each thread handles one pre-synaptic neuron.  If that neuron spiked, the
// thread walks the CSR row [row_ptr[pre], row_ptr[pre+1]) and atomically
// accumulates  sign(nt) * weight * weight_scale  onto every post-synaptic
// neuron's i_syn.
//
// atomicAdd on float is natively supported since Kepler (sm_30).  For
// networks where a single post-synaptic neuron receives thousands of
// inputs in one timestep the contention is modest because spikes are sparse
// (typically <5% of neurons fire per step).
//
// An alternative design (one thread per *synapse*) would need a binary
// search to find the owning row.  The per-row approach is simpler and
// matches the CPU hot loop exactly, making validation trivial.
// ---------------------------------------------------------------------------
__global__ void PropagateSpikesKernel(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ post,
    const float*    __restrict__ weight,
    const uint8_t*  __restrict__ nt_type,
    const uint8_t*  __restrict__ spiked,
    float*          __restrict__ i_syn,
    float           weight_scale,
    int             n_neurons) {

  int pre = blockIdx.x * blockDim.x + threadIdx.x;
  if (pre >= n_neurons) return;

  // Early exit; most neurons are silent in any given timestep
  if (!spiked[pre]) return;

  uint32_t start = row_ptr[pre];
  uint32_t end   = row_ptr[pre + 1];

  for (uint32_t s = start; s < end; ++s) {
    float contribution = NTSign(nt_type[s]) * weight[s] * weight_scale;
    atomicAdd(&i_syn[post[s]], contribution);
  }
}

// ---------------------------------------------------------------------------
// Host-side wrapper.
//
// d_spiked and d_i_syn are device pointers from NeuronArrayGPU.  They are
// passed separately so the caller can clear i_syn independently (e.g. with
// cudaMemsetAsync) without touching the synapse table.
// ---------------------------------------------------------------------------
void PropagateSpikesGPU(const SynapseTableGPU& synapses,
                        const uint8_t*         d_spiked,
                        float*                 d_i_syn,
                        float                  weight_scale,
                        cudaStream_t           stream) {
  if (synapses.n_neurons == 0 || synapses.n_synapses == 0) return;

  int n      = static_cast<int>(synapses.n_neurons);
  int blocks = (n + kCudaBlockSize - 1) / kCudaBlockSize;

  PropagateSpikesKernel<<<blocks, kCudaBlockSize, 0, stream>>>(
      synapses.d_row_ptr,
      synapses.d_post,
      synapses.d_weight,
      synapses.d_nt_type,
      d_spiked,
      d_i_syn,
      weight_scale,
      n);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mechabrain

#endif  // FWMC_CUDA
