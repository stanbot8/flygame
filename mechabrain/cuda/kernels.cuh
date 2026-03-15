#ifndef FWMC_CUDA_KERNELS_CUH_
#define FWMC_CUDA_KERNELS_CUH_

#ifdef FWMC_CUDA

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace mechabrain {

// ---------------------------------------------------------------------------
// Error-checking macro.  Wraps every CUDA API call; prints file/line on error
// and aborts.  In release builds the message is terse but still fatal.
// silently swallowing GPU errors leads to impossible-to-debug corruption.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n  %s\n", __FILE__,      \
                   __LINE__, cudaGetErrorName(err_),                           \
                   cudaGetErrorString(err_));                                   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Host / device annotation helpers.  These mirror the pattern used by cuBLAS
// and cuDNN headers so call-site code reads the same regardless of compiler.
// ---------------------------------------------------------------------------
#define FWMC_HOST       __host__
#define FWMC_DEVICE     __device__
#define FWMC_HOST_DEVICE __host__ __device__

// Default CUDA block size for 1-D kernels.  256 is a safe choice across
// all architectures from Maxwell (sm_50) through Hopper (sm_90).
constexpr int kCudaBlockSize = 256;

// ---------------------------------------------------------------------------
// GPU neuron storage: flat device pointers mirroring NeuronArray's SoA layout.
// Only the fields required by the three hot-path kernels are kept on-device;
// metadata (root_id, x/y/z) stays on the host.
// ---------------------------------------------------------------------------
struct NeuronArrayGPU {
  size_t n = 0;

  // Izhikevich state
  float*    d_v               = nullptr;  // membrane potential  (mV)
  float*    d_u               = nullptr;  // recovery variable
  float*    d_i_syn           = nullptr;  // synaptic input current
  float*    d_i_ext           = nullptr;  // external stimulus current
  uint8_t*  d_spiked          = nullptr;  // 1 if neuron fired this step

  // Neuromodulators
  float*    d_dopamine        = nullptr;
  float*    d_serotonin       = nullptr;
  float*    d_octopamine      = nullptr;

  // Metadata needed by STDP / spike propagation
  uint8_t*  d_type            = nullptr;
  uint8_t*  d_region          = nullptr;
  float*    d_last_spike_time = nullptr;
};

// ---------------------------------------------------------------------------
// GPU synapse storage: CSR format, device pointers.
// Mirrors SynapseTable on the host side.
// ---------------------------------------------------------------------------
struct SynapseTableGPU {
  size_t n_neurons  = 0;
  size_t n_synapses = 0;

  uint32_t* d_row_ptr = nullptr;  // length = n_neurons + 1
  uint32_t* d_post    = nullptr;  // length = n_synapses
  float*    d_weight  = nullptr;  // length = n_synapses
  uint8_t*  d_nt_type = nullptr;  // length = n_synapses
};

// ---------------------------------------------------------------------------
// Device-side NT sign lookup. Matches SynapseTable::Sign().
// ---------------------------------------------------------------------------
FWMC_HOST_DEVICE inline float NTSign(uint8_t nt) {
  // kGABA == 1, kGlut == 2 (both inhibitory, matching SynapseTable::Sign())
  return (nt == 1 || nt == 2) ? -1.0f : 1.0f;
}

}  // namespace mechabrain

#endif  // FWMC_CUDA
#endif  // FWMC_CUDA_KERNELS_CUH_
