#ifdef FWMC_CUDA

#include "cuda/kernels.cuh"
#include "core/izhikevich.h"  // IzhikevichParams (shared with CPU path)

namespace mechabrain {

// ---------------------------------------------------------------------------
// GPU Izhikevich neuron update kernel.
//
// One thread per neuron.  The math is identical to the CPU IzhikevichStep():
//   1. Compute total input current  I = i_syn + i_ext
//   2. Two half-step Euler updates for numerical stability
//   3. Recovery variable update
//   4. NaN/Inf guard (reset divergent neurons)
//   5. Spike detection + reset
//
// All reads/writes are to global memory.  With 256 threads/block the L1/L2
// hit rates are excellent because adjacent threads access adjacent floats
// (unit-stride / coalesced).
// ---------------------------------------------------------------------------
__global__ void IzhikevichStepKernel(
    float*       __restrict__ v,
    float*       __restrict__ u,
    const float* __restrict__ i_syn,
    const float* __restrict__ i_ext,
    uint8_t*     __restrict__ spiked,
    float*       __restrict__ last_spike_time,
    int          n,
    float        dt_ms,
    float        sim_time_ms,
    float        a,
    float        b,
    float        c,
    float        d,
    float        v_thresh,
    float        refractory_ms) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float vi = v[idx];
  float ui = u[idx];
  float I  = i_syn[idx] + i_ext[idx];

  // Two half-steps (Izhikevich 2003)
  vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
  vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
  ui += dt_ms * a * (b * vi - ui);

  // NaN / Inf guard: reset divergent neurons to resting state
  if (!isfinite(vi)) vi = c;
  if (!isfinite(ui)) ui = b * c;

  // Spike detection (with absolute refractory period, matching CPU path)
  bool in_refractory = (sim_time_ms - last_spike_time[idx]) < refractory_ms;
  uint8_t fired = (!in_refractory && vi >= v_thresh) ? 1 : 0;
  if (fired) {
    vi = c;
    ui += d;
    last_spike_time[idx] = sim_time_ms;
  }

  v[idx]      = vi;
  u[idx]      = ui;
  spiked[idx] = fired;
}

// ---------------------------------------------------------------------------
// Host-side wrapper.  Launches the kernel on the given CUDA stream.
// ---------------------------------------------------------------------------
void IzhikevichStepGPU(NeuronArrayGPU&        neurons,
                       float                   dt_ms,
                       float                   sim_time_ms,
                       const IzhikevichParams& p,
                       cudaStream_t            stream) {
  if (neurons.n == 0) return;

  int n      = static_cast<int>(neurons.n);
  int blocks = (n + kCudaBlockSize - 1) / kCudaBlockSize;

  IzhikevichStepKernel<<<blocks, kCudaBlockSize, 0, stream>>>(
      neurons.d_v,
      neurons.d_u,
      neurons.d_i_syn,
      neurons.d_i_ext,
      neurons.d_spiked,
      neurons.d_last_spike_time,
      n,
      dt_ms,
      sim_time_ms,
      p.a, p.b, p.c, p.d, p.v_thresh, p.refractory_ms);

  // Check for launch errors (misconfigurations, OOM, etc.)
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mechabrain

#endif  // FWMC_CUDA
