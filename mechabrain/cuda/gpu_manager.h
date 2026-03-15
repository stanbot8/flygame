#ifndef FWMC_CUDA_GPU_MANAGER_H_
#define FWMC_CUDA_GPU_MANAGER_H_

#ifdef FWMC_CUDA

#include <cstdio>
#include <cuda_runtime.h>
#include "cuda/kernels.cuh"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// ---------------------------------------------------------------------------
// GPUManager: handles device lifecycle, memory allocation, and host⇄device
// transfers for the FWMC spiking neural network simulator.
//
// Design notes:
//   - Header-only so there is no separate .cu compilation unit for this file.
//   - Uses a dedicated CUDA stream for async H↔D transfers, separate from the
//     compute stream(s) used by the kernels.  This allows overlap of transfer
//     and compute on GPUs with a copy engine (all discrete GPUs since Fermi).
//   - Every allocation is matched by a Free*() method.  The Shutdown() method
//     destroys only the manager's own stream. It does NOT free neuron/synapse
//     memory, so the caller can manage lifetimes independently.
// ---------------------------------------------------------------------------
struct GPUManager {
  int          device_id      = 0;
  cudaStream_t transfer_stream = nullptr;

  // -------------------------------------------------------------------
  // Initialization: select device and create the transfer stream.
  // -------------------------------------------------------------------
  void Init(int device = 0) {
    device_id = device;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::fprintf(stdout,
                 "[GPUManager] Using device %d: %s  (%.1f GB, SM %d.%d)\n",
                 device_id, prop.name,
                 static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                 prop.major, prop.minor);

    CUDA_CHECK(cudaStreamCreate(&transfer_stream));
  }

  // -------------------------------------------------------------------
  // Shutdown: destroy the transfer stream.
  // -------------------------------------------------------------------
  void Shutdown() {
    if (transfer_stream) {
      CUDA_CHECK(cudaStreamDestroy(transfer_stream));
      transfer_stream = nullptr;
    }
  }

  // ===================================================================
  //  Neuron allocation / transfer
  // ===================================================================

  NeuronArrayGPU AllocateNeurons(size_t n) {
    NeuronArrayGPU gpu;
    gpu.n = n;

    size_t fb = n * sizeof(float);
    size_t ub = n * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&gpu.d_v,               fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_u,               fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_i_syn,           fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_i_ext,           fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_spiked,          ub));
    CUDA_CHECK(cudaMalloc(&gpu.d_dopamine,        fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_serotonin,       fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_octopamine,      fb));
    CUDA_CHECK(cudaMalloc(&gpu.d_type,            ub));
    CUDA_CHECK(cudaMalloc(&gpu.d_region,          ub));
    CUDA_CHECK(cudaMalloc(&gpu.d_last_spike_time, fb));

    // Zero-initialize all buffers
    CUDA_CHECK(cudaMemsetAsync(gpu.d_i_syn,  0, fb, transfer_stream));
    CUDA_CHECK(cudaMemsetAsync(gpu.d_i_ext,  0, fb, transfer_stream));
    CUDA_CHECK(cudaMemsetAsync(gpu.d_spiked, 0, ub, transfer_stream));

    return gpu;
  }

  void FreeNeurons(NeuronArrayGPU& gpu) {
    if (gpu.d_v)               { CUDA_CHECK(cudaFree(gpu.d_v));               gpu.d_v               = nullptr; }
    if (gpu.d_u)               { CUDA_CHECK(cudaFree(gpu.d_u));               gpu.d_u               = nullptr; }
    if (gpu.d_i_syn)           { CUDA_CHECK(cudaFree(gpu.d_i_syn));           gpu.d_i_syn           = nullptr; }
    if (gpu.d_i_ext)           { CUDA_CHECK(cudaFree(gpu.d_i_ext));           gpu.d_i_ext           = nullptr; }
    if (gpu.d_spiked)          { CUDA_CHECK(cudaFree(gpu.d_spiked));          gpu.d_spiked          = nullptr; }
    if (gpu.d_dopamine)        { CUDA_CHECK(cudaFree(gpu.d_dopamine));        gpu.d_dopamine        = nullptr; }
    if (gpu.d_serotonin)       { CUDA_CHECK(cudaFree(gpu.d_serotonin));       gpu.d_serotonin       = nullptr; }
    if (gpu.d_octopamine)      { CUDA_CHECK(cudaFree(gpu.d_octopamine));      gpu.d_octopamine      = nullptr; }
    if (gpu.d_type)            { CUDA_CHECK(cudaFree(gpu.d_type));            gpu.d_type            = nullptr; }
    if (gpu.d_region)          { CUDA_CHECK(cudaFree(gpu.d_region));          gpu.d_region          = nullptr; }
    if (gpu.d_last_spike_time) { CUDA_CHECK(cudaFree(gpu.d_last_spike_time)); gpu.d_last_spike_time = nullptr; }
    gpu.n = 0;
  }

  // Upload full neuron state from host to device (async on transfer_stream).
  void UploadNeurons(const NeuronArray& cpu, NeuronArrayGPU& gpu) {
    size_t n  = cpu.n;
    size_t fb = n * sizeof(float);
    size_t ub = n * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpyAsync(gpu.d_v,               cpu.v.data(),               fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_u,               cpu.u.data(),               fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_i_syn,           cpu.i_syn.data(),           fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_i_ext,           cpu.i_ext.data(),           fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_spiked,          cpu.spiked.data(),          ub, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_dopamine,        cpu.dopamine.data(),        fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_serotonin,       cpu.serotonin.data(),       fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_octopamine,      cpu.octopamine.data(),      fb, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_type,            cpu.type.data(),            ub, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_region,          cpu.region.data(),          ub, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_last_spike_time, cpu.last_spike_time.data(), fb, cudaMemcpyHostToDevice, transfer_stream));
  }

  // Download neuron state from device to host (async on transfer_stream).
  // Caller must cudaStreamSynchronize(transfer_stream) before reading cpu data.
  void DownloadNeurons(const NeuronArrayGPU& gpu, NeuronArray& cpu) {
    size_t n  = gpu.n;
    size_t fb = n * sizeof(float);
    size_t ub = n * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpyAsync(cpu.v.data(),               gpu.d_v,               fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.u.data(),               gpu.d_u,               fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.i_syn.data(),           gpu.d_i_syn,           fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.i_ext.data(),           gpu.d_i_ext,           fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.spiked.data(),          gpu.d_spiked,          ub, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.dopamine.data(),        gpu.d_dopamine,        fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.serotonin.data(),       gpu.d_serotonin,       fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.octopamine.data(),      gpu.d_octopamine,      fb, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.type.data(),            gpu.d_type,            ub, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.region.data(),          gpu.d_region,          ub, cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.last_spike_time.data(), gpu.d_last_spike_time, fb, cudaMemcpyDeviceToHost, transfer_stream));
  }

  // ===================================================================
  //  Synapse allocation / transfer
  // ===================================================================

  SynapseTableGPU AllocateSynapses(size_t n_neurons, size_t n_synapses) {
    SynapseTableGPU gpu;
    gpu.n_neurons  = n_neurons;
    gpu.n_synapses = n_synapses;

    CUDA_CHECK(cudaMalloc(&gpu.d_row_ptr, (n_neurons + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_post,     n_synapses     * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_weight,   n_synapses     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_nt_type,  n_synapses     * sizeof(uint8_t)));

    return gpu;
  }

  void FreeSynapses(SynapseTableGPU& gpu) {
    if (gpu.d_row_ptr) { CUDA_CHECK(cudaFree(gpu.d_row_ptr)); gpu.d_row_ptr = nullptr; }
    if (gpu.d_post)    { CUDA_CHECK(cudaFree(gpu.d_post));    gpu.d_post    = nullptr; }
    if (gpu.d_weight)  { CUDA_CHECK(cudaFree(gpu.d_weight));  gpu.d_weight  = nullptr; }
    if (gpu.d_nt_type) { CUDA_CHECK(cudaFree(gpu.d_nt_type)); gpu.d_nt_type = nullptr; }
    gpu.n_neurons  = 0;
    gpu.n_synapses = 0;
  }

  // Upload synapse table from host to device (async on transfer_stream).
  void UploadSynapses(const SynapseTable& cpu, SynapseTableGPU& gpu) {
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_row_ptr, cpu.row_ptr.data(),
                               (cpu.n_neurons + 1) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_post,    cpu.post.data(),
                               cpu.Size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_weight,  cpu.weight.data(),
                               cpu.Size() * sizeof(float),
                               cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_nt_type, cpu.nt_type.data(),
                               cpu.Size() * sizeof(uint8_t),
                               cudaMemcpyHostToDevice, transfer_stream));
  }

  // Download synapse weights from device to host (weights are the only
  // field that changes during simulation, via STDP).
  void DownloadSynapseWeights(const SynapseTableGPU& gpu, SynapseTable& cpu) {
    CUDA_CHECK(cudaMemcpyAsync(cpu.weight.data(), gpu.d_weight,
                               gpu.n_synapses * sizeof(float),
                               cudaMemcpyDeviceToHost, transfer_stream));
  }

  // -------------------------------------------------------------------
  // Convenience: synchronize the transfer stream (block until all
  // pending async copies complete).
  // -------------------------------------------------------------------
  void SyncTransfers() {
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
  }
};

}  // namespace mechabrain

#endif  // FWMC_CUDA
#endif  // FWMC_CUDA_GPU_MANAGER_H_
