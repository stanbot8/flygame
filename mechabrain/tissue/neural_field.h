#ifndef FWMC_NEURAL_FIELD_H_
#define FWMC_NEURAL_FIELD_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "tissue/voxel_grid.h"

namespace mechabrain {

// Wilson-Cowan neural field model on a 3D voxel grid.
// Two coupled fields: excitatory activity (E) and inhibitory activity (I).
//
//   tau_e * dE/dt = -E + S(w_ee*E - w_ei*I + h_e) + D_e * laplacian(E)
//   tau_i * dI/dt = -I + S(w_ie*E - w_ii*I + h_i) + D_i * laplacian(I)
//
// S(x) = 1 / (1 + exp(-beta*(x - theta)))  (sigmoidal activation)
//
// This provides LOD 0: the coarsest representation of brain activity
// as a continuous field. Each voxel's E/I values represent the mean
// activity of the neural population at that location.
//
// The SDF mask restricts dynamics to voxels inside the brain (sdf < 0).
// Outside voxels are clamped to zero.

struct NeuralFieldParams {
  // Time constants (ms)
  float tau_e = 10.0f;     // excitatory time constant
  float tau_i = 20.0f;     // inhibitory time constant

  // Connectivity weights (dimensionless)
  float w_ee = 12.0f;      // E -> E (recurrent excitation)
  float w_ei = 4.0f;       // I -> E (inhibition of excitatory)
  float w_ie = 13.0f;      // E -> I (excitation of inhibitory)
  float w_ii = 2.0f;       // I -> I (recurrent inhibition)

  // External drive
  float h_e = -2.0f;       // tonic excitatory input
  float h_i = -3.5f;       // tonic inhibitory input

  // Sigmoid parameters
  float beta = 1.0f;       // steepness
  float theta = 4.0f;      // threshold

  // Spatial coupling (diffusion coefficients, um^2/ms)
  float D_e = 50.0f;       // excitatory lateral spread
  float D_i = 20.0f;       // inhibitory lateral spread (shorter range)
};

struct NeuralField {
  size_t ch_e = SIZE_MAX;       // excitatory activity channel
  size_t ch_i = SIZE_MAX;       // inhibitory activity channel
  size_t ch_sdf = SIZE_MAX;     // SDF channel (mask: sdf < 0 = inside brain)
  NeuralFieldParams params;

  // Sigmoid activation function.
  static float Sigmoid(float x, float beta, float theta) {
    return 1.0f / (1.0f + std::exp(-beta * (x - theta)));
  }

  // Initialize E/I channels on the grid.
  // Assumes SDF channel already exists and is baked.
  void Init(VoxelGrid& grid) {
    ch_e = grid.AddChannel("field_E", params.D_e, 0.0f);
    ch_i = grid.AddChannel("field_I", params.D_i, 0.0f);

    // Set small initial activity inside the brain
    auto& sdf = grid.channels[ch_sdf].data;
    auto& E = grid.channels[ch_e].data;
    auto& I = grid.channels[ch_i].data;
    for (size_t i = 0; i < grid.NumVoxels(); ++i) {
      if (sdf[i] < 0.0f) {
        E[i] = 0.1f;
        I[i] = 0.05f;
      }
    }
  }

  // Step the neural field forward by dt_ms.
  // Uses operator splitting: reaction step (Wilson-Cowan ODE) then
  // diffusion step (handled by VoxelGrid::Diffuse).
  void Step(VoxelGrid& grid, float dt_ms) {
    auto& sdf = grid.channels[ch_sdf].data;
    auto& E = grid.channels[ch_e].data;
    auto& I = grid.channels[ch_i].data;

    // Reaction step (forward Euler)
    for (size_t i = 0; i < grid.NumVoxels(); ++i) {
      if (sdf[i] >= 0.0f) {
        E[i] = 0.0f;
        I[i] = 0.0f;
        continue;
      }

      float e = E[i];
      float ii = I[i];

      float se = Sigmoid(params.w_ee * e - params.w_ei * ii + params.h_e,
                         params.beta, params.theta);
      float si = Sigmoid(params.w_ie * e - params.w_ii * ii + params.h_i,
                         params.beta, params.theta);

      E[i] = e + dt_ms / params.tau_e * (-e + se);
      I[i] = ii + dt_ms / params.tau_i * (-ii + si);

      // Clamp to [0, 1]
      E[i] = std::clamp(E[i], 0.0f, 1.0f);
      I[i] = std::clamp(I[i], 0.0f, 1.0f);
    }

    // Diffusion step (lateral coupling)
    // Only diffuse E and I channels. The grid's Diffuse() method
    // uses the diffusion_coeff set when the channel was created.
    DiffuseChannel(grid, ch_e, dt_ms);
    DiffuseChannel(grid, ch_i, dt_ms);

    // Re-mask: zero out anything that diffused outside the brain
    for (size_t i = 0; i < grid.NumVoxels(); ++i) {
      if (sdf[i] >= 0.0f) {
        E[i] = 0.0f;
        I[i] = 0.0f;
      }
    }
  }

  // Inject external stimulus into the excitatory field at a world position.
  void Stimulate(VoxelGrid& grid, float wx, float wy, float wz,
                 float radius_um, float intensity) {
    grid.InjectSphere(ch_e, wx, wy, wz, radius_um, intensity);
  }

  // Read the excitatory activity at a world position (trilinear interpolated).
  float ReadActivity(const VoxelGrid& grid, float wx, float wy, float wz) const {
    return grid.Sample(ch_e, wx, wy, wz);
  }

 private:
  // Diffuse a single channel. Separated from VoxelGrid::Diffuse so we
  // can selectively diffuse only the neural field channels per step.
  static void DiffuseChannel(VoxelGrid& grid, size_t ch, float dt_ms) {
    auto& data = grid.channels[ch].data;
    float D = grid.channels[ch].diffusion_coeff;
    if (D <= 0.0f) return;

    float alpha = D * dt_ms / (grid.dx * grid.dx);

    // Subdivide for stability
    int substeps = static_cast<int>(std::ceil(alpha * 6.0f));
    if (substeps < 1) substeps = 1;
    float sub_alpha = alpha / substeps;

    std::vector<float> temp(data.size());

    for (int sub = 0; sub < substeps; ++sub) {
      for (uint32_t z = 0; z < grid.nz; ++z) {
        for (uint32_t y = 0; y < grid.ny; ++y) {
          for (uint32_t x = 0; x < grid.nx; ++x) {
            size_t i = grid.Idx(x, y, z);
            float c = data[i];

            float xm = (x > 0)          ? data[grid.Idx(x-1,y,z)] : c;
            float xp = (x < grid.nx-1)  ? data[grid.Idx(x+1,y,z)] : c;
            float ym = (y > 0)          ? data[grid.Idx(x,y-1,z)] : c;
            float yp = (y < grid.ny-1)  ? data[grid.Idx(x,y+1,z)] : c;
            float zm = (z > 0)          ? data[grid.Idx(x,y,z-1)] : c;
            float zp = (z < grid.nz-1)  ? data[grid.Idx(x,y,z+1)] : c;

            temp[i] = c + sub_alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
          }
        }
      }
      data.swap(temp);
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_NEURAL_FIELD_H_
