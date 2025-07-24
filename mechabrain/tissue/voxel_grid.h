#ifndef FWMC_VOXEL_GRID_H_
#define FWMC_VOXEL_GRID_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mechabrain {

// 3D regular grid of scalar fields.
// Stores multiple named channels (firing rate, neuromodulator concentrations,
// temperature, light intensity, etc.) on a shared spatial grid.
// Coordinates are in micrometers. Drosophila brain is ~500x300x200 um.
struct VoxelGrid {
  uint32_t nx = 0, ny = 0, nz = 0;  // grid dimensions (voxels)
  float dx = 1.0f;                    // voxel spacing (um)

  // Origin in world coordinates (um). Bottom-left-front corner.
  float origin_x = 0.0f, origin_y = 0.0f, origin_z = 0.0f;

  // Each channel is a flat array of nx*ny*nz floats.
  // Indexed as [z * ny * nx + y * nx + x].
  struct Channel {
    std::string name;
    std::vector<float> data;
    std::vector<float> temp;       // scratch buffer for diffusion (avoids per-step alloc)
    float diffusion_coeff = 0.0f;  // D in um^2/ms
    float decay_rate = 0.0f;       // lambda: dc/dt includes -lambda*c
  };
  std::vector<Channel> channels;

  void Init(uint32_t w, uint32_t h, uint32_t d, float spacing) {
    nx = w; ny = h; nz = d;
    dx = spacing;
    channels.clear();
  }

  size_t NumVoxels() const { return static_cast<size_t>(nx) * ny * nz; }

  // Add a named channel, returns channel index.
  size_t AddChannel(const std::string& name, float diffusion = 0.0f,
                    float decay = 0.0f) {
    Channel ch;
    ch.name = name;
    ch.data.assign(NumVoxels(), 0.0f);
    ch.temp.resize(NumVoxels());
    ch.diffusion_coeff = diffusion;
    ch.decay_rate = decay;
    channels.push_back(std::move(ch));
    return channels.size() - 1;
  }

  size_t FindChannel(const std::string& name) const {
    for (size_t i = 0; i < channels.size(); ++i) {
      if (channels[i].name == name) return i;
    }
    return SIZE_MAX;
  }

  // Linear index from 3D coordinates.
  size_t Idx(uint32_t x, uint32_t y, uint32_t z) const {
    return static_cast<size_t>(z) * ny * nx + y * nx + x;
  }

  // World position (um) to grid coordinates.
  // Returns false if outside the grid.
  bool WorldToGrid(float wx, float wy, float wz,
                   uint32_t& gx, uint32_t& gy, uint32_t& gz) const {
    float fx = (wx - origin_x) / dx;
    float fy = (wy - origin_y) / dx;
    float fz = (wz - origin_z) / dx;
    if (fx < 0 || fy < 0 || fz < 0) return false;
    gx = static_cast<uint32_t>(fx);
    gy = static_cast<uint32_t>(fy);
    gz = static_cast<uint32_t>(fz);
    return gx < nx && gy < ny && gz < nz;
  }

  // Grid coordinates to world position (center of voxel).
  void GridToWorld(uint32_t gx, uint32_t gy, uint32_t gz,
                   float& wx, float& wy, float& wz) const {
    wx = origin_x + (gx + 0.5f) * dx;
    wy = origin_y + (gy + 0.5f) * dx;
    wz = origin_z + (gz + 0.5f) * dx;
  }

  // Trilinear interpolation of a channel at world coordinates.
  float Sample(size_t channel, float wx, float wy, float wz) const {
    float fx = (wx - origin_x) / dx - 0.5f;
    float fy = (wy - origin_y) / dx - 0.5f;
    float fz = (wz - origin_z) / dx - 0.5f;

    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int z0 = static_cast<int>(std::floor(fz));

    float tx = fx - x0;
    float ty = fy - y0;
    float tz = fz - z0;

    auto clamp_get = [&](int x, int y, int z) -> float {
      x = std::clamp(x, 0, static_cast<int>(nx) - 1);
      y = std::clamp(y, 0, static_cast<int>(ny) - 1);
      z = std::clamp(z, 0, static_cast<int>(nz) - 1);
      return channels[channel].data[Idx(x, y, z)];
    };

    // Trilinear
    float c000 = clamp_get(x0, y0, z0);
    float c100 = clamp_get(x0+1, y0, z0);
    float c010 = clamp_get(x0, y0+1, z0);
    float c110 = clamp_get(x0+1, y0+1, z0);
    float c001 = clamp_get(x0, y0, z0+1);
    float c101 = clamp_get(x0+1, y0, z0+1);
    float c011 = clamp_get(x0, y0+1, z0+1);
    float c111 = clamp_get(x0+1, y0+1, z0+1);

    float c00 = c000 * (1-tx) + c100 * tx;
    float c10 = c010 * (1-tx) + c110 * tx;
    float c01 = c001 * (1-tx) + c101 * tx;
    float c11 = c011 * (1-tx) + c111 * tx;
    float c0 = c00 * (1-ty) + c10 * ty;
    float c1 = c01 * (1-ty) + c11 * ty;
    return c0 * (1-tz) + c1 * tz;
  }

  // Diffuse all channels with nonzero diffusion_coeff.
  // Explicit Euler, 7-point stencil. Neumann (zero-flux) boundaries.
  // Stability requires dt < dx^2 / (6*D).
  void Diffuse(float dt_ms) {
    for (auto& ch : channels) {
      // Diffusion step (skip if no diffusion coefficient)
      if (ch.diffusion_coeff > 0.0f) {
        float D = ch.diffusion_coeff;
        float alpha = D * dt_ms / (dx * dx);

        // Stability check: subdivide if needed
        int substeps = static_cast<int>(std::ceil(alpha * 6.0f));
        if (substeps < 1) substeps = 1;
        float sub_alpha = alpha / substeps;

        // Use pre-allocated scratch buffer (avoids heap alloc per timestep)
        if (ch.temp.size() != ch.data.size()) ch.temp.resize(ch.data.size());

        // Precompute index strides to avoid per-voxel multiply in Idx()
        const size_t stride_x = 1;
        const size_t stride_y = nx;
        const size_t stride_z = static_cast<size_t>(ny) * nx;

        for (int sub = 0; sub < substeps; ++sub) {
          const float* src = ch.data.data();
          float* dst = ch.temp.data();
          size_t i = 0;
          for (uint32_t z = 0; z < nz; ++z) {
            for (uint32_t y = 0; y < ny; ++y) {
              for (uint32_t x = 0; x < nx; ++x, ++i) {
                float c = src[i];

                // Neighbors via offset arithmetic (no multiply per voxel).
                // Neumann BC: boundary voxels mirror the center value.
                float xm = (x > 0)    ? src[i - stride_x] : c;
                float xp = (x < nx-1) ? src[i + stride_x] : c;
                float ym = (y > 0)    ? src[i - stride_y] : c;
                float yp = (y < ny-1) ? src[i + stride_y] : c;
                float zm = (z > 0)    ? src[i - stride_z] : c;
                float zp = (z < nz-1) ? src[i + stride_z] : c;

                float laplacian = (xm + xp + ym + yp + zm + zp - 6.0f * c);
                dst[i] = c + sub_alpha * laplacian;
              }
            }
          }
          ch.data.swap(ch.temp);
        }
      }

      // Decay (always applied, independent of diffusion)
      if (ch.decay_rate > 0.0f) {
        float decay = std::exp(-ch.decay_rate * dt_ms);
        for (auto& v : ch.data) v *= decay;
      }
    }
  }

  // Inject a point source at world coordinates.
  void Inject(size_t channel, float wx, float wy, float wz, float amount) {
    uint32_t gx, gy, gz;
    if (WorldToGrid(wx, wy, wz, gx, gy, gz)) {
      channels[channel].data[Idx(gx, gy, gz)] += amount;
    }
  }

  // Inject a spherical source (distributes amount across voxels within radius).
  void InjectSphere(size_t channel, float wx, float wy, float wz,
                    float radius_um, float amount) {
    int r_vox = static_cast<int>(std::ceil(radius_um / dx));
    uint32_t cx, cy, cz;
    if (!WorldToGrid(wx, wy, wz, cx, cy, cz)) return;

    float r2 = radius_um * radius_um;
    int count = 0;

    // Count voxels in sphere first
    for (int dz = -r_vox; dz <= r_vox; ++dz) {
      for (int dy = -r_vox; dy <= r_vox; ++dy) {
        for (int ddx = -r_vox; ddx <= r_vox; ++ddx) {
          float dist2 = (ddx*dx)*(ddx*dx) + (dy*dx)*(dy*dx) + (dz*dx)*(dz*dx);
          if (dist2 > r2) continue;
          int gx = static_cast<int>(cx) + ddx;
          int gy = static_cast<int>(cy) + dy;
          int gz = static_cast<int>(cz) + dz;
          if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny &&
              gz >= 0 && gz < (int)nz) {
            count++;
          }
        }
      }
    }

    if (count == 0) return;
    float per_voxel = amount / count;

    for (int dz = -r_vox; dz <= r_vox; ++dz) {
      for (int dy = -r_vox; dy <= r_vox; ++dy) {
        for (int ddx = -r_vox; ddx <= r_vox; ++ddx) {
          float dist2 = (ddx*dx)*(ddx*dx) + (dy*dx)*(dy*dx) + (dz*dx)*(dz*dx);
          if (dist2 > r2) continue;
          int gx = static_cast<int>(cx) + ddx;
          int gy = static_cast<int>(cy) + dy;
          int gz = static_cast<int>(cz) + dz;
          if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny &&
              gz >= 0 && gz < (int)nz) {
            channels[channel].data[Idx(gx, gy, gz)] += per_voxel;
          }
        }
      }
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_VOXEL_GRID_H_
