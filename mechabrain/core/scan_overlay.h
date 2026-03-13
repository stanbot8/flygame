#ifndef FWMC_SCAN_OVERLAY_H_
#define FWMC_SCAN_OVERLAY_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/parametric_gen.h"

namespace mechabrain {

// Scan source specification: points a region or projection at real
// connectome data (EM reconstruction, calcium-inferred connectivity, etc.).
//
// Design: parametric generation fills everything first, then scan data
// overwrites specific regions/projections. This lets you run a full
// simulation with parametric-only, then progressively swap in scan data
// as it becomes available, without changing the spec structure.
//
// Binary formats match ConnectomeLoader (neurons.bin / synapses.bin).
struct ScanSource {
  std::string region_name;         // which region this scan covers
  std::string neurons_path;        // path to neurons.bin (positions, types)
  std::string synapses_path;       // path to synapses.bin (connectivity)
};

// Overlay scan data onto a parametrically generated brain.
//
// For each ScanSource:
//   1. Load neurons.bin: overwrite positions and types in the region slice
//   2. Load synapses.bin: replace internal synapses for that region
//
// The region must already exist in the ParametricGenerator's region_ranges
// (i.e., Generate() must have been called first).
//
// Neuron counts: if the scan has more neurons than the parametric region
// allocated, the extra neurons are ignored (with a warning). If the scan
// has fewer, remaining neurons keep their parametric values.
// This is intentional: the parametric spec defines the slot count,
// and scans fill what they can.
struct ScanOverlay {
  // Apply all scan sources to the existing parametric brain.
  // Returns the number of regions successfully overlaid.
  static Result<int> Apply(
      const std::vector<ScanSource>& sources,
      const std::vector<ParametricGenerator::RegionRange>& regions,
      NeuronArray& neurons,
      SynapseTable& synapses) {

    if (sources.empty()) return 0;

    int applied = 0;

    for (const auto& src : sources) {
      // Find the region
      int reg_idx = -1;
      for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].name == src.region_name) {
          reg_idx = static_cast<int>(i);
          break;
        }
      }
      if (reg_idx < 0) {
        Log(LogLevel::kWarn,
            "ScanOverlay: region '%s' not found, skipping",
            src.region_name.c_str());
        continue;
      }

      const auto& range = regions[static_cast<size_t>(reg_idx)];
      uint32_t slot_count = range.end - range.start;

      // Overlay neurons if path is provided
      if (!src.neurons_path.empty()) {
        auto res = OverlayNeurons(src.neurons_path, range.start, slot_count,
                                  neurons);
        if (!res.has_value()) {
          return MakeError(res.error().code,
                           "ScanOverlay neurons for '" + src.region_name +
                           "': " + res.error().message);
        }
        Log(LogLevel::kInfo,
            "ScanOverlay: overlaid %u neurons in region '%s'",
            *res, src.region_name.c_str());
      }

      // Overlay synapses if path is provided
      if (!src.synapses_path.empty()) {
        auto res = OverlaySynapses(src.synapses_path, range.start, slot_count,
                                   neurons.n, synapses);
        if (!res.has_value()) {
          return MakeError(res.error().code,
                           "ScanOverlay synapses for '" + src.region_name +
                           "': " + res.error().message);
        }
        Log(LogLevel::kInfo,
            "ScanOverlay: overlaid %u synapses in region '%s'",
            *res, src.region_name.c_str());
      }

      ++applied;
    }

    return applied;
  }

  // Overlay neuron positions and types from a scan file.
  // Returns the number of neurons actually overlaid.
  static Result<uint32_t> OverlayNeurons(
      const std::string& path,
      uint32_t region_start, uint32_t slot_count,
      NeuronArray& neurons) {

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path);
    }

    uint32_t scan_count = 0;
    if (fread(&scan_count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to read neuron count from " + path);
    }

    // Clamp to both slot_count and actual array bounds
    uint32_t max_safe = (region_start < static_cast<uint32_t>(neurons.n))
        ? static_cast<uint32_t>(neurons.n) - region_start : 0u;
    uint32_t overlay_count = std::min({scan_count, slot_count, max_safe});
    if (scan_count > slot_count) {
      Log(LogLevel::kWarn,
          "ScanOverlay: scan has %u neurons but region has %u slots, "
          "truncating",
          scan_count, slot_count);
    }

    for (uint32_t i = 0; i < overlay_count; ++i) {
      uint32_t idx = region_start + i;
      uint64_t root_id;
      float x, y, z;
      uint8_t type;

      size_t ok = 0;
      ok += fread(&root_id, sizeof(uint64_t), 1, f);
      ok += fread(&x, sizeof(float), 1, f);
      ok += fread(&y, sizeof(float), 1, f);
      ok += fread(&z, sizeof(float), 1, f);
      ok += fread(&type, sizeof(uint8_t), 1, f);
      if (ok != 5) {
        fclose(f);
        return MakeError(ErrorCode::kCorruptedData,
                         "Truncated neuron data at index " +
                         std::to_string(i));
      }

      neurons.root_id[idx] = root_id;
      neurons.x[idx] = x;
      neurons.y[idx] = y;
      neurons.z[idx] = z;
      neurons.type[idx] = type;
    }

    fclose(f);
    return overlay_count;
  }

  // Overlay synapses from a scan file.
  // Replaces internal synapses within the region (pre and post both in range)
  // while preserving all other synapses (cross-region projections, etc.).
  // Returns the number of scan synapses added.
  static Result<uint32_t> OverlaySynapses(
      const std::string& path,
      uint32_t region_start, uint32_t slot_count,
      size_t total_neurons,
      SynapseTable& synapses) {

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path);
    }

    uint32_t scan_count = 0;
    if (fread(&scan_count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to read synapse count from " + path);
    }

    uint32_t region_end = region_start + slot_count;

    // Read scan synapses. Indices in the scan file are local to the region
    // (0-based), so we offset them to global neuron indices.
    std::vector<uint32_t> scan_pre, scan_post;
    std::vector<float> scan_weight;
    std::vector<uint8_t> scan_nt;

    for (uint32_t i = 0; i < scan_count; ++i) {
      uint32_t pre_local, post_local;
      float w;
      uint8_t nt;

      size_t ok = 0;
      ok += fread(&pre_local, sizeof(uint32_t), 1, f);
      ok += fread(&post_local, sizeof(uint32_t), 1, f);
      ok += fread(&w, sizeof(float), 1, f);
      ok += fread(&nt, sizeof(uint8_t), 1, f);
      if (ok != 4) {
        fclose(f);
        return MakeError(ErrorCode::kCorruptedData,
                         "Truncated synapse data at index " +
                         std::to_string(i));
      }

      // Skip synapses that reference neurons beyond the slot count
      if (pre_local >= slot_count || post_local >= slot_count) continue;

      scan_pre.push_back(region_start + pre_local);
      scan_post.push_back(region_start + post_local);
      scan_weight.push_back(w);
      scan_nt.push_back(nt);
    }
    fclose(f);

    // Collect all existing synapses EXCEPT internal ones in this region
    std::vector<uint32_t> keep_pre, keep_post;
    std::vector<float> keep_weight;
    std::vector<uint8_t> keep_nt;

    for (size_t pre = 0; pre < total_neurons; ++pre) {
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        bool pre_in_region = (pre >= region_start && pre < region_end);
        bool post_in_region = (synapses.post[s] >= region_start &&
                               synapses.post[s] < region_end);
        // Keep synapse if it's NOT internal to this region
        if (!(pre_in_region && post_in_region)) {
          keep_pre.push_back(static_cast<uint32_t>(pre));
          keep_post.push_back(synapses.post[s]);
          keep_weight.push_back(synapses.weight[s]);
          keep_nt.push_back(synapses.nt_type[s]);
        }
      }
    }

    // Merge: kept synapses + scan synapses
    keep_pre.insert(keep_pre.end(), scan_pre.begin(), scan_pre.end());
    keep_post.insert(keep_post.end(), scan_post.begin(), scan_post.end());
    keep_weight.insert(keep_weight.end(), scan_weight.begin(), scan_weight.end());
    keep_nt.insert(keep_nt.end(), scan_nt.begin(), scan_nt.end());

    // Rebuild CSR
    synapses.BuildFromCOO(total_neurons, keep_pre, keep_post,
                          keep_weight, keep_nt);

    return static_cast<uint32_t>(scan_pre.size());
  }
};

}  // namespace mechabrain

#endif  // FWMC_SCAN_OVERLAY_H_
