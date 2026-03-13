#ifndef FWMC_CONNECTOME_LOADER_H_
#define FWMC_CONNECTOME_LOADER_H_

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "core/error.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Load binary connectome files produced by scripts/import_connectome.py.
// Format:
//   neurons.bin: [count:u32] [root_id:u64, x:f32, y:f32, z:f32, type:u8] * count
//   synapses.bin: [count:u32] [pre:u32, post:u32, weight:f32, nt:u8] * count
struct ConnectomeLoader {
  static constexpr size_t kMaxNeurons = 10'000'000;
  static constexpr size_t kMaxSynapses = 100'000'000;

  static Result<size_t> LoadNeurons(const std::string& path,
                                    NeuronArray& neurons) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path);
    }

    uint32_t count;
    if (fread(&count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to read neuron count from " + path);
    }

    if (count == 0 || count > kMaxNeurons) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Invalid neuron count: " + std::to_string(count));
    }

    neurons.Resize(count);

    for (uint32_t i = 0; i < count; ++i) {
      size_t ok = 0;
      ok += fread(&neurons.root_id[i], sizeof(uint64_t), 1, f);
      ok += fread(&neurons.x[i], sizeof(float), 1, f);
      ok += fread(&neurons.y[i], sizeof(float), 1, f);
      ok += fread(&neurons.z[i], sizeof(float), 1, f);
      ok += fread(&neurons.type[i], sizeof(uint8_t), 1, f);
      if (ok != 5) {
        fclose(f);
        return MakeError(ErrorCode::kCorruptedData,
                         "Truncated neuron data at index " +
                         std::to_string(i));
      }
    }

    fclose(f);
    Log(LogLevel::kInfo, "Loaded %u neurons from %s", count, path.c_str());
    return static_cast<size_t>(count);
  }

  static Result<size_t> LoadSynapses(const std::string& path,
                                     size_t n_neurons,
                                     SynapseTable& table) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path);
    }

    uint32_t count;
    if (fread(&count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to read synapse count from " + path);
    }

    if (count > kMaxSynapses) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Invalid synapse count: " + std::to_string(count));
    }

    // Read into COO format first, then convert to CSR
    std::vector<uint32_t> pre(count), post(count);
    std::vector<float> weight(count);
    std::vector<uint8_t> nt(count);

    for (uint32_t i = 0; i < count; ++i) {
      size_t ok = 0;
      ok += fread(&pre[i], sizeof(uint32_t), 1, f);
      ok += fread(&post[i], sizeof(uint32_t), 1, f);
      ok += fread(&weight[i], sizeof(float), 1, f);
      ok += fread(&nt[i], sizeof(uint8_t), 1, f);
      if (ok != 4) {
        fclose(f);
        return MakeError(ErrorCode::kCorruptedData,
                         "Truncated synapse data at index " +
                         std::to_string(i));
      }

      if (pre[i] >= n_neurons || post[i] >= n_neurons) {
        fclose(f);
        return MakeError(ErrorCode::kOutOfBounds,
                         "Synapse index out of bounds at " +
                         std::to_string(i) + ": pre=" +
                         std::to_string(pre[i]) + " post=" +
                         std::to_string(post[i]));
      }
    }

    fclose(f);

    table.BuildFromCOO(n_neurons, pre, post, weight, nt);
    Log(LogLevel::kInfo, "Loaded %u synapses from %s (CSR built)",
        count, path.c_str());
    return static_cast<size_t>(count);
  }
};

}  // namespace mechabrain

#endif  // FWMC_CONNECTOME_LOADER_H_
