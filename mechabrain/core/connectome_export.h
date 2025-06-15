#ifndef FWMC_CONNECTOME_EXPORT_H_
#define FWMC_CONNECTOME_EXPORT_H_

#include <cstdio>
#include <cstdint>
#include <string>
#include "core/error.h"
#include "core/log.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace mechabrain {

// Export NeuronArray and SynapseTable to the binary connectome format
// read by ConnectomeLoader.
// Format:
//   neurons.bin: [count:u32] [root_id:u64, x:f32, y:f32, z:f32, type:u8] * count
//   synapses.bin: [count:u32] [pre:u32, post:u32, weight:f32, nt:u8] * count
struct ConnectomeExport {
  static Result<size_t> ExportNeurons(const std::string& path,
                                      const NeuronArray& neurons) {
    if (neurons.n == 0) {
      return MakeError(ErrorCode::kInvalidParam,
                       "Cannot export empty NeuronArray");
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path + " for writing");
    }

    uint32_t count = static_cast<uint32_t>(neurons.n);
    if (fwrite(&count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to write neuron count to " + path);
    }

    for (uint32_t i = 0; i < count; ++i) {
      size_t ok = 0;
      ok += fwrite(&neurons.root_id[i], sizeof(uint64_t), 1, f);
      ok += fwrite(&neurons.x[i], sizeof(float), 1, f);
      ok += fwrite(&neurons.y[i], sizeof(float), 1, f);
      ok += fwrite(&neurons.z[i], sizeof(float), 1, f);
      ok += fwrite(&neurons.type[i], sizeof(uint8_t), 1, f);
      if (ok != 5) {
        fclose(f);
        return MakeError(ErrorCode::kCorruptedData,
                         "Failed to write neuron data at index " +
                         std::to_string(i));
      }
    }

    fclose(f);
    Log(LogLevel::kInfo, "Exported %u neurons to %s", count, path.c_str());
    return static_cast<size_t>(count);
  }

  static Result<size_t> ExportSynapses(const std::string& path,
                                       const SynapseTable& table) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open " + path + " for writing");
    }

    uint32_t count = static_cast<uint32_t>(table.Size());
    if (fwrite(&count, sizeof(uint32_t), 1, f) != 1) {
      fclose(f);
      return MakeError(ErrorCode::kCorruptedData,
                       "Failed to write synapse count to " + path);
    }

    // Reconstruct COO from CSR and write each synapse record
    for (uint32_t pre = 0; pre < table.n_neurons; ++pre) {
      uint32_t start = table.row_ptr[pre];
      uint32_t end = table.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        size_t ok = 0;
        ok += fwrite(&pre, sizeof(uint32_t), 1, f);
        ok += fwrite(&table.post[s], sizeof(uint32_t), 1, f);
        ok += fwrite(&table.weight[s], sizeof(float), 1, f);
        ok += fwrite(&table.nt_type[s], sizeof(uint8_t), 1, f);
        if (ok != 4) {
          fclose(f);
          return MakeError(ErrorCode::kCorruptedData,
                           "Failed to write synapse data at index " +
                           std::to_string(s));
        }
      }
    }

    fclose(f);
    Log(LogLevel::kInfo, "Exported %u synapses to %s", count, path.c_str());
    return static_cast<size_t>(count);
  }
};

}  // namespace mechabrain

#endif  // FWMC_CONNECTOME_EXPORT_H_
