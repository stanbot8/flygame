#ifndef FWMC_NWB_EXPORT_H_
#define FWMC_NWB_EXPORT_H_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include "core/cell_type_defs.h"
#include "core/log.h"
#include "core/neuron_array.h"

namespace mechabrain {

// Lightweight NWB-compatible exporter that writes spike times, voltage traces,
// and session metadata as CSV/JSON files. This avoids the HDF5 dependency
// while producing data that NWB-reading tools can ingest with minimal
// conversion.
//
// Output files:
//   spikes.nwb.csv:    one row per spike event
//   voltages.nwb.csv:  one row per timestep, columns for tracked neurons
//   session.nwb.json:  NWB-compatible session metadata
struct NWBExporter {
  NWBExporter() = default;
  NWBExporter(const NWBExporter&) = delete;
  NWBExporter& operator=(const NWBExporter&) = delete;
  NWBExporter(NWBExporter&&) = delete;
  NWBExporter& operator=(NWBExporter&&) = delete;

  // ---- public state ----
  FILE* spike_file = nullptr;
  FILE* voltage_file = nullptr;
  std::string output_dir;
  std::string session_description;
  std::string session_id;  // UUID-style identifier

  uint32_t n_neurons = 0;
  uint32_t n_recorded_steps = 0;
  uint32_t total_spikes = 0;

  // Which neurons to record voltage traces for (indices into NeuronArray).
  // Leave empty for no voltage recording (default).
  std::vector<uint32_t> voltage_subset;

  // Snapshot of per-neuron metadata taken at BeginSession for the JSON export.
  struct UnitMeta {
    uint64_t root_id;
    float x, y, z;
    uint8_t region;
    uint8_t cell_type;
  };
  std::vector<UnitMeta> units;

  // Stimulus descriptions added between Begin/End for metadata export.
  struct StimulusInfo {
    float start_ms;
    float stop_ms;
    std::string name;
    std::string description;
  };
  std::vector<StimulusInfo> stimuli;

  // ISO-8601 session start timestamp captured in BeginSession.
  std::string session_start_time;

  // ---- public API ----

  // Create output directory, open CSV files, capture neuron metadata.
  inline bool BeginSession(const std::string& dir,
                           const std::string& description,
                           const NeuronArray& neurons) {
    output_dir = dir;
    session_description = description;
    n_neurons = static_cast<uint32_t>(neurons.n);
    n_recorded_steps = 0;
    total_spikes = 0;
    stimuli.clear();

    std::filesystem::create_directories(dir);

    session_id = GenerateUUID();
    session_start_time = ISOTimestamp();

    // Snapshot unit metadata.
    units.resize(neurons.n);
    for (size_t i = 0; i < neurons.n; ++i) {
      units[i].root_id = neurons.root_id[i];
      units[i].x = neurons.x[i];
      units[i].y = neurons.y[i];
      units[i].z = neurons.z[i];
      units[i].region = neurons.region[i];
      units[i].cell_type = neurons.type[i];
    }

    auto fail = [&](const char* msg, const std::string& path) {
      Log(LogLevel::kError, "NWBExporter: %s: %s", msg, path.c_str());
      EndSession();
      return false;
    };

    // Open spike CSV.
    {
      std::string path = dir + "/spikes.nwb.csv";
      spike_file = fopen(path.c_str(), "w");
      if (!spike_file) return fail("Cannot open", path);
      fprintf(spike_file, "neuron_id,spike_time_ms,region,cell_type\n");
    }

    // Open voltage CSV if subset is configured.
    if (!voltage_subset.empty()) {
      std::string path = dir + "/voltages.nwb.csv";
      voltage_file = fopen(path.c_str(), "w");
      if (!voltage_file) return fail("Cannot open", path);

      // Header row: time_ms, neuron_0, neuron_1, ...
      fprintf(voltage_file, "time_ms");
      for (uint32_t idx : voltage_subset) {
        fprintf(voltage_file, ",neuron_%u", idx);
      }
      fprintf(voltage_file, "\n");
    }

    Log(LogLevel::kInfo, "NWBExporter: session started in %s (%u neurons, %zu voltage channels)",
        dir.c_str(), n_neurons, voltage_subset.size());
    return true;
  }

  // Configure which neuron indices get voltage traces recorded.
  // Call before BeginSession or between sessions; calling after BeginSession
  // takes effect only if you reopen the voltage file (not recommended).
  inline void SetVoltageSubset(const std::vector<uint32_t>& neuron_indices) {
    voltage_subset = neuron_indices;
  }

  // Record one simulation timestep: emit spike rows and voltage row.
  inline void RecordTimestep(float time_ms, const NeuronArray& neurons) {
    n_recorded_steps++;

    // Spikes.
    if (spike_file) {
      for (uint32_t i = 0; i < n_neurons; ++i) {
        if (neurons.spiked[i]) {
          fprintf(spike_file, "%u,%.4f,%s,%s\n",
                  i, time_ms,
                  RegionLabel(neurons.region[i]),
                  CellTypeLabel(neurons.type[i]));
          ++total_spikes;
        }
      }
    }

    // Voltages.
    if (voltage_file && !voltage_subset.empty()) {
      fprintf(voltage_file, "%.4f", time_ms);
      for (uint32_t idx : voltage_subset) {
        if (idx < neurons.n) {
          fprintf(voltage_file, ",%.6f", neurons.v[idx]);
        } else {
          fprintf(voltage_file, ",");
        }
      }
      fprintf(voltage_file, "\n");
    }
  }

  // Register a stimulus for the metadata export.
  inline void AddStimulus(float start_ms, float stop_ms,
                          const std::string& name,
                          const std::string& desc) {
    stimuli.push_back({start_ms, stop_ms, name, desc});
  }

  // Flush and close CSVs, then write the metadata JSON.
  inline void EndSession() {
    if (spike_file) {
      fclose(spike_file);
      spike_file = nullptr;
    }
    if (voltage_file) {
      fclose(voltage_file);
      voltage_file = nullptr;
    }

    if (!output_dir.empty()) {
      WriteMetadataJSON();
      Log(LogLevel::kInfo, "NWBExporter: session ended, %u steps, %u spikes",
          n_recorded_steps, total_spikes);
    }
    output_dir.clear();
  }

  ~NWBExporter() { EndSession(); }

 private:
  // ---- JSON writer (hand-rolled, no dependency) ----

  inline void WriteMetadataJSON() const {
    std::string path = output_dir + "/session.nwb.json";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
      Log(LogLevel::kError, "NWBExporter: cannot write %s", path.c_str());
      return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"neurodata_type\": \"NWBFile\",\n");
    fprintf(f, "  \"nwb_version\": \"2.7.0\",\n");
    fprintf(f, "  \"identifier\": \"%s\",\n", session_id.c_str());
    fprintf(f, "  \"session_description\": \"%s\",\n", EscapeJSON(session_description).c_str());
    fprintf(f, "  \"session_start_time\": \"%s\",\n", session_start_time.c_str());
    fprintf(f, "  \"timestamps_reference_time\": \"%s\",\n", session_start_time.c_str());
    fprintf(f, "  \"file_create_date\": \"%s\",\n", ISOTimestamp().c_str());

    // Subject
    fprintf(f, "  \"subject\": {\n");
    fprintf(f, "    \"neurodata_type\": \"Subject\",\n");
    fprintf(f, "    \"species\": \"Drosophila melanogaster\",\n");
    fprintf(f, "    \"description\": \"In-silico Drosophila brain model (FlyWire connectome)\"\n");
    fprintf(f, "  },\n");

    // General
    fprintf(f, "  \"general\": {\n");
    fprintf(f, "    \"source_script\": \"FWMC (FlyWire Mind Couple)\",\n");
    fprintf(f, "    \"data_collection\": \"Simulated spiking neural network, Izhikevich model\",\n");
    fprintf(f, "    \"stimulus\": \"see stimulus_presentations\"\n");
    fprintf(f, "  },\n");

    // Units (electrode/neuron metadata)
    fprintf(f, "  \"units\": {\n");
    fprintf(f, "    \"neurodata_type\": \"Units\",\n");
    fprintf(f, "    \"description\": \"Simulated neuron units\",\n");
    fprintf(f, "    \"colnames\": [\"id\", \"root_id\", \"x\", \"y\", \"z\", \"region\", \"cell_type\"],\n");
    fprintf(f, "    \"count\": %u,\n", n_neurons);
    fprintf(f, "    \"spike_times_file\": \"spikes.nwb.csv\",\n");

    // Write a compact sample of units (first 20 and last 5) to keep JSON small.
    // Full unit table is in the CSV spike file via region/cell_type columns.
    fprintf(f, "    \"sample\": [\n");
    size_t sample_count = 0;
    auto write_unit = [&](size_t i) {
      if (i >= units.size()) return;
      if (sample_count > 0) fprintf(f, ",\n");
      fprintf(f, "      {\"id\": %zu, \"root_id\": %llu, "
                 "\"x\": %.1f, \"y\": %.1f, \"z\": %.1f, "
                 "\"region\": \"%s\", \"cell_type\": \"%s\"}",
              i, static_cast<unsigned long long>(units[i].root_id),
              units[i].x, units[i].y, units[i].z,
              RegionLabel(units[i].region),
              CellTypeLabel(units[i].cell_type));
      ++sample_count;
    };

    size_t head = (units.size() < 20) ? units.size() : 20;
    for (size_t i = 0; i < head; ++i) write_unit(i);
    if (units.size() > 25) {
      for (size_t i = units.size() - 5; i < units.size(); ++i) write_unit(i);
    }
    fprintf(f, "\n    ]\n");
    fprintf(f, "  },\n");

    // Stimulus presentations
    fprintf(f, "  \"stimulus_presentations\": [\n");
    for (size_t i = 0; i < stimuli.size(); ++i) {
      fprintf(f, "    {\"name\": \"%s\", \"start_ms\": %.4f, \"stop_ms\": %.4f, "
                 "\"description\": \"%s\"}%s\n",
              EscapeJSON(stimuli[i].name).c_str(),
              stimuli[i].start_ms, stimuli[i].stop_ms,
              EscapeJSON(stimuli[i].description).c_str(),
              (i + 1 < stimuli.size()) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // Acquisition info
    fprintf(f, "  \"acquisition\": {\n");
    fprintf(f, "    \"spike_times\": {\"file\": \"spikes.nwb.csv\", \"format\": \"csv\"},\n");
    if (!voltage_subset.empty()) {
      fprintf(f, "    \"voltage_traces\": {\n");
      fprintf(f, "      \"file\": \"voltages.nwb.csv\",\n");
      fprintf(f, "      \"format\": \"csv\",\n");
      fprintf(f, "      \"num_channels\": %zu,\n", voltage_subset.size());
      fprintf(f, "      \"neuron_indices\": [");
      for (size_t i = 0; i < voltage_subset.size(); ++i) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%u", voltage_subset[i]);
      }
      fprintf(f, "]\n");
      fprintf(f, "    }\n");
    } else {
      fprintf(f, "    \"voltage_traces\": null\n");
    }
    fprintf(f, "  },\n");

    // Summary statistics
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"num_neurons\": %u,\n", n_neurons);
    fprintf(f, "    \"num_timesteps\": %u,\n", n_recorded_steps);
    fprintf(f, "    \"total_spikes\": %u\n", total_spikes);
    fprintf(f, "  }\n");

    fprintf(f, "}\n");
    fclose(f);

    Log(LogLevel::kInfo, "NWBExporter: metadata written to %s", path.c_str());
  }

  // ---- string helpers ----

  static inline const char* RegionLabel(uint8_t r) {
    switch (static_cast<Region>(r)) {
      case Region::kAL:   return "AL";
      case Region::kMB:   return "MB";
      case Region::kMBON: return "MBON";
      case Region::kDAN:  return "DAN";
      case Region::kLH:   return "LH";
      case Region::kCX:   return "CX";
      case Region::kOL:   return "OL";
      case Region::kSEZ:  return "SEZ";
      case Region::kPN:   return "PN";
      default:            return "unknown";
    }
  }

  static inline const char* CellTypeLabel(uint8_t ct) {
    switch (static_cast<CellType>(ct)) {
      case CellType::kKenyonCell:          return "KenyonCell";
      case CellType::kMBON_cholinergic:    return "MBON_cholinergic";
      case CellType::kMBON_gabaergic:      return "MBON_gabaergic";
      case CellType::kMBON_glutamatergic:  return "MBON_glutamatergic";
      case CellType::kDAN_PPL1:            return "DAN_PPL1";
      case CellType::kDAN_PAM:             return "DAN_PAM";
      case CellType::kPN_excitatory:       return "PN_excitatory";
      case CellType::kPN_inhibitory:       return "PN_inhibitory";
      case CellType::kLN_local:            return "LN_local";
      case CellType::kORN:                 return "ORN";
      case CellType::kFastSpiking:         return "FastSpiking";
      case CellType::kBursting:            return "Bursting";
      case CellType::kSerotonergic:        return "Serotonergic";
      case CellType::kOctopaminergic:      return "Octopaminergic";
      case CellType::kL23_Pyramidal:       return "L23_Pyramidal";
      case CellType::kL4_Stellate:         return "L4_Stellate";
      case CellType::kL5_Pyramidal:        return "L5_Pyramidal";
      case CellType::kL6_Pyramidal:        return "L6_Pyramidal";
      case CellType::kPV_Basket:           return "PV_Basket";
      case CellType::kSST_Martinotti:      return "SST_Martinotti";
      case CellType::kVIP_Interneuron:     return "VIP_Interneuron";
      case CellType::kThalamocortical:     return "Thalamocortical";
      case CellType::kTRN:                 return "TRN";
      case CellType::kCholinergic_NBM:     return "Cholinergic_NBM";
      // Central Complex types
      case CellType::kEPG:                return "EPG";
      case CellType::kPEN:                return "PEN";
      case CellType::kPEG:                return "PEG";
      case CellType::kPFL:                return "PFL";
      case CellType::kDelta7:             return "Delta7";
      case CellType::kRingNeuron:         return "RingNeuron";
      case CellType::kFC:                 return "FC";
      // Zebrafish types
      case CellType::kRGC:                return "RGC";
      case CellType::kTectalPVN:          return "TectalPVN";
      case CellType::kTectalSIN:          return "TectalSIN";
      case CellType::kReticulospinal:     return "Reticulospinal";
      case CellType::kGranuleCell:        return "GranuleCell";
      // Grooming circuit types
      case CellType::kDNg_command:        return "DNg_command";
      case CellType::kGroomingSensory:    return "GroomingSensory";
      case CellType::kGroomingInhibitory: return "GroomingInhibitory";
      default:                             return "Generic";
    }
  }

  // Minimal JSON string escaping (backslash, double-quote, control chars).
  static inline std::string EscapeJSON(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:   out += c;      break;
      }
    }
    return out;
  }

  // Pseudo-UUID v4 for session identifiers.
  static inline std::string GenerateUUID() {
    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<uint32_t> dist(0, 15);

    auto hex = [](uint32_t v) -> char {
      return "0123456789abcdef"[v & 0xF];
    };

    // 8-4-4-4-12 format
    std::string uuid;
    uuid.reserve(36);
    for (int i = 0; i < 32; ++i) {
      if (i == 8 || i == 12 || i == 16 || i == 20) uuid += '-';
      if (i == 12) {
        uuid += '4';  // version 4
      } else if (i == 16) {
        uuid += hex((dist(rng) & 0x3) | 0x8);  // variant 1
      } else {
        uuid += hex(dist(rng));
      }
    }
    return uuid;
  }

  // ISO-8601 UTC timestamp.
  static inline std::string ISOTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm t;
#ifdef _WIN32
    gmtime_s(&t, &tt);
#else
    gmtime_r(&tt, &t);
#endif
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
             t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
             t.tm_hour, t.tm_min, t.tm_sec);
    return buf;
  }
};

}  // namespace mechabrain

#endif  // FWMC_NWB_EXPORT_H_
