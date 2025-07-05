#ifndef FWMC_BRAIN_SPEC_LOADER_H_
#define FWMC_BRAIN_SPEC_LOADER_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include "core/error.h"
#include "core/parametric_gen.h"
#include "core/scan_overlay.h"

namespace mechabrain {

// Parse a BrainSpec from a config file.
// Format (one key=value per line, regions and projections use prefixed keys):
//
//   name = mushroom_body_model
//   seed = 42
//   weight_mean = 1.0
//   weight_std = 0.3
//
//   region.0.name = antennal_lobe
//   region.0.n_neurons = 500
//   region.0.density = 0.15
//   region.0.nt = ACh
//   region.0.types = PN:0.4 LN:0.6
//   region.0.nt_dist = ACh:0.6 GABA:0.4
//
//   projection.0.from = antennal_lobe
//   projection.0.to = mushroom_body
//   projection.0.density = 0.01
//   projection.0.nt = ACh
//   projection.0.weight_mean = 1.2
//   projection.0.weight_std = 0.3
//
struct BrainSpecLoader {
  static CellType ParseCellTypeName(const std::string& name) {
    if (name == "KC" || name == "KenyonCell") return CellType::kKenyonCell;
    if (name == "MBON_cholinergic") return CellType::kMBON_cholinergic;
    if (name == "MBON_gabaergic") return CellType::kMBON_gabaergic;
    if (name == "MBON_glutamatergic") return CellType::kMBON_glutamatergic;
    if (name == "DAN_PPL1") return CellType::kDAN_PPL1;
    if (name == "DAN_PAM") return CellType::kDAN_PAM;
    if (name == "PN" || name == "PN_excitatory") return CellType::kPN_excitatory;
    if (name == "PN_inhibitory") return CellType::kPN_inhibitory;
    if (name == "LN" || name == "LN_local") return CellType::kLN_local;
    if (name == "ORN") return CellType::kORN;
    if (name == "FastSpiking") return CellType::kFastSpiking;
    if (name == "Bursting") return CellType::kBursting;
    if (name == "Serotonergic" || name == "5HT") return CellType::kSerotonergic;
    if (name == "Octopaminergic" || name == "OA" || name == "Tdc2") return CellType::kOctopaminergic;

    // Mammalian cortical cell types
    if (name == "L23_Pyramidal" || name == "L2/3_Pyramidal") return CellType::kL23_Pyramidal;
    if (name == "L4_Stellate") return CellType::kL4_Stellate;
    if (name == "L5_Pyramidal") return CellType::kL5_Pyramidal;
    if (name == "L6_Pyramidal") return CellType::kL6_Pyramidal;
    if (name == "PV_Basket" || name == "PV") return CellType::kPV_Basket;
    if (name == "SST_Martinotti" || name == "SST") return CellType::kSST_Martinotti;
    if (name == "VIP_Interneuron" || name == "VIP") return CellType::kVIP_Interneuron;
    if (name == "Thalamocortical" || name == "TC") return CellType::kThalamocortical;
    if (name == "TRN") return CellType::kTRN;
    if (name == "Cholinergic_NBM" || name == "NBM") return CellType::kCholinergic_NBM;

    return CellType::kGeneric;
  }

  static NTType ParseNTName(const std::string& name) {
    if (name == "ACh") return kACh;
    if (name == "GABA") return kGABA;
    if (name == "Glut") return kGlut;
    if (name == "DA") return kDA;
    if (name == "5HT") return k5HT;
    if (name == "OA") return kOA;
    return kACh;
  }

  // Parse "TypeName:fraction TypeName:fraction ..." into CellTypeFraction vector
  static std::vector<CellTypeFraction> ParseTypeFractions(const std::string& s) {
    std::vector<CellTypeFraction> result;
    size_t pos = 0;
    while (pos < s.size()) {
      while (pos < s.size() && s[pos] == ' ') pos++;
      size_t colon = s.find(':', pos);
      if (colon == std::string::npos) break;
      std::string name = s.substr(pos, colon - pos);
      pos = colon + 1;
      size_t end = s.find(' ', pos);
      if (end == std::string::npos) end = s.size();
      float frac = std::stof(s.substr(pos, end - pos));
      result.push_back({ParseCellTypeName(name), frac});
      pos = end;
    }
    return result;
  }

  // Parse "NTName:fraction NTName:fraction ..."
  static std::vector<RegionSpec::NTFraction> ParseNTFractions(const std::string& s) {
    std::vector<RegionSpec::NTFraction> result;
    size_t pos = 0;
    while (pos < s.size()) {
      while (pos < s.size() && s[pos] == ' ') pos++;
      size_t colon = s.find(':', pos);
      if (colon == std::string::npos) break;
      std::string name = s.substr(pos, colon - pos);
      pos = colon + 1;
      size_t end = s.find(' ', pos);
      if (end == std::string::npos) end = s.size();
      float frac = std::stof(s.substr(pos, end - pos));
      result.push_back({ParseNTName(name), frac});
      pos = end;
    }
    return result;
  }

  static Result<BrainSpec> Load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
      return MakeError(ErrorCode::kFileNotFound,
                       "Cannot open brain spec: " + path);
    }

    BrainSpec spec;
    char line[4096];

    // Temporary maps for indexed regions/projections/stimuli
    std::unordered_map<int, RegionSpec> regions;
    std::unordered_map<int, ProjectionSpec> projections;
    std::unordered_map<int, StimulusSpec> stimuli;

    while (fgets(line, sizeof(line), f)) {
      char* p = line;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '#' || *p == '\n' || *p == '\0') continue;

      // Strip trailing newline
      size_t len = strlen(p);
      while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r')) p[--len] = '\0';

      char* eq = strchr(p, '=');
      if (!eq) continue;

      std::string key(p, eq);
      std::string val(eq + 1);
      auto trim = [](std::string& s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
          s.erase(0, 1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
          s.pop_back();
      };
      trim(key);
      trim(val);

      try {
        // Global keys
        if (key == "name") { spec.name = val; continue; }
        if (key == "species") { spec.species = ParseSpecies(val); continue; }
        if (key == "seed") { spec.seed = static_cast<uint32_t>(std::stoul(val)); continue; }
        if (key == "weight_mean") { spec.global_weight_mean = std::stof(val); continue; }
        if (key == "weight_std") { spec.global_weight_std = std::stof(val); continue; }
        if (key == "background_mean") { spec.background_current_mean = std::stof(val); continue; }
        if (key == "background_std") { spec.background_current_std = std::stof(val); continue; }

        // Region keys: region.N.field
        if (key.starts_with("region.")) {
          size_t dot1 = 7;
          size_t dot2 = key.find('.', dot1);
          if (dot2 == std::string::npos) continue;
          int idx = std::stoi(key.substr(dot1, dot2 - dot1));
          std::string field = key.substr(dot2 + 1);

          auto& reg = regions[idx];
          if (field == "name") reg.name = val;
          else if (field == "n_neurons") reg.n_neurons = static_cast<uint32_t>(std::stoul(val));
          else if (field == "density") reg.internal_density = std::stof(val);
          else if (field == "nt") reg.default_nt = ParseNTName(val);
          else if (field == "types") reg.cell_types = ParseTypeFractions(val);
          else if (field == "nt_dist") reg.nt_distribution = ParseNTFractions(val);
          else if (field == "release_probability") reg.release_probability = std::stof(val);
          else if (field == "background_mean") reg.background_mean = std::stof(val);
          else if (field == "scan_neurons") reg.scan_neurons_path = val;
          else if (field == "scan_synapses") reg.scan_synapses_path = val;
          continue;
        }

        // Projection keys: projection.N.field
        if (key.starts_with("projection.")) {
          size_t dot1 = 11;
          size_t dot2 = key.find('.', dot1);
          if (dot2 == std::string::npos) continue;
          int idx = std::stoi(key.substr(dot1, dot2 - dot1));
          std::string field = key.substr(dot2 + 1);

          auto& proj = projections[idx];
          if (field == "from") proj.from_region = val;
          else if (field == "to") proj.to_region = val;
          else if (field == "density") proj.density = std::stof(val);
          else if (field == "nt") proj.nt_type = ParseNTName(val);
          else if (field == "weight_mean") proj.weight_mean = std::stof(val);
          else if (field == "weight_std") proj.weight_std = std::stof(val);
          else if (field == "release_probability") proj.release_probability = std::stof(val);
          continue;
        }

        // Stimulus keys: stimulus.N.field
        if (key.starts_with("stimulus.")) {
          size_t dot1 = 9;
          size_t dot2 = key.find('.', dot1);
          if (dot2 == std::string::npos) continue;
          int idx = std::stoi(key.substr(dot1, dot2 - dot1));
          std::string field = key.substr(dot2 + 1);

          auto& stim = stimuli[idx];
          if (field == "label") stim.label = val;
          else if (field == "region") stim.target_region = val;
          else if (field == "start") stim.start_ms = std::stof(val);
          else if (field == "end") stim.end_ms = std::stof(val);
          else if (field == "intensity") stim.intensity = std::stof(val);
          else if (field == "fraction") stim.fraction = std::stof(val);
          continue;
        }
      } catch (const std::exception&) {
        fclose(f);
        return MakeError(ErrorCode::kInvalidParam,
                         "Invalid value for '" + key + "': " + val);
      }
    }
    fclose(f);

    // Collect regions in order
    if (regions.empty()) {
      return MakeError(ErrorCode::kInvalidParam, "Brain spec has no regions");
    }
    int max_reg = 0;
    for (auto& [k, _] : regions) max_reg = std::max(max_reg, k);
    for (int i = 0; i <= max_reg; ++i) {
      if (regions.count(i)) {
        spec.regions.push_back(std::move(regions[i]));
      }
    }

    // Collect projections in order
    if (!projections.empty()) {
      int max_proj = 0;
      for (auto& [k, _] : projections) max_proj = std::max(max_proj, k);
      for (int i = 0; i <= max_proj; ++i) {
        if (projections.count(i)) {
          spec.projections.push_back(std::move(projections[i]));
        }
      }
    }

    // Collect stimuli in order
    if (!stimuli.empty()) {
      int max_stim = 0;
      for (auto& [k, _] : stimuli) max_stim = std::max(max_stim, k);
      for (int i = 0; i <= max_stim; ++i) {
        if (stimuli.count(i)) {
          spec.stimuli.push_back(std::move(stimuli[i]));
        }
      }
    }

    return spec;
  }

  // Collect ScanSource entries from a loaded BrainSpec.
  // Only regions with at least one scan path set are included.
  static std::vector<ScanSource> CollectScanSources(const BrainSpec& spec) {
    std::vector<ScanSource> sources;
    for (const auto& reg : spec.regions) {
      if (!reg.scan_neurons_path.empty() || !reg.scan_synapses_path.empty()) {
        sources.push_back({reg.name, reg.scan_neurons_path,
                           reg.scan_synapses_path});
      }
    }
    return sources;
  }
};

}  // namespace mechabrain

#endif  // FWMC_BRAIN_SPEC_LOADER_H_
