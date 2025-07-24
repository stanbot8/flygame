#ifndef FWMC_LOD_MANAGER_H_
#define FWMC_LOD_MANAGER_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/species.h"

namespace mechabrain {

// Level-of-detail manager for multi-scale brain simulation.
// Assigns each spatial region an LOD level based on distance from
// a focus point (camera position or algorithmic interest).
//
// LOD 0: Continuum (neural field PDE on voxel grid)
// LOD 1: Region-level (population mean-field statistics)
// LOD 2: Neuron-level (individual Izhikevich neurons + CSR synapses)
// LOD 3: Compartmental (multi-compartment morphology, ion channels)
//
// Escalation: when a region transitions from a coarser to a finer LOD,
// the finer model is initialized from the coarser state.
// De-escalation: when a region moves away from focus, its fine-grained
// state is collapsed back into the coarser representation.

enum class LODLevel : uint8_t {
  kContinuum = 0,
  kRegion = 1,
  kNeuron = 2,
  kCompartmental = 3
};

struct LODZone {
  float radius;        // distance from focus within which this LOD applies (um)
  LODLevel level;
};

struct LODManager {
  // Focus point in world coordinates (um).
  float focus_x = 250.0f;
  float focus_y = 150.0f;
  float focus_z = 100.0f;

  // LOD zones: sorted innermost to outermost.
  // Default: compartmental within 30um, neuron within 100um,
  // region within 200um, continuum beyond.
  std::vector<LODZone> zones = {
    {30.0f,  LODLevel::kCompartmental},
    {100.0f, LODLevel::kNeuron},
    {200.0f, LODLevel::kRegion},
  };
  LODLevel default_level = LODLevel::kContinuum;

  // Hysteresis band (um). A region must move this far beyond a zone
  // boundary before de-escalating. Prevents flickering at boundaries.
  float hysteresis = 10.0f;

  // Determine LOD level for a point in world coordinates.
  LODLevel GetLOD(float x, float y, float z) const {
    float dx = x - focus_x;
    float dy = y - focus_y;
    float dz = z - focus_z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    for (const auto& zone : zones) {
      if (dist <= zone.radius) return zone.level;
    }
    return default_level;
  }

  // Determine LOD with hysteresis (needs previous LOD to decide).
  LODLevel GetLODWithHysteresis(float x, float y, float z,
                                 LODLevel current) const {
    float dx = x - focus_x;
    float dy = y - focus_y;
    float dz = z - focus_z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Find what the new LOD would be without hysteresis
    LODLevel raw = default_level;
    for (const auto& zone : zones) {
      if (dist <= zone.radius) { raw = zone.level; break; }
    }

    // If raw wants to de-escalate (coarser), require extra distance
    if (raw < current) {
      // Check if we're far enough past the boundary
      for (const auto& zone : zones) {
        if (zone.level == current) {
          if (dist <= zone.radius + hysteresis) return current;
          break;
        }
      }
    }

    return raw;
  }

  // Move the focus point.
  void SetFocus(float x, float y, float z) {
    focus_x = x; focus_y = y; focus_z = z;
  }

  // Per-region LOD tracking.
  struct RegionLOD {
    std::string name;
    float center_x, center_y, center_z;
    LODLevel current_lod = LODLevel::kContinuum;
  };
  std::vector<RegionLOD> region_lods;

  // Update all tracked regions' LOD levels. Returns number of transitions.
  int UpdateAll() {
    int transitions = 0;
    for (auto& r : region_lods) {
      LODLevel next = GetLODWithHysteresis(
          r.center_x, r.center_y, r.center_z, r.current_lod);
      if (next != r.current_lod) {
        r.current_lod = next;
        transitions++;
      }
    }
    return transitions;
  }

  // Configure LOD zones and focus for a given species.
  // Brain sizes and zone radii grounded in literature:
  //
  //   Drosophila: ~500x300x200 um brain (Ito et al. 2014)
  //     - Compartmental: 30 um (single neuropil like MB calyx)
  //     - Neuron: 100 um (one brain region)
  //     - Region: 200 um (half-brain)
  //     - Hysteresis: 10 um
  //
  //   Zebrafish (larval): ~700x400x200 um (Randlett et al. 2015)
  //     - Compartmental: 50 um (optic tectum layer)
  //     - Neuron: 150 um (one lobe)
  //     - Region: 300 um (whole tectum)
  //     - Hysteresis: 15 um
  //
  //   Mouse: ~12x6x8 mm cortex (Paxinos & Franklin 2012)
  //     - Compartmental: 500 um (single barrel in S1, ~300-500 um)
  //     - Neuron: 2000 um (one cortical area)
  //     - Region: 5000 um (multiple areas / one lobe)
  //     - Hysteresis: 200 um
  //
  //   Rat: ~15x8x10 mm cortex (Paxinos & Watson 2007)
  //     - Compartmental: 600 um
  //     - Neuron: 2500 um
  //     - Region: 6000 um
  //     - Hysteresis: 250 um
  //
  //   Human: ~170x140x130 mm cerebrum (Fischl et al. 2002)
  //     NOTE: human brain SDF uses mm coordinates, so zones are in mm.
  //     - Compartmental: 2 mm (~cortical column, Mountcastle 1997)
  //     - Neuron: 10 mm (one Brodmann area)
  //     - Region: 30 mm (one lobe)
  //     - Hysteresis: 2 mm
  //
  void ConfigureForSpecies(Species species) {
    switch (species) {
      case Species::kDrosophila:
        zones = {
          {30.0f,    LODLevel::kCompartmental},
          {100.0f,   LODLevel::kNeuron},
          {200.0f,   LODLevel::kRegion},
        };
        hysteresis = 10.0f;
        // Center of Drosophila central brain
        focus_x = 250.0f; focus_y = 150.0f; focus_z = 100.0f;
        break;

      case Species::kZebrafish:
        zones = {
          {50.0f,    LODLevel::kCompartmental},
          {150.0f,   LODLevel::kNeuron},
          {300.0f,   LODLevel::kRegion},
        };
        hysteresis = 15.0f;
        focus_x = 350.0f; focus_y = 200.0f; focus_z = 100.0f;
        break;

      case Species::kMouse:
        zones = {
          {500.0f,   LODLevel::kCompartmental},
          {2000.0f,  LODLevel::kNeuron},
          {5000.0f,  LODLevel::kRegion},
        };
        hysteresis = 200.0f;
        // Center of mouse brain (um, matches brain_sdf.h InitMouse)
        focus_x = 6000.0f; focus_y = 4500.0f; focus_z = 3500.0f;
        break;

      case Species::kRat:
        zones = {
          {600.0f,   LODLevel::kCompartmental},
          {2500.0f,  LODLevel::kNeuron},
          {6000.0f,  LODLevel::kRegion},
        };
        hysteresis = 250.0f;
        focus_x = 7500.0f; focus_y = 4000.0f; focus_z = 5000.0f;
        break;

      case Species::kHuman:
        // Human brain SDF uses mm coordinates (not um).
        // Zone radii and focus must match the SDF coordinate system.
        zones = {
          {2.0f,   LODLevel::kCompartmental},
          {10.0f,  LODLevel::kNeuron},
          {30.0f,  LODLevel::kRegion},
        };
        hysteresis = 2.0f;
        // Center of human cerebrum in mm (matches brain_sdf.h InitHuman)
        focus_x = 70.0f; focus_y = 85.0f; focus_z = 65.0f;
        break;

      case Species::kGeneric:
      default:
        // Keep current zones unchanged
        break;
    }
  }
};

}  // namespace mechabrain

#endif  // FWMC_LOD_MANAGER_H_
