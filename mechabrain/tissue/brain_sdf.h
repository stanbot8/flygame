#ifndef FWMC_BRAIN_SDF_H_
#define FWMC_BRAIN_SDF_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "tissue/voxel_grid.h"

namespace mechabrain {

// Signed distance field primitives for constructing brain geometry.
// Negative = inside, positive = outside, zero = surface.
// The brain shape emerges from compositing ellipsoid primitives via
// smooth CSG unions, then refining with PDE-based smoothing (diffusion
// of the distance field itself, which rounds sharp junctions).

struct SDFPrimitive {
  std::string name;      // anatomical label
  float cx, cy, cz;      // center (um)
  float rx, ry, rz;      // radii (um)

  // Signed distance to an axis-aligned ellipsoid.
  // Uses the scaling trick: transform to unit sphere, compute distance,
  // scale back. Not exact SDF but good enough for smooth union.
  float Evaluate(float x, float y, float z) const {
    float dx = (x - cx) / rx;
    float dy = (y - cy) / ry;
    float dz = (z - cz) / rz;
    float r = std::sqrt(dx*dx + dy*dy + dz*dz);
    // Approximate SDF: scale by minimum radius for reasonable gradients
    float min_r = std::min({rx, ry, rz});
    return (r - 1.0f) * min_r;
  }
};

// Smooth minimum (polynomial). Blends two SDF values with smooth
// transition of width k. This creates organic-looking junctions
// between brain regions instead of sharp creases.
inline float SmoothMin(float a, float b, float k) {
  float h = std::max(k - std::abs(a - b), 0.0f) / k;
  return std::min(a, b) - h * h * h * k * (1.0f / 6.0f);
}

// Drosophila brain shape as a composite SDF.
// Built from ellipsoid primitives representing the major neuropil regions.
// Dimensions based on adult Drosophila melanogaster brain atlas
// (Ito et al. 2014, ~500um wide, ~300um tall, ~200um deep).
struct BrainSDF {
  std::vector<SDFPrimitive> primitives;
  float smooth_k = 15.0f;  // smooth union blend radius (um)

  // Initialize with Drosophila brain anatomy.
  // All coordinates in micrometers, centered at (250, 150, 100).
  void InitDrosophila() {
    primitives.clear();

    // Central brain (protocerebrum): the main mass
    primitives.push_back({"central_brain", 250, 150, 100, 120, 90, 70});

    // Optic lobes: large lateral structures for visual processing
    primitives.push_back({"optic_lobe_L", 80, 150, 100, 70, 80, 60});
    primitives.push_back({"optic_lobe_R", 420, 150, 100, 70, 80, 60});

    // Mushroom bodies: learning/memory centers, dorsal-posterior
    // Calyx (input region) is larger, pedunculus is elongated
    primitives.push_back({"mb_calyx_L", 180, 200, 120, 35, 30, 25});
    primitives.push_back({"mb_calyx_R", 320, 200, 120, 35, 30, 25});
    primitives.push_back({"mb_lobe_L", 180, 130, 80, 20, 40, 20});
    primitives.push_back({"mb_lobe_R", 320, 130, 80, 20, 40, 20});

    // Antennal lobes: olfactory processing, anterior-ventral
    primitives.push_back({"antennal_lobe_L", 200, 90, 60, 30, 25, 25});
    primitives.push_back({"antennal_lobe_R", 300, 90, 60, 30, 25, 25});

    // Central complex: navigation/motor coordination, midline
    primitives.push_back({"central_complex", 250, 170, 100, 30, 15, 15});

    // Lateral horn: innate olfactory behavior, lateral to MB
    primitives.push_back({"lateral_horn_L", 150, 180, 110, 25, 20, 20});
    primitives.push_back({"lateral_horn_R", 350, 180, 110, 25, 20, 20});

    // Subesophageal zone: gustatory/motor, ventral
    primitives.push_back({"sez", 250, 70, 90, 60, 40, 40});
  }

  // Initialize with larval zebrafish (Danio rerio) brain anatomy.
  // Coordinate units: micrometers. Brain centered at (350, 250, 100).
  // Total extent: ~700x500x200 um at 5-7 dpf.
  // Based on Z-Brain atlas (Randlett et al. 2015, Cell) and
  // Portugues et al. 2014, Robles et al. 2011.
  //
  // X = left-right (midline at 350 um)
  // Y = posterior-anterior (posterior=0, anterior=500)
  // Z = inferior-superior (inferior=0, superior=200)
  void InitZebrafish() {
    primitives.clear();

    // Optic tectum: largest midbrain structure, retinotopic visual map.
    // ~200x150x80 um per hemisphere (Nevin et al. 2010).
    primitives.push_back({"tectum_L",  250, 200, 140,  80, 75, 40});
    primitives.push_back({"tectum_R",  450, 200, 140,  80, 75, 40});

    // Retina: input to tectum, contralateral projection.
    // ~150 um diameter (Easter & Nicola 1996).
    primitives.push_back({"retina_L",  150, 350, 100,  75, 60, 50});
    primitives.push_back({"retina_R",  550, 350, 100,  75, 60, 50});

    // Cerebellum: motor learning, posterior dorsal.
    // ~120x100x60 um (Bae et al. 2009).
    primitives.push_back({"cerebellum", 350, 100, 150,  60, 50, 30});

    // Telencephalon: anterior, homolog of cortex/hippocampus.
    // ~80x60x50 um per hemisphere (Mueller & Wullimann 2009).
    primitives.push_back({"telencephalon_L",  280, 400, 120,  40, 30, 25});
    primitives.push_back({"telencephalon_R",  420, 400, 120,  40, 30, 25});

    // Pretectum: direction-selective neurons, prey capture.
    // ~60x40x40 um (Semmelhack et al. 2014).
    primitives.push_back({"pretectum_L",  280, 280, 130,  30, 20, 20});
    primitives.push_back({"pretectum_R",  420, 280, 130,  30, 20, 20});

    // Hindbrain: reticulospinal neurons, motor output.
    // ~200x80x60 um (Kinkhabwala et al. 2011).
    primitives.push_back({"hindbrain",  350, 80, 80,  50, 100, 30});

    // Habenula: dorsal diencephalon, learning from aversion.
    // ~30x20x20 um per side (Amo et al. 2010).
    primitives.push_back({"habenula_L",  320, 330, 160,  15, 10, 10});
    primitives.push_back({"habenula_R",  380, 330, 160,  15, 10, 10});

    // Hypothalamus: ventral, autonomic, feeding.
    primitives.push_back({"hypothalamus", 350, 300, 50,  30, 30, 25});

    // Spinal cord: extends posteriorly from hindbrain.
    primitives.push_back({"spinal_cord",  350, 10, 60,  15, 40, 15});
  }

  // Initialize with mouse brain anatomy.
  // Coordinate units: micrometers. Brain centered at (6000, 4500, 3500).
  // Total extent: ~12mm (LR) x ~9mm (AP) x ~7mm (SI).
  // Based on Allen Mouse Brain Atlas (Lein et al. 2007) and
  // Paxinos & Franklin (2012) stereotaxic atlas.
  //
  // X = left-right (midline at 6000 um)
  // Y = posterior-anterior (posterior=0, anterior=9000)
  // Z = inferior-superior (inferior=0, superior=7000)
  void InitMouse() {
    primitives.clear();

    // ---- Cerebral cortex (bilateral) ----

    // Somatosensory cortex (S1): barrel cortex, largest cortical area.
    // ~3x2x1 mm per hemisphere (Woolsey & Van der Loos 1970).
    primitives.push_back({"S1_L",  4000, 5500, 6200,  1500, 1000, 500});
    primitives.push_back({"S1_R",  8000, 5500, 6200,  1500, 1000, 500});

    // Motor cortex (M1): rostral to S1.
    // ~2x1.5x0.8 mm (Tennant et al. 2011).
    primitives.push_back({"M1_L",  4500, 7000, 6000,  1000, 750, 400});
    primitives.push_back({"M1_R",  7500, 7000, 6000,  1000, 750, 400});

    // Visual cortex (V1): posterior, binocular zone.
    // ~2x1.5x0.8 mm (Wang & Bhatt 2015).
    primitives.push_back({"V1_L",  4500, 2000, 5800,  1000, 750, 400});
    primitives.push_back({"V1_R",  7500, 2000, 5800,  1000, 750, 400});

    // Auditory cortex (A1): lateral temporal.
    // ~1x1x0.6 mm (Stiebler et al. 1997).
    primitives.push_back({"A1_L",  3000, 4500, 5500,  500, 500, 300});
    primitives.push_back({"A1_R",  9000, 4500, 5500,  500, 500, 300});

    // Prefrontal/cingulate: medial frontal.
    primitives.push_back({"PFC_L", 5200, 8000, 5500,  600, 500, 400});
    primitives.push_back({"PFC_R", 6800, 8000, 5500,  600, 500, 400});

    // ---- Subcortical structures ----

    // Thalamus: central relay, ~2x2x1.5 mm.
    // VPM barreloids relay whisker info to S1 (Haidarliu & Bhatt 2015).
    primitives.push_back({"thalamus_L",  5000, 4500, 4000,  800, 1000, 700});
    primitives.push_back({"thalamus_R",  7000, 4500, 4000,  800, 1000, 700});

    // Hippocampus: dorsal arc, ~4x1x1 mm.
    // Episodic memory, spatial navigation (Moser et al. 2008).
    primitives.push_back({"hippocampus_L",  4000, 3500, 4500,  1000, 2000, 500});
    primitives.push_back({"hippocampus_R",  8000, 3500, 4500,  1000, 2000, 500});

    // Striatum (caudate-putamen): ~2x2x2 mm.
    // Motor planning, reward learning (Kreitzer & Malenka 2008).
    primitives.push_back({"striatum_L",  4500, 6000, 4500,  800, 1000, 800});
    primitives.push_back({"striatum_R",  7500, 6000, 4500,  800, 1000, 800});

    // Amygdala: emotion, fear conditioning.
    // ~1x1x1 mm (Herry & Bhatt 2010).
    primitives.push_back({"amygdala_L",  3800, 4000, 3200,  500, 500, 500});
    primitives.push_back({"amygdala_R",  8200, 4000, 3200,  500, 500, 500});

    // Hypothalamus: midline, autonomic regulation.
    primitives.push_back({"hypothalamus",  6000, 5000, 2800,  400, 600, 400});

    // ---- Cerebellum ----
    // ~6x3x3 mm (Sillitoe & Bhatt 2007).
    primitives.push_back({"cerebellum_L",  4500, 1500, 3000,  1500, 1200, 1200});
    primitives.push_back({"cerebellum_R",  7500, 1500, 3000,  1500, 1200, 1200});

    // ---- Brainstem ----
    primitives.push_back({"midbrain",   6000, 3500, 3000,  600, 600, 500});
    primitives.push_back({"pons",       6000, 2500, 2500,  700, 600, 500});
    primitives.push_back({"medulla",    6000, 1500, 2000,  500, 800, 500});

    // Olfactory bulb: rostral, large in mouse.
    // ~2x2x2 mm (Bhatt et al. 2020).
    primitives.push_back({"olfactory_bulb_L",  4500, 8500, 4500,  700, 500, 700});
    primitives.push_back({"olfactory_bulb_R",  7500, 8500, 4500,  700, 500, 700});
  }

  // Initialize with human brain anatomy.
  // Coordinate units: millimeters. Brain centered at (70, 85, 65).
  // Total extent: ~140mm (LR) x ~170mm (AP) x ~130mm (SI).
  // Regions derived from MNI-152 atlas parcellations and standard
  // neuroanatomy (Mai et al. 2015, Duvernoy 2012, Talairach & Tournoux 1988).
  //
  // X = left-right (midline at 70mm)
  // Y = posterior-anterior (posterior=0, anterior=170)
  // Z = inferior-superior (inferior=0, superior=130)
  //
  // The LOD system should use mm-scale zones for human brain:
  //   Compartmental: ~5mm radius (single cortical column)
  //   Neuron:        ~20mm radius (cortical area)
  //   Region:        ~50mm radius (lobe-level)
  //   Continuum:     beyond 50mm
  void InitHuman() {
    primitives.clear();

    // ---- Cerebral cortex (6 lobes, bilateral) ----
    // Cortical lobes modeled as large ellipsoids. The cortex is ~2-4mm thick
    // but here each lobe is a solid region; the LOD system spawns neurons
    // within the volume. At human scale most regions stay at LOD 0 (field)
    // with only the focus region escalating to spiking.

    // Frontal lobe: largest lobe, anterior. Motor cortex, prefrontal.
    // ~70mm AP x 60mm SI x 35mm LR per hemisphere (Rademacher et al. 1992)
    primitives.push_back({"frontal_L",   40, 125, 80,  25, 35, 30});
    primitives.push_back({"frontal_R",  100, 125, 80,  25, 35, 30});

    // Prefrontal cortex: anterior portion of frontal lobe.
    // Working memory, executive function (Goldman-Rakic 1995).
    primitives.push_back({"prefrontal_L",  40, 155, 75,  20, 15, 25});
    primitives.push_back({"prefrontal_R", 100, 155, 75,  20, 15, 25});

    // Motor cortex: precentral gyrus, dorsal frontal.
    // Primary motor area (Penfield & Boldrey 1937).
    primitives.push_back({"motor_L",   45, 100, 100,  20, 8, 20});
    primitives.push_back({"motor_R",   95, 100, 100,  20, 8, 20});

    // Parietal lobe: somatosensory, spatial awareness.
    // ~50mm AP x 55mm SI x 30mm LR per hemisphere.
    primitives.push_back({"parietal_L",   40, 80, 95,  25, 25, 25});
    primitives.push_back({"parietal_R",  100, 80, 95,  25, 25, 25});

    // Temporal lobe: auditory, language (Wernicke), memory.
    // ~70mm AP x 25mm SI x 30mm LR, lateral and inferior.
    primitives.push_back({"temporal_L",   30, 85, 40,  18, 35, 18});
    primitives.push_back({"temporal_R",  110, 85, 40,  18, 35, 18});

    // Occipital lobe: visual cortex, posterior.
    // ~40mm AP x 45mm SI x 25mm LR per hemisphere.
    primitives.push_back({"occipital_L",   45, 25, 70,  20, 20, 25});
    primitives.push_back({"occipital_R",   95, 25, 70,  20, 20, 25});

    // Insular cortex: deep to lateral sulcus, interoception.
    // ~40mm AP x 25mm SI x 5mm LR (Naidich et al. 2004).
    primitives.push_back({"insula_L",   38, 100, 55,   6, 20, 15});
    primitives.push_back({"insula_R",  102, 100, 55,   6, 20, 15});

    // Cingulate cortex: medial surface, limbic. Anterior + posterior.
    // Error monitoring, emotion regulation (Bush et al. 2000).
    primitives.push_back({"cingulate_L",  62, 110, 85,   5, 30, 15});
    primitives.push_back({"cingulate_R",  78, 110, 85,   5, 30, 15});

    // ---- Subcortical structures (bilateral) ----

    // Thalamus: central relay station, ~30mm AP x 20mm SI x 15mm LR.
    // All sensory pathways except olfaction (Jones 2007).
    primitives.push_back({"thalamus_L",   58, 78, 58,   7, 13, 10});
    primitives.push_back({"thalamus_R",   82, 78, 58,   7, 13, 10});

    // Caudate nucleus: dorsal striatum, learning.
    // C-shaped, ~50mm along its curve (Yelnik 2002).
    primitives.push_back({"caudate_L",   55, 100, 65,   5, 18, 8});
    primitives.push_back({"caudate_R",   85, 100, 65,   5, 18, 8});

    // Putamen: dorsal striatum, motor planning.
    // ~30mm AP x 20mm SI x 10mm LR (Yelnik 2002).
    primitives.push_back({"putamen_L",   48, 95, 55,    6, 15, 10});
    primitives.push_back({"putamen_R",   92, 95, 55,    6, 15, 10});

    // Globus pallidus: output nucleus of basal ganglia.
    primitives.push_back({"pallidum_L",  50, 85, 52,    5, 10, 7});
    primitives.push_back({"pallidum_R",  90, 85, 52,    5, 10, 7});

    // Hippocampus: episodic memory, spatial navigation.
    // ~45mm AP x 10mm LR x 10mm SI (Duvernoy 2005).
    primitives.push_back({"hippocampus_L",  48, 65, 38,   6, 22, 7});
    primitives.push_back({"hippocampus_R",  92, 65, 38,   6, 22, 7});

    // Amygdala: emotion, fear conditioning.
    // ~15mm each dimension (Schumann & Amaral 2005).
    primitives.push_back({"amygdala_L",   48, 80, 35,   7, 8, 8});
    primitives.push_back({"amygdala_R",   92, 80, 35,   7, 8, 8});

    // Hypothalamus: autonomic regulation, homeostasis.
    // Small midline structure, ~10mm each dimension.
    primitives.push_back({"hypothalamus", 70, 90, 42,   6, 7, 5});

    // Subthalamic nucleus: basal ganglia circuit node.
    primitives.push_back({"stn_L",  56, 82, 45,  3, 5, 3});
    primitives.push_back({"stn_R",  84, 82, 45,  3, 5, 3});

    // Nucleus accumbens: ventral striatum, reward.
    primitives.push_back({"accumbens_L",  55, 108, 42,  4, 5, 4});
    primitives.push_back({"accumbens_R",  85, 108, 42,  4, 5, 4});

    // ---- Cerebellum ----
    // ~100mm LR x 55mm AP x 50mm SI (Schmahmann et al. 1999).
    // Lateral hemispheres + vermis.
    primitives.push_back({"cerebellum_L",   40, 30, 25,  28, 25, 22});
    primitives.push_back({"cerebellum_R",  100, 30, 25,  28, 25, 22});
    primitives.push_back({"vermis",         70, 30, 28,   8, 20, 18});

    // Deep cerebellar nuclei (dentate, interposed, fastigial).
    primitives.push_back({"dentate_L",  50, 35, 30,  6, 6, 5});
    primitives.push_back({"dentate_R",  90, 35, 30,  6, 6, 5});

    // ---- Brainstem ----
    // Midbrain + pons + medulla, ~25mm diameter x 70mm long.
    primitives.push_back({"midbrain",       70, 65, 38,   8, 8, 6});
    primitives.push_back({"pons",           70, 55, 30,  10, 10, 8});
    primitives.push_back({"medulla",        70, 50, 18,   7, 8, 10});

    // ---- White matter tracts (modeled as connective volumes) ----
    // Corpus callosum: ~80mm AP x 5mm SI x 8mm LR midline.
    // Largest commissure, connects hemispheres (Aboitiz et al. 1992).
    primitives.push_back({"corpus_callosum", 70, 90, 78,  5, 35, 4});

    // Internal capsule: carries corticofugal and thalamocortical fibers.
    primitives.push_back({"int_capsule_L",  52, 90, 58,   4, 15, 12});
    primitives.push_back({"int_capsule_R",  88, 90, 58,   4, 15, 12});
  }

  // Evaluate the composite SDF at a point.
  // Smooth union of all primitives.
  float Evaluate(float x, float y, float z) const {
    if (primitives.empty()) return 1.0f;
    float d = primitives[0].Evaluate(x, y, z);
    for (size_t i = 1; i < primitives.size(); ++i) {
      d = SmoothMin(d, primitives[i].Evaluate(x, y, z), smooth_k);
    }
    return d;
  }

  // Which region is closest to a point? Returns index into primitives,
  // or -1 if outside all regions.
  int NearestRegion(float x, float y, float z) const {
    // Among all primitives the point is inside, pick the smallest
    // (most specific) one. This prevents large regions like central_brain
    // from swallowing smaller sub-regions like mushroom body lobes.
    int best = -1;
    float best_vol = 1e30f;
    float closest_d = 1e30f;
    for (size_t i = 0; i < primitives.size(); ++i) {
      auto& p = primitives[i];
      float d = p.Evaluate(x, y, z);
      if (d <= 0.0f) {
        // Point is inside this primitive; prefer smaller volume
        float vol = p.rx * p.ry * p.rz;
        if (vol < best_vol) {
          best_vol = vol;
          best = static_cast<int>(i);
        }
      } else if (best < 0 && d < closest_d) {
        // Fallback: if not inside any, track the closest
        closest_d = d;
        best = static_cast<int>(i);
      }
    }
    return (best_vol < 1e30f) ? best : -1;
  }

  // Bake the SDF into a VoxelGrid channel.
  // Also creates a "region_id" channel mapping each voxel to its
  // nearest anatomical region (or -1 if outside).
  void BakeToGrid(VoxelGrid& grid, size_t sdf_channel,
                  size_t region_channel) const {
    for (uint32_t z = 0; z < grid.nz; ++z) {
      for (uint32_t y = 0; y < grid.ny; ++y) {
        for (uint32_t x = 0; x < grid.nx; ++x) {
          float wx, wy, wz;
          grid.GridToWorld(x, y, z, wx, wy, wz);
          size_t idx = grid.Idx(x, y, z);
          grid.channels[sdf_channel].data[idx] = Evaluate(wx, wy, wz);
          grid.channels[region_channel].data[idx] =
              static_cast<float>(NearestRegion(wx, wy, wz));
        }
      }
    }
  }

  // Smooth the baked SDF using diffusion (Laplacian smoothing).
  // This rounds sharp junctions between primitives, making the
  // brain surface organic rather than a union of lumps.
  // Operates only on voxels near the surface (|sdf| < band).
  static void DiffuseSmooth(VoxelGrid& grid, size_t sdf_channel,
                            int iterations, float band_um = 30.0f) {
    auto& data = grid.channels[sdf_channel].data;
    std::vector<float> temp(data.size());

    for (int iter = 0; iter < iterations; ++iter) {
      for (uint32_t z = 0; z < grid.nz; ++z) {
        for (uint32_t y = 0; y < grid.ny; ++y) {
          for (uint32_t x = 0; x < grid.nx; ++x) {
            size_t i = grid.Idx(x, y, z);
            float c = data[i];

            // Only smooth near the surface
            if (std::abs(c) > band_um) {
              temp[i] = c;
              continue;
            }

            float xm = (x > 0)          ? data[grid.Idx(x-1,y,z)] : c;
            float xp = (x < grid.nx-1)  ? data[grid.Idx(x+1,y,z)] : c;
            float ym = (y > 0)          ? data[grid.Idx(x,y-1,z)] : c;
            float yp = (y < grid.ny-1)  ? data[grid.Idx(x,y+1,z)] : c;
            float zm = (z > 0)          ? data[grid.Idx(x,y,z-1)] : c;
            float zp = (z < grid.nz-1)  ? data[grid.Idx(x,y,z+1)] : c;

            // Laplacian smoothing (1/6 weight to neighbors)
            temp[i] = (xm + xp + ym + yp + zm + zp) / 6.0f;
          }
        }
      }
      data.swap(temp);
    }
  }

  // Compute surface normal at a point via central differences on the SDF.
  void Normal(float x, float y, float z, float eps,
              float& nx_out, float& ny_out, float& nz_out) const {
    float gx = Evaluate(x + eps, y, z) - Evaluate(x - eps, y, z);
    float gy = Evaluate(x, y + eps, z) - Evaluate(x, y - eps, z);
    float gz = Evaluate(x, y, z + eps) - Evaluate(x, y, z - eps);
    float len = std::sqrt(gx*gx + gy*gy + gz*gz);
    if (len > 1e-10f) { gx /= len; gy /= len; gz /= len; }
    nx_out = gx; ny_out = gy; nz_out = gz;
  }
};

}  // namespace mechabrain

#endif  // FWMC_BRAIN_SDF_H_
