#ifndef FWMC_CELL_TYPE_DEFS_H_
#define FWMC_CELL_TYPE_DEFS_H_

#include <cstdint>
#include "core/izhikevich.h"

namespace mechabrain {

// Functional cell type within a region.
// Lightweight enum + parameter lookup, no heavy dependencies.
enum class CellType : uint8_t {
  kGeneric = 0,
  kKenyonCell = 1,
  kMBON_cholinergic = 2,
  kMBON_gabaergic = 3,
  kMBON_glutamatergic = 4,
  kDAN_PPL1 = 5,
  kDAN_PAM = 6,
  kPN_excitatory = 7,
  kPN_inhibitory = 8,
  kLN_local = 9,        // local interneuron
  kORN = 10,            // olfactory receptor neuron
  kFastSpiking = 11,
  kBursting = 12,
  kSerotonergic = 13,   // 5HT neurons (CSD, IP, SE clusters)
  kOctopaminergic = 14, // Tdc2+ neurons (arousal, flight, aggression)

  // ---- Central Complex cell types (25-31) ----
  // Columnar and tangential neurons of the CX compass/steering circuit.
  // Parameters fit to CX calcium imaging: 2-20 Hz tonic (Seelig & Jayaraman
  // 2015 Nature), sustained bump dynamics (Kim et al. 2017 Science),
  // electrophysiology in locust CX (Heinze & Homberg 2007 J Neurosci).
  kEPG = 25,            // E-PG: ellipsoid body -> protocerebral bridge, compass
  kPEN = 26,            // P-EN: protocerebral bridge -> ellipsoid body, angular velocity
  kPEG = 27,            // P-EG: protocerebral bridge -> ellipsoid body, gain
  kPFL = 28,            // PFL: protocerebral bridge -> fan-shaped body -> LAL, steering
  kDelta7 = 29,         // Delta7: inhibitory ring neuron, winner-take-all in PB
  kRingNeuron = 30,     // R neurons (R1-R4d): EB ring neurons, visual input
  kFC = 31,             // Fan-shaped columnar neuron, FB layers

  // ---- Zebrafish cell types (32-36) ----
  // Larval zebrafish (6-7 dpf) optic tectum and motor circuit.
  // Parameters fit to zebrafish whole-cell patch clamp: Bhatt et al. 2007
  // J Neurophysiol (reticulospinal), Niell & Smith 2005 Neuron (tectal),
  // Zhang et al. 2021 Current Biology (tectal PVN calcium dynamics).
  // Resting potential ~-55 to -65 mV, spike peak ~+25 mV.
  kRGC = 32,              // retinal ganglion cell, transient ON/OFF spiking
  kTectalPVN = 33,        // periventricular neuron (main tectal excitatory)
  kTectalSIN = 34,        // superficial interneuron (GABAergic, fast-spiking)
  kReticulospinal = 35,   // hindbrain reticulospinal (Mauthner-like, high threshold)
  kGranuleCell = 36,      // cerebellar granule cell (small soma, regular spiking)

  // ---- Grooming circuit cell types (37-39) ----
  // Drosophila grooming hierarchy (Seeds et al. 2014 eLife).
  // Descending neurons (DNg) that command specific grooming movements.
  kDNg_command = 37,        // descending grooming command neuron (cholinergic)
  kGroomingSensory = 38,    // mechanosensory neuron detecting dust/irritant
  kGroomingInhibitory = 39, // GABAergic inhibitory interneuron (suppression hierarchy)

  // ---- Mammalian cortical cell types (15-24) ----
  // For scaling from fly to human brain. Parameters from Izhikevich (2003)
  // canonical forms, validated against cortical electrophysiology.
  kL23_Pyramidal = 15,      // L2/3 regular spiking pyramidal (glutamatergic)
  kL4_Stellate = 16,        // L4 spiny stellate (glutamatergic)
  kL5_Pyramidal = 17,       // L5 thick-tufted pyramidal, intrinsic bursting (glutamatergic)
  kL6_Pyramidal = 18,       // L6 corticothalamic pyramidal (glutamatergic)
  kPV_Basket = 19,          // PV+ fast-spiking basket cell (GABAergic)
  kSST_Martinotti = 20,     // SST+ Martinotti cell, low-threshold spiking (GABAergic)
  kVIP_Interneuron = 21,    // VIP+ bipolar interneuron, irregular spiking (GABAergic)
  kThalamocortical = 22,    // TC relay neuron, tonic mode (glutamatergic)
  kTRN = 23,                // thalamic reticular nucleus (GABAergic)
  kCholinergic_NBM = 24,    // nucleus basalis of Meynert cholinergic projection
};

// Per-cell-type Izhikevich parameters.
//
// Drosophila types (0-14):
//   Resting potential: ~-60 mV (Wilson & Laurent 2005; Gu & O'Dowd 2006)
//   Spike peak:        ~+20 mV (lower than mammalian +30 mV)
//   Input resistance:  ~0.6-3 GOhm (vs mammalian ~100 MOhm)
//   Membrane tau:      ~5-20 ms (vs mammalian 20-30 ms)
//   References:
//     Gu & O'Dowd (2006) J Neurosci 26:265
//     Gouwens & Wilson (2009) J Neurosci 29:6239
//     Chou et al (2010) J Neurophysiol 104:1006
//     Nagel & Wilson (2011) J Neurosci 31:772
//
// Mammalian cortical types (15-24):
//   Resting potential: ~-65 mV
//   Spike peak:        ~+30 mV
//   Input resistance:  ~50-300 MOhm
//   Membrane tau:      ~15-30 ms
//   References:
//     Izhikevich (2003) IEEE Trans Neural Netw 14:1569  canonical forms
//     Izhikevich & Edelman (2008) PNAS 105:3593         thalamocortical model
//     Markram et al (2015) Cell 163:456                  Blue Brain column
//     Testa-Silva et al (2014) Cereb Cortex 24:541       human L2/3 pyramidal
//     Beaulieu-Laroche et al (2018) Cell 175:643         human L5 pyramidal
//     Hu et al (2014) J Neurosci 34:15509                PV fast-spiking basket
//     Silberberg & Markram (2007) J Physiol 583:891      SST Martinotti cells
//     Prönneke et al (2015) Cereb Cortex 25:4854         VIP interneuron types
//     McCormick & Huguenard (1992) J Neurophysiol 68:1384 thalamocortical relay
//     Pinault (2004) Brain Res Rev 46:1                  TRN bursting/tonic
inline IzhikevichParams ParamsForCellType(CellType ct) {
  switch (ct) {
    case CellType::kKenyonCell:
      return {0.02f, 0.25f, -60.0f, 10.0f, 20.0f, 2.0f};
    case CellType::kPN_excitatory:
      return {0.03f, 0.20f, -58.0f, 6.0f, 20.0f, 1.5f};
    case CellType::kPN_inhibitory:
      return {0.05f, 0.20f, -58.0f, 4.0f, 20.0f, 1.5f};
    case CellType::kLN_local:
      return {0.10f, 0.20f, -60.0f, 2.0f, 20.0f, 1.0f};
    case CellType::kORN:
      return {0.03f, 0.20f, -60.0f, 5.0f, 20.0f, 2.0f};
    case CellType::kMBON_cholinergic:
      return {0.03f, 0.20f, -57.0f, 5.0f, 20.0f, 1.5f};
    case CellType::kMBON_gabaergic:
      return {0.03f, 0.20f, -58.0f, 6.0f, 20.0f, 1.5f};
    case CellType::kMBON_glutamatergic:
      return {0.02f, 0.20f, -58.0f, 7.0f, 20.0f, 2.0f};
    case CellType::kDAN_PPL1:
      return {0.02f, 0.25f, -60.0f, 6.0f, 20.0f, 2.0f};
    case CellType::kDAN_PAM:
      return {0.03f, 0.20f, -60.0f, 4.0f, 20.0f, 2.0f};
    case CellType::kFastSpiking:
      return {0.10f, 0.20f, -60.0f, 2.0f, 20.0f, 0.8f};
    case CellType::kBursting:
      return {0.02f, 0.20f, -50.0f, 2.0f, 20.0f, 0.5f};
    case CellType::kSerotonergic:
      return {0.02f, 0.25f, -60.0f, 8.0f, 20.0f, 2.0f};
    case CellType::kOctopaminergic:
      return {0.03f, 0.20f, -58.0f, 5.0f, 20.0f, 1.5f};

    // ---- Central Complex types ----
    // E-PG compass neurons: tonic 5-15 Hz, sustained bump (Seelig &
    // Jayaraman 2015). Regular spiking, moderate adaptation.
    case CellType::kEPG:
      return {0.02f, 0.20f, -60.0f, 6.0f, 20.0f, 1.5f};

    // P-EN angular velocity neurons: slightly faster than E-PG,
    // encode head turns (Green et al. 2017, Turner-Evans et al. 2017).
    case CellType::kPEN:
      return {0.03f, 0.20f, -60.0f, 5.0f, 20.0f, 1.2f};

    // P-EG gain neurons: similar to E-PG but with less adaptation.
    case CellType::kPEG:
      return {0.02f, 0.20f, -60.0f, 5.0f, 20.0f, 1.5f};

    // PFL steering neurons: higher excitability, drive motor output
    // via LAL (lateral accessory lobe). Slightly bursty (lower c).
    case CellType::kPFL:
      return {0.02f, 0.20f, -55.0f, 4.0f, 20.0f, 1.0f};

    // Delta7: inhibitory interneurons in PB, mediate ring attractor
    // winner-take-all. Fast spiking for sharp lateral inhibition
    // (Heinze & Homberg 2007, Hulse et al. 2021).
    case CellType::kDelta7:
      return {0.10f, 0.20f, -60.0f, 2.0f, 20.0f, 0.8f};

    // Ring neurons (R1-R4d): EB tangential neurons, convey visual
    // landmarks, inhibitory. GABAergic (Omoto et al. 2017 eLife).
    case CellType::kRingNeuron:
      return {0.05f, 0.20f, -60.0f, 4.0f, 20.0f, 1.0f};

    // Fan-shaped columnar: FB layer neurons, integrate heading +
    // visual features. Regular spiking (Hulse et al. 2021).
    case CellType::kFC:
      return {0.02f, 0.20f, -60.0f, 6.0f, 20.0f, 1.5f};

    // ---- Zebrafish types ----
    // v_peak = 25 mV (intermediate between fly 20 mV and mammal 30 mV).
    // Zebrafish larval neurons are small (soma ~5-10 um) with high input
    // resistance (~1-3 GOhm) and fast membrane time constants (~5-15 ms).

    // RGC: transient ON/OFF responses, 10-40 Hz during stimulation.
    // Nikolaou et al. 2012 Neuron: zebrafish RGC functional types.
    case CellType::kRGC:
      return {0.03f, 0.20f, -60.0f, 5.0f, 25.0f, 1.5f};

    // Tectal PVN: main excitatory neuron, regular spiking 5-20 Hz.
    // Niell & Smith 2005: direction-selective tectal responses.
    case CellType::kTectalPVN:
      return {0.02f, 0.20f, -60.0f, 6.0f, 25.0f, 1.5f};

    // Tectal SIN: GABAergic superficial interneuron, fast spiking.
    // Del Bene et al. 2010 Science: gain control for size selectivity.
    case CellType::kTectalSIN:
      return {0.10f, 0.20f, -60.0f, 2.0f, 25.0f, 0.8f};

    // Reticulospinal: large hindbrain motor neurons (Mauthner array).
    // High threshold, burst on strong input (Bhatt et al. 2007).
    // c=-55 for burst capability, high d for adaptation after escape.
    case CellType::kReticulospinal:
      return {0.02f, 0.20f, -55.0f, 8.0f, 25.0f, 1.0f};

    // Cerebellar granule cell: small soma, regular spiking, low rate.
    // Sengupta & bhatt 2017: zebrafish cerebellar circuit.
    case CellType::kGranuleCell:
      return {0.02f, 0.25f, -65.0f, 6.0f, 25.0f, 2.0f};

    // ---- Grooming circuit types ----
    // DNg command neurons: tonic regular spiking, moderate adaptation.
    // Seeds et al. 2014: distinct DNg groups drive specific grooming actions.
    // Hampel et al. 2015: DNg11/DNg12 for antennal grooming.
    case CellType::kDNg_command:
      return {0.02f, 0.20f, -60.0f, 6.0f, 20.0f, 1.5f};

    // Grooming mechanosensory: transient ON response to dust stimuli.
    // Fast recovery, low adaptation for sustained signaling during contact.
    case CellType::kGroomingSensory:
      return {0.03f, 0.20f, -60.0f, 4.0f, 20.0f, 1.5f};

    // Grooming inhibitory: fast-spiking GABAergic interneurons.
    // Mediate suppression hierarchy (higher priority -> inhibits lower).
    case CellType::kGroomingInhibitory:
      return {0.10f, 0.20f, -60.0f, 2.0f, 20.0f, 0.8f};

    // ---- Mammalian cortical types ----
    // v_peak = 30 mV (mammalian cortical spike amplitude, vs 20 mV for fly).
    // Parameters from Izhikevich (2003) Table 1 and (2008) supplementary.

    case CellType::kL23_Pyramidal:
      // Regular spiking. Testa-Silva et al (2014): human L2/3 pyramidal,
      // Vm=-65mV, tau_m=22ms, AP half-width 1.2ms. Main cortical workhorse.
      return {0.02f, 0.20f, -65.0f, 8.0f, 30.0f, 1.0f};

    case CellType::kL4_Stellate:
      // Spiny stellate (regular spiking). Receives thalamocortical input.
      // Slightly faster recovery than L2/3 pyramidal (smaller d).
      // Bruno & Sakmann (2006): short latency thalamic responses.
      return {0.02f, 0.20f, -65.0f, 6.0f, 30.0f, 1.0f};

    case CellType::kL5_Pyramidal:
      // Intrinsic bursting. Large thick-tufted pyramidal cells.
      // Beaulieu-Laroche et al (2018): human L5 neurons, larger dendrites.
      // Hay et al (2011): dendritic Ca2+ spikes in apical tuft.
      // c=-55 (shallower reset) enables burst initiation.
      return {0.02f, 0.20f, -55.0f, 4.0f, 30.0f, 1.0f};

    case CellType::kL6_Pyramidal:
      // Regular spiking corticothalamic cells.
      // Thomson (2010): L6 cells provide feedback to thalamus.
      // Moderate recovery (d=6), slightly slower than L2/3.
      return {0.02f, 0.20f, -65.0f, 6.0f, 30.0f, 1.0f};

    case CellType::kPV_Basket:
      // Fast spiking. PV+ basket cells: primary perisomatic inhibition.
      // Hu et al (2014): sustained >200 Hz firing, narrow spikes.
      // a=0.1 (fast recovery), d=2 (minimal after-spike depression).
      return {0.10f, 0.20f, -65.0f, 2.0f, 30.0f, 1.0f};

    case CellType::kSST_Martinotti:
      // Low-threshold spiking. SST+ Martinotti cells target apical dendrites.
      // Silberberg & Markram (2007): LTS firing from hyperpolarization.
      // b=0.25 (stronger subthreshold dynamics, rebound bursts).
      return {0.02f, 0.25f, -65.0f, 2.0f, 30.0f, 1.0f};

    case CellType::kVIP_Interneuron:
      // Irregular/adapting spiking. VIP+ cells inhibit SST+ cells,
      // creating disinhibitory circuits (Pi et al. 2013 Nature).
      // Moderate a, higher d for adaptation.
      return {0.02f, 0.20f, -65.0f, 6.0f, 30.0f, 1.0f};

    case CellType::kThalamocortical:
      // Tonic mode thalamocortical relay neurons.
      // McCormick & Huguenard (1992): tonic regular spiking during wakefulness.
      // Very small d=0.05 allows sustained tonic firing.
      // In burst mode (from hyperpolarized state), these produce
      // low-threshold Ca2+ spike bursts mediated by T-type channels.
      return {0.02f, 0.25f, -65.0f, 0.05f, 30.0f, 1.0f};

    case CellType::kTRN:
      // Thalamic reticular nucleus: GABAergic, gates thalamocortical relay.
      // Pinault (2004): burst and tonic firing modes.
      // b=0.25 for rebound burst capability (like LTS).
      return {0.02f, 0.25f, -65.0f, 2.0f, 30.0f, 1.0f};

    case CellType::kCholinergic_NBM:
      // Nucleus basalis of Meynert cholinergic projection neurons.
      // Zaborszky et al (2015): slow tonic firing ~2-5 Hz.
      // Provides cortical ACh for attention and arousal.
      return {0.02f, 0.25f, -60.0f, 8.0f, 30.0f, 1.5f};

    default:
      return {0.02f, 0.20f, -60.0f, 6.0f, 20.0f, 2.0f};
  }
}

}  // namespace mechabrain

#endif  // FWMC_CELL_TYPE_DEFS_H_
