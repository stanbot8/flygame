#ifndef FWMC_SPECIES_H_
#define FWMC_SPECIES_H_

// Species-specific biophysical defaults.
//
// Different organisms have fundamentally different neural biophysics:
// membrane time constants, spike peaks, conduction velocities, body
// temperature, synaptic recovery kinetics, etc. This header provides
// a clean way to select species-appropriate defaults for all subsystems
// from a single configuration point.
//
// Usage:
//   auto defaults = SpeciesDefaults::For(Species::kMouse);
//   STPParams stp = defaults.DefaultSTP();
//   float ref_temp = defaults.ref_temperature_c;
//
// Adding a new species:
//   1. Add to Species enum
//   2. Add case to SpeciesDefaults::For()
//   3. Fill in biophysical parameters from literature

#include <cstdint>
#include <string>

namespace mechabrain {

enum class Species : uint8_t {
  kDrosophila = 0,  // Drosophila melanogaster (fruit fly)
  kMouse      = 1,  // Mus musculus
  kRat        = 2,  // Rattus norvegicus
  kHuman      = 3,  // Homo sapiens
  kZebrafish  = 4,  // Danio rerio (larval)
  kGeneric    = 255 // no species-specific tuning
};

inline const char* SpeciesName(Species s) {
  switch (s) {
    case Species::kDrosophila: return "Drosophila melanogaster";
    case Species::kMouse:      return "Mus musculus";
    case Species::kRat:        return "Rattus norvegicus";
    case Species::kHuman:      return "Homo sapiens";
    case Species::kZebrafish:  return "Danio rerio";
    default:                   return "Generic";
  }
}

inline const char* SpeciesCommonName(Species s) {
  switch (s) {
    case Species::kDrosophila: return "fruit fly";
    case Species::kMouse:      return "mouse";
    case Species::kRat:        return "rat";
    case Species::kHuman:      return "human";
    case Species::kZebrafish:  return "zebrafish";
    default:                   return "generic";
  }
}

// Bundles all species-specific biophysical defaults.
// Each field has a literature citation in the comment.
struct SpeciesDefaults {
  Species species = Species::kGeneric;

  // -- Membrane biophysics --
  float resting_potential_mv = -65.0f;  // resting membrane potential
  float spike_peak_mv       = 30.0f;   // action potential peak
  float membrane_tau_ms     = 20.0f;   // typical membrane time constant
  float input_resistance_mohm = 100.0f; // typical input resistance

  // -- Temperature --
  float ref_temperature_c   = 37.0f;   // reference body temperature
  float q10_channels        = 2.5f;    // ion channel Q10
  float q10_synapses        = 2.0f;    // synaptic release Q10
  float q10_membrane        = 1.5f;    // membrane tau Q10

  // -- Axonal conduction --
  float conduction_velocity_m_s = 1.0f; // unmyelinated axon velocity
  float myelinated_velocity_m_s = 5.0f; // myelinated axon velocity (if applicable)
  bool  has_myelin = false;

  // -- Short-term plasticity (Tsodyks-Markram) --
  float stp_U_se  = 0.5f;   // baseline release probability
  float stp_tau_d = 200.0f;  // depression recovery (ms)
  float stp_tau_f = 50.0f;   // facilitation decay (ms)

  // -- STDP --
  float stdp_a_plus  = 0.01f;   // potentiation amplitude
  float stdp_a_minus = 0.012f;  // depression amplitude
  float stdp_tau_plus  = 20.0f; // potentiation window (ms)
  float stdp_tau_minus = 20.0f; // depression window (ms)

  // -- Synaptic time constants --
  float tau_syn_excitatory = 3.0f;  // fast excitatory (ms)
  float tau_syn_inhibitory = 5.0f;  // fast inhibitory (ms)
  float tau_syn_modulatory = 10.0f; // neuromodulatory (ms)

  // -- Background current --
  float background_mean = 5.0f;  // pA
  float background_std  = 2.0f;  // pA

  // -- Optogenetics tissue optics --
  float tissue_scattering_inv_mm = 10.0f; // scattering coefficient at 590nm
  float brain_depth_um = 5000.0f;          // max addressable depth

  // Factory: return species-appropriate defaults.
  static SpeciesDefaults For(Species s) {
    SpeciesDefaults d;
    d.species = s;

    switch (s) {
      case Species::kDrosophila:
        // Wilson & Laurent 2005; Gu & O'Dowd 2006
        d.resting_potential_mv = -60.0f;
        d.spike_peak_mv       = 20.0f;
        d.membrane_tau_ms     = 10.0f;
        d.input_resistance_mohm = 1500.0f; // 1.5 GOhm (small neurons)
        // Soto-Padilla et al. 2018
        d.ref_temperature_c   = 22.0f;
        d.q10_channels        = 2.5f;
        d.q10_synapses        = 2.0f;
        d.q10_membrane        = 1.5f;
        // Tanouye & Wyman 1980
        d.conduction_velocity_m_s = 0.5f;
        d.myelinated_velocity_m_s = 0.5f; // no myelin
        d.has_myelin = false;
        // Hallermann et al. 2010; Kittel et al. 2006
        d.stp_U_se  = 0.5f;
        d.stp_tau_d = 40.0f;
        d.stp_tau_f = 50.0f;
        // Hige et al. 2015 (depression-dominant MB)
        d.stdp_a_plus  = 0.005f;
        d.stdp_a_minus = 0.015f;
        d.stdp_tau_plus  = 15.0f;
        d.stdp_tau_minus = 25.0f;
        // Wilson & Laurent 2005
        d.tau_syn_excitatory = 2.0f;  // fast ACh
        d.tau_syn_inhibitory = 5.0f;  // GABA
        d.tau_syn_modulatory = 10.0f;
        d.background_mean = 8.0f;
        d.background_std  = 3.0f;
        // Prakash et al. 2012
        d.tissue_scattering_inv_mm = 15.0f;
        d.brain_depth_um = 500.0f;
        break;

      case Species::kMouse:
        // Gentet et al. 2010; Crochet et al. 2011
        d.resting_potential_mv = -65.0f;
        d.spike_peak_mv       = 30.0f;
        d.membrane_tau_ms     = 20.0f;
        d.input_resistance_mohm = 150.0f;
        d.ref_temperature_c   = 37.0f;
        d.q10_channels        = 3.0f;  // Hille 2001
        d.q10_synapses        = 2.5f;
        d.q10_membrane        = 1.5f;
        // Swadlow 1985 (unmyelinated cortical)
        d.conduction_velocity_m_s = 0.5f;
        // Salami et al. 2003 (myelinated callosal)
        d.myelinated_velocity_m_s = 3.5f;
        d.has_myelin = true;
        // Markram et al. 1998 (cortical)
        d.stp_U_se  = 0.5f;
        d.stp_tau_d = 200.0f;
        d.stp_tau_f = 50.0f;
        // Markram et al. 1997 (cortical L5)
        d.stdp_a_plus  = 0.01f;
        d.stdp_a_minus = 0.012f;
        d.stdp_tau_plus  = 20.0f;
        d.stdp_tau_minus = 20.0f;
        // Bhatt et al. 2013
        d.tau_syn_excitatory = 3.0f;  // AMPA
        d.tau_syn_inhibitory = 8.0f;  // GABAa
        d.tau_syn_modulatory = 15.0f;
        d.background_mean = 5.0f;
        d.background_std  = 2.0f;
        // Yona et al. 2016 (mouse cortex, 1040nm)
        d.tissue_scattering_inv_mm = 8.0f;
        d.brain_depth_um = 4000.0f;  // ~4mm cortex
        break;

      case Species::kRat:
        // Similar to mouse with adjustments
        d = For(Species::kMouse);
        d.species = Species::kRat;
        d.input_resistance_mohm = 120.0f; // slightly larger neurons
        // Markram et al. 2015 (Blue Brain, rat S1)
        d.stp_U_se  = 0.5f;
        d.stp_tau_d = 200.0f;
        d.stp_tau_f = 50.0f;
        d.brain_depth_um = 5000.0f; // ~5mm cortex
        break;

      case Species::kHuman:
        // Testa-Silva et al. 2014; Beaulieu-Laroche et al. 2018
        d.resting_potential_mv = -65.0f;
        d.spike_peak_mv       = 30.0f;
        d.membrane_tau_ms     = 25.0f;  // larger neurons
        d.input_resistance_mohm = 100.0f;
        d.ref_temperature_c   = 37.0f;
        d.q10_channels        = 3.0f;
        d.q10_synapses        = 2.5f;
        d.q10_membrane        = 1.5f;
        // Aboitiz et al. 1992 (corpus callosum)
        d.conduction_velocity_m_s = 1.0f;
        d.myelinated_velocity_m_s = 10.0f;
        d.has_myelin = true;
        // Similar to rodent cortical
        d.stp_U_se  = 0.5f;
        d.stp_tau_d = 200.0f;
        d.stp_tau_f = 50.0f;
        d.stdp_a_plus  = 0.01f;
        d.stdp_a_minus = 0.012f;
        d.stdp_tau_plus  = 20.0f;
        d.stdp_tau_minus = 20.0f;
        d.tau_syn_excitatory = 3.0f;
        d.tau_syn_inhibitory = 8.0f;
        d.tau_syn_modulatory = 15.0f;
        d.background_mean = 5.0f;
        d.background_std  = 2.0f;
        d.tissue_scattering_inv_mm = 8.0f;
        d.brain_depth_um = 10000.0f; // ~10mm cortex
        break;

      case Species::kZebrafish:
        // Ahrens et al. 2013 (larval zebrafish)
        d.resting_potential_mv = -65.0f;
        d.spike_peak_mv       = 25.0f;
        d.membrane_tau_ms     = 15.0f;
        d.input_resistance_mohm = 500.0f;
        d.ref_temperature_c   = 28.5f;  // standard rearing temperature
        d.q10_channels        = 2.5f;
        d.q10_synapses        = 2.0f;
        d.q10_membrane        = 1.5f;
        d.conduction_velocity_m_s = 0.3f;
        d.myelinated_velocity_m_s = 0.3f;
        d.has_myelin = false;
        d.stp_U_se  = 0.4f;
        d.stp_tau_d = 60.0f;
        d.stp_tau_f = 80.0f;
        d.stdp_a_plus  = 0.008f;
        d.stdp_a_minus = 0.01f;
        d.stdp_tau_plus  = 18.0f;
        d.stdp_tau_minus = 22.0f;
        d.tau_syn_excitatory = 3.0f;
        d.tau_syn_inhibitory = 6.0f;
        d.tau_syn_modulatory = 12.0f;
        d.background_mean = 6.0f;
        d.background_std  = 2.5f;
        // Nearly transparent larval brain
        d.tissue_scattering_inv_mm = 3.0f;
        d.brain_depth_um = 300.0f;
        break;

      default:
        break;
    }
    return d;
  }
};

// Parse species name from string (for .brain spec files).
inline Species ParseSpecies(const std::string& name) {
  if (name == "drosophila" || name == "Drosophila" ||
      name == "Drosophila melanogaster" || name == "fly")
    return Species::kDrosophila;
  if (name == "mouse" || name == "Mouse" || name == "Mus musculus")
    return Species::kMouse;
  if (name == "rat" || name == "Rat" || name == "Rattus norvegicus")
    return Species::kRat;
  if (name == "human" || name == "Human" || name == "Homo sapiens")
    return Species::kHuman;
  if (name == "zebrafish" || name == "Zebrafish" || name == "Danio rerio")
    return Species::kZebrafish;
  return Species::kGeneric;
}

}  // namespace mechabrain

#endif  // FWMC_SPECIES_H_
