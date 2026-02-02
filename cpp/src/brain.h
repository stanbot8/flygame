#pragma once
// Embedded brain module for flygame.
//
// Wraps the fwmc brain-model spiking neural network as an optional
// "plugin" that runs in-process. The brain receives WASD-driven intent
// and proprioceptive feedback, and produces motor commands via descending
// neuron activity.
//
// Three controller modes in flygame:
//   1. Standalone CPG (no brain, WASD -> CPG directly)
//   2. Embedded brain (this module: WASD -> brain neurons -> motor output -> CPG)
//   3. TCP bridge   (brain runs in a separate process)
//
// The brain module is header-only (all fwmc brain-model headers are).
// Enable with -DNMFLY_BRAIN=ON in CMake, requires FWMC_BRAIN_DIR.

// WasdIntent is used by main.cc even without the brain module,
// so it lives outside the NMFLY_BRAIN guard.
namespace nmfly {

// WASD intent: smoothed drive signals from keyboard input.
struct WasdIntent {
    float forward  = 0.0f;  // [0, 1] W key
    float turn     = 0.0f;  // [-1, 1] A/D keys
    bool  running  = false;  // Space key
    bool  flying   = false;  // Shift key
};

}  // namespace nmfly

#ifdef NMFLY_BRAIN

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/neuron_array.h"
#include "core/izhikevich.h"
#include "core/synapse_table.h"
#include "core/stdp.h"
#include "core/cell_types.h"
#include "core/motor_output.h"
#include "core/cpg.h"
#include "core/gap_junctions.h"
#include "core/intrinsic_homeostasis.h"
#include "core/spike_frequency_adaptation.h"
#include "core/neuromodulator_effects.h"
#include "core/inhibitory_plasticity.h"
#include "core/sim_features.h"
#include "core/parametric_gen.h"
#include "core/brain_spec_loader.h"
#include "core/structural_plasticity.h"
#include "core/temperature.h"

#include "types.h"  // mjgame types

namespace nmfly {

using mjgame::MotorCommand;
using mjgame::SensoryReading;
using mjgame::BodyState;

// The embedded brain. Owns all neural state and runs the spiking simulation.
class Brain {
public:
    Brain() = default;

    // Load brain spec from .brain file and generate the connectome.
    bool Init(const std::string& spec_path) {
        using namespace mechabrain;

        auto spec_result = BrainSpecLoader::Load(spec_path);
        if (!spec_result) {
            fprintf(stderr, "[brain] Failed to load spec: %s\n",
                    spec_path.c_str());
            return false;
        }
        spec_ = *spec_result;

        // Generate connectome (also assigns cell types)
        gen_.Generate(spec_, neurons_, synapses_, cell_types_);
        printf("[brain] Generated %zu neurons, %zu synapses\n",
               neurons_.n, synapses_.post.size());

        // Gap junctions in antennal lobes
        // Find AL region indices from spec
        int al_l = FindRegionIndex("antennal_lobe_L");
        int al_r = FindRegionIndex("antennal_lobe_R");
        if (al_l < 0) al_l = FindRegionIndex("antennal_lobe");
        if (al_l >= 0)
            gap_junctions_.BuildFromRegion(neurons_, al_l, 0.02f, 0.5f, 42);
        if (al_r >= 0)
            gap_junctions_.BuildFromRegion(neurons_, al_r, 0.02f, 0.5f, 43);

        // Homeostasis
        homeostasis_.Init(neurons_.n, 2.0f, 1.0f);
        homeostasis_.SetTargetsFromTypes(neurons_);
        homeostasis_.learning_rate = 0.005f;
        homeostasis_.max_bias = 3.0f;

        // SFA
        sfa_.Init(neurons_.n);

        // Temperature (Drosophila in-vivo 25C, measured at 22C)
        temperature_.enabled = true;
        temperature_.reference_temp_c = 22.0f;
        temperature_.current_temp_c = 25.0f;

        // Conduction delays
        synapses_.InitDistanceDelay(
            neurons_,
            mechabrain::SynapseTable::kVelocityDrosophilaUnmyel * 1000.0f,
            1.0f);

        // STP
        if (features_.short_term_plasticity) {
            STPParams stp_params;
            synapses_.InitSTP(stp_params);
        }

        // Synaptic scaling
        synaptic_scaling_.Init(neurons_.n);

        // Motor output: find SEZ/premotor region for descending neurons.
        // The .brain spec may use split (sez, lateral_horn_L/R) or combined
        // (protocerebrum, lateral_horn) region names. Fall back gracefully.
        int sez_idx = FindRegionIndex("sez");
        if (sez_idx < 0) sez_idx = FindRegionIndex("subesophageal_zone");
        if (sez_idx < 0) sez_idx = FindRegionIndex("protocerebrum");
        int mbon_idx = FindRegionIndex("mushroom_body");
        if (mbon_idx < 0) mbon_idx = FindRegionIndex("mb_calyx");

        // Compute SEZ/premotor midline from actual neuron positions
        sez_midline_ = 250.0f;
        if (sez_idx >= 0) {
            float sez_min = 1e9f, sez_max = -1e9f;
            for (size_t i = 0; i < neurons_.n; ++i) {
                if (neurons_.region[i] == sez_idx) {
                    sez_min = std::min(sez_min, neurons_.x[i]);
                    sez_max = std::max(sez_max, neurons_.x[i]);
                }
            }
            if (sez_max > sez_min)
                sez_midline_ = (sez_min + sez_max) * 0.5f;
        }
        sez_region_ = sez_idx >= 0 ? sez_idx : 6;
        lh_l_region_ = FindRegionIndex("lateral_horn_L");
        lh_r_region_ = FindRegionIndex("lateral_horn_R");
        // If no split L/R, use the combined lateral_horn for both.
        // SetIntent uses x-position vs midline to distinguish L/R halves.
        if (lh_l_region_ < 0 || lh_r_region_ < 0) {
            int lh = FindRegionIndex("lateral_horn");
            if (lh >= 0) { lh_l_region_ = lh; lh_r_region_ = lh; }
            else { lh_l_region_ = 2; lh_r_region_ = 2; }
        }

        // VNC for CPG oscillator and motor suppression
        int vnc_idx = FindRegionIndex("vnc");
        if (vnc_idx < 0) vnc_idx = FindRegionIndex("ventral_nerve_cord");
        vnc_region_ = vnc_idx;

        // Build VNC sensory neuron list (first 30% of VNC neurons).
        // These receive proprioceptive feedback from the body.
        if (vnc_region_ >= 0) {
            std::vector<uint32_t> all_vnc;
            for (size_t i = 0; i < neurons_.n; ++i)
                if (neurons_.region[i] == vnc_region_)
                    all_vnc.push_back(static_cast<uint32_t>(i));
            size_t n_sensory = all_vnc.size() * 30 / 100;
            vnc_sensory_.assign(all_vnc.begin(), all_vnc.begin() + n_sensory);
        }

        // Motor output monitors VNC (descending motor neurons) for good
        // gait dynamics. WASD gating happens in the main loop to prevent
        // spontaneous walking from background neural activity.
        int motor_region = vnc_idx >= 0 ? vnc_idx : sez_region_;
        motor_.InitFromRegions(neurons_, motor_region,
                                mbon_idx >= 0 ? mbon_idx : 1,
                                sez_midline_);
        // Lower velocity gain for large VNC populations.
        // Default 5.0 was tuned for smaller circuits; with 15k VNC neurons
        // the background firing saturates forward_velocity at max. Lower
        // gain produces moderate walking speed (~10-15 mm/s) and smoother
        // gait with less body bounce.
        motor_.velocity_gain = 1.5f;
        cpg_.Init(neurons_, vnc_idx >= 0 ? vnc_idx : 9, sez_midline_);

        // Background drive per region (from spec)
        int n_regions = static_cast<int>(spec_.regions.size());
        sdf_background_.assign(std::max(n_regions, 13), 1.0f);
        for (size_t r = 0; r < spec_.regions.size(); ++r) {
            float bg = spec_.regions[r].background_mean;
            if (bg > 0.0f)
                sdf_background_[r] = bg;
        }

        spike_time_ = 0.0f;
        initialized_ = true;

        printf("[brain] Ready: %zu neurons, %zu synapses, %zu gap junctions\n",
               neurons_.n, synapses_.post.size(), gap_junctions_.Size());
        printf("[brain] SEZ=%d (midline=%.1f), VNC=%d, MBON=%d\n",
               sez_region_, sez_midline_,
               vnc_idx >= 0 ? vnc_idx : 9,
               mbon_idx >= 0 ? mbon_idx : 5);
        return true;
    }

    // Inject proprioceptive feedback from body into VNC sensory neurons.
    // The first 30% of VNC neurons are sensory afferents (Takemura et al. 2024).
    // Call before Step() to close the sensorimotor loop.
    void SetProprio(const SensoryReading* readings, int n_readings) {
        if (!initialized_ || vnc_region_ < 0 || n_readings <= 0) return;
        if (vnc_sensory_.empty()) return;

        // Distribute readings across VNC sensory neurons.
        // Each sensory neuron gets current proportional to its channel's
        // activation, scaled to a biologically plausible range (0-8 pA).
        constexpr float kProprioGain = 8.0f;

        // Reset sensory neurons to baseline first.
        int bg_idx = std::clamp(vnc_region_, 0, (int)sdf_background_.size() - 1);
        float bg = sdf_background_[bg_idx];
        for (uint32_t idx : vnc_sensory_)
            neurons_.i_ext[idx] = bg;

        // Map each proprioceptive channel to a slice of sensory neurons.
        size_t neurons_per_channel = vnc_sensory_.size() / static_cast<size_t>(n_readings);
        if (neurons_per_channel < 1) neurons_per_channel = 1;

        for (int ch = 0; ch < n_readings; ++ch) {
            float activation = readings[ch].activation;
            float current = activation * kProprioGain;
            size_t start = static_cast<size_t>(ch) * neurons_per_channel;
            size_t end = std::min(start + neurons_per_channel, vnc_sensory_.size());
            for (size_t j = start; j < end; ++j)
                neurons_.i_ext[vnc_sensory_[j]] += current;
        }
    }

    // Inject WASD intent into brain neurons as excitatory currents.
    // Call before Step() each frame.
    void SetIntent(const WasdIntent& intent) {
        if (!initialized_) return;

        float speed_mult = intent.flying ? 3.0f : (intent.running ? 2.0f : 1.0f);
        float base_current = intent.flying ? 18.0f : (intent.running ? 15.0f : 12.0f);

        float sez_fwd = intent.forward * base_current * speed_mult;
        float turn_strength = std::abs(intent.turn) * base_current;
        float sez_L = sez_fwd + (intent.turn > 0 ? turn_strength : -turn_strength * 0.8f);
        float sez_R = sez_fwd + (intent.turn < 0 ? turn_strength : -turn_strength * 0.8f);
        float lh_L = intent.turn > 0 ? intent.turn * base_current : 0.0f;
        float lh_R = intent.turn < 0 ? -intent.turn * base_current : 0.0f;

        sez_L = std::max(0.0f, sez_L);
        sez_R = std::max(0.0f, sez_R);

        constexpr float kTonicInhib = -3.0f;

        for (size_t i = 0; i < neurons_.n; ++i) {
            int r = neurons_.region[i];
            if (r == sez_region_) {
                float drive = (neurons_.x[i] < sez_midline_) ? sez_L : sez_R;
                neurons_.i_ext[i] = kTonicInhib + drive;
            }
            else if (r == lh_l_region_ || r == lh_r_region_) {
                // If L/R are the same region, split by position vs midline.
                bool is_left = (lh_l_region_ == lh_r_region_)
                    ? (neurons_.x[i] < sez_midline_) : (r == lh_l_region_);
                neurons_.i_ext[i] = kTonicInhib + (is_left ? lh_L : lh_R);
            }
            else if (intent.flying) {
                neurons_.i_ext[i] = 6.0f;
            }
            else {
                int bg_idx = std::clamp(r, 0, (int)sdf_background_.size() - 1);
                neurons_.i_ext[i] = sdf_background_[bg_idx];
            }
        }

        // CPG drive from forward intent
        cpg_.Step(neurons_, 1.0f, intent.forward);
    }

    // Step the brain simulation by dt_ms milliseconds.
    void Step(float dt_ms) {
        if (!initialized_) return;
        using namespace mechabrain;

        float syn_tau = temperature_.ScaledTauSyn(3.0f);

        if (features_.conduction_delays)
            synapses_.DeliverDelayed(neurons_.i_syn.data());

        neurons_.DecaySynapticInput(dt_ms, syn_tau);

        if (features_.gap_junctions)
            gap_junctions_.PropagateGapCurrents(neurons_);

        if (features_.short_term_plasticity && synapses_.HasSTP())
            synapses_.RecoverSTP(dt_ms);

        synapses_.PropagateSpikes(neurons_.spiked.data(),
                                  neurons_.i_syn.data(), 1.0f);

        IzhikevichStepHeterogeneousFast(neurons_, dt_ms, spike_time_, cell_types_);

        if (features_.sfa)
            sfa_.Update(neurons_, dt_ms);

        if (features_.neuromodulation)
            NeuromodulatorUpdate(neurons_, synapses_, dt_ms);

        if (features_.neuromodulator_effects)
            neuromod_effects_.Apply(neurons_);

        if (features_.stdp)
            STDPUpdate(synapses_, neurons_, spike_time_, stdp_params_);

        if (features_.homeostasis) {
            homeostasis_.RecordSpikes(neurons_);
            homeostasis_.MaybeApply(neurons_);
        }

        if (features_.conduction_delays)
            synapses_.AdvanceDelayRing();

        motor_.Update(neurons_, dt_ms);
        spike_time_ += dt_ms;
    }

    // Get the current motor command from descending neuron activity.
    MotorCommand GetMotorCommand() const {
        if (!initialized_) return {};
        auto cmd = motor_.Command();
        nmfly::MotorCommand out;
        out.forward_velocity = cmd.forward_velocity;
        out.angular_velocity = cmd.angular_velocity;
        out.approach_drive   = cmd.approach_drive;
        out.freeze           = cmd.freeze;
        return out;
    }

    bool IsInitialized() const { return initialized_; }
    size_t NeuronCount() const { return neurons_.n; }
    size_t SynapseCount() const { return synapses_.post.size(); }
    mechabrain::SimFeatures& Features() { return features_; }
    const mechabrain::NeuronArray& Neurons() const { return neurons_; }

    // Current spike count (for HUD display).
    int CountSpikes() const {
        if (!initialized_) return 0;
        return neurons_.CountSpikes();
    }

    // Spike rate as fraction of neurons firing.
    float SpikeRate() const {
        if (!initialized_ || neurons_.n == 0) return 0.0f;
        return static_cast<float>(neurons_.CountSpikes()) /
               static_cast<float>(neurons_.n);
    }

    // Motor output rates (for display).
    float MotorRateLeft() const { return motor_.rate_left; }
    float MotorRateRight() const { return motor_.rate_right; }
    float SimTimeMs() const { return spike_time_; }

    // Region count and names (for display).
    int RegionCount() const {
        return static_cast<int>(gen_.region_ranges.size());
    }
    const char* RegionName(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(gen_.region_ranges.size()))
            return "?";
        return gen_.region_ranges[idx].name.c_str();
    }

private:
    int FindRegionIndex(const std::string& name) const {
        for (size_t i = 0; i < gen_.region_ranges.size(); ++i) {
            if (gen_.region_ranges[i].name == name)
                return static_cast<int>(i);
        }
        return -1;
    }

    bool initialized_ = false;
    float spike_time_ = 0.0f;
    float sez_midline_ = 250.0f;
    int sez_region_ = 12;
    int lh_l_region_ = 10;
    int lh_r_region_ = 11;
    int vnc_region_ = -1;
    std::vector<uint32_t> vnc_sensory_;  // first 30% of VNC = sensory afferents
    std::vector<float> sdf_background_;

    mechabrain::BrainSpec spec_;
    mechabrain::ParametricGenerator gen_;
    mechabrain::NeuronArray neurons_;
    mechabrain::SynapseTable synapses_;
    mechabrain::CellTypeManager cell_types_;
    mechabrain::GapJunctionTable gap_junctions_;
    mechabrain::IntrinsicHomeostasis homeostasis_;
    mechabrain::SpikeFrequencyAdaptation sfa_;
    mechabrain::TemperatureModel temperature_;
    mechabrain::NeuromodulatorEffects neuromod_effects_;
    mechabrain::SynapticScaling synaptic_scaling_;
    mechabrain::STDPParams stdp_params_;
    mechabrain::MotorOutput motor_;
    mechabrain::CPGOscillator cpg_;
    mechabrain::SimFeatures features_;
};

}  // namespace nmfly

#endif  // NMFLY_BRAIN
