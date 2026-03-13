#ifndef FWMC_EXPERIMENT_PROTOCOL_H_
#define FWMC_EXPERIMENT_PROTOCOL_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "core/experiment_config.h"
#include "core/log.h"

namespace mechabrain {

// A single phase in a multi-phase experiment
struct ExperimentPhase {
    std::string name;
    float duration_ms;
    int bridge_mode;  // 0=open-loop, 1=shadow, 2=closed-loop
    bool enable_stdp;
    float weight_scale;
    std::vector<StimulusEvent> stimuli;

    // Optional: modify params at phase start
    std::optional<float> dt_ms;
    std::optional<float> monitor_threshold;
    std::optional<float> bridge_threshold;

    // Transition condition (checked each step)
    // If set, phase ends early when condition returns true
    // Can be: correlation > threshold, time elapsed, spike rate condition
    enum class TransitionType { kTimeOnly, kCorrelationAbove, kSpikeRateBelow, kCustom };
    TransitionType transition = TransitionType::kTimeOnly;
    float transition_threshold = 0.0f;
};

// Pre-defined experiment templates
struct ExperimentProtocol {
    std::string name;
    std::string description;
    std::vector<ExperimentPhase> phases;

    // Reproducibility
    uint32_t random_seed = 42;
    std::string config_hash;  // SHA256 of serialized config

    // Results aggregation
    struct PhaseResult {
        std::string phase_name;
        float actual_duration_ms;
        float final_correlation;
        float mean_spike_rate;
        int total_resyncs;
        float replacement_fraction;
    };
    std::vector<PhaseResult> results;

    // ---------------------------------------------------------------
    // Factory methods for common protocols
    // ---------------------------------------------------------------

    // Open-loop baseline: run simulation with no bridge feedback
    static ExperimentProtocol OpenLoopBaseline(float duration_ms) {
        ExperimentProtocol p;
        p.name = "Open-Loop Baseline";
        p.description = "Run the digital twin in open-loop mode to establish "
                         "baseline activity statistics.";

        ExperimentPhase ph;
        ph.name = "Baseline";
        ph.duration_ms = duration_ms;
        ph.bridge_mode = 0;
        ph.enable_stdp = false;
        ph.weight_scale = 1.0f;
        p.phases.push_back(std::move(ph));

        return p;
    }

    // Shadow validation: observe biological data without writing back.
    // Ends early if spike correlation exceeds the given threshold.
    static ExperimentProtocol ShadowValidation(float shadow_ms,
                                                float threshold) {
        ExperimentProtocol p;
        p.name = "Shadow Validation";
        p.description = "Shadow-mode observation to validate digital twin "
                         "fidelity against live biological data.";

        // Phase 1: brief open-loop warmup
        {
            ExperimentPhase ph;
            ph.name = "Warmup";
            ph.duration_ms = 1000.0f;
            ph.bridge_mode = 0;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            p.phases.push_back(std::move(ph));
        }

        // Phase 2: shadow monitoring with early-exit on high correlation
        {
            ExperimentPhase ph;
            ph.name = "Shadow Monitor";
            ph.duration_ms = shadow_ms;
            ph.bridge_mode = 1;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            ph.transition = ExperimentPhase::TransitionType::kCorrelationAbove;
            ph.transition_threshold = threshold;
            p.phases.push_back(std::move(ph));
        }

        return p;
    }

    // Full twinning pipeline: open-loop -> shadow -> closed-loop replacement
    static ExperimentProtocol FullTwinning(float observe_ms,
                                            float shadow_ms,
                                            float closed_ms) {
        ExperimentProtocol p;
        p.name = "Full Twinning Pipeline";
        p.description = "Three-phase protocol: open-loop baseline observation, "
                         "shadow-mode validation, then closed-loop neuron "
                         "replacement.";

        // Phase 1: open-loop baseline
        {
            ExperimentPhase ph;
            ph.name = "Open-Loop Baseline";
            ph.duration_ms = observe_ms;
            ph.bridge_mode = 0;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            p.phases.push_back(std::move(ph));
        }

        // Phase 2: shadow monitoring; end early if correlation > 0.85
        {
            ExperimentPhase ph;
            ph.name = "Shadow Monitoring";
            ph.duration_ms = shadow_ms;
            ph.bridge_mode = 1;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            ph.transition = ExperimentPhase::TransitionType::kCorrelationAbove;
            ph.transition_threshold = 0.85f;
            p.phases.push_back(std::move(ph));
        }

        // Phase 3: closed-loop replacement with STDP
        {
            ExperimentPhase ph;
            ph.name = "Closed-Loop Replacement";
            ph.duration_ms = closed_ms;
            ph.bridge_mode = 2;
            ph.enable_stdp = true;
            ph.weight_scale = 1.0f;
            ph.monitor_threshold = 0.6f;
            ph.bridge_threshold = 0.8f;
            p.phases.push_back(std::move(ph));
        }

        return p;
    }

    // Odor learning: N trials of (CS presentation + ITI).
    // Odd trials: CS+ (odor A) paired with reward (DAN activation).
    // Even trials: CS- (odor B), no reward.
    // STDP is dopamine-gated so only CS+ trials cause learning in KC->MBON
    // synapses.
    static ExperimentProtocol OdorLearning(int n_trials,
                                            float trial_ms,
                                            float iti_ms) {
        ExperimentProtocol p;
        p.name = "Odor Associative Learning";
        p.description = "Classical conditioning protocol with CS+/CS- "
                         "interleaved trials. STDP is dopamine-gated; only "
                         "CS+ trials (paired with DAN reward) produce lasting "
                         "weight changes in KC->MBON pathways.";

        // Example target neuron ranges. Real experiments would load these
        // from the connectome. We use placeholder indices that map to the
        // Region/CellType scheme used elsewhere.
        std::vector<uint32_t> orn_odor_a;   // ORNs responding to odor A
        std::vector<uint32_t> orn_odor_b;   // ORNs responding to odor B
        std::vector<uint32_t> dan_reward;    // DAN-PAM reward neurons

        for (uint32_t i = 0; i < 50; ++i) orn_odor_a.push_back(i);
        for (uint32_t i = 50; i < 100; ++i) orn_odor_b.push_back(i);
        for (uint32_t i = 200; i < 220; ++i) dan_reward.push_back(i);

        float t = 0.0f;
        for (int trial = 0; trial < n_trials; ++trial) {
            bool is_cs_plus = (trial % 2 == 0);

            ExperimentPhase ph;
            ph.name = is_cs_plus
                ? "CS+ trial " + std::to_string(trial / 2 + 1)
                : "CS- trial " + std::to_string(trial / 2 + 1);
            ph.duration_ms = trial_ms;
            ph.bridge_mode = 0;  // open-loop learning
            ph.enable_stdp = true;
            ph.weight_scale = 1.0f;

            // Odor stimulus for the first 80% of trial
            StimulusEvent odor;
            odor.start_ms = 0.0f;
            odor.end_ms = trial_ms * 0.8f;
            odor.intensity = 0.8f;
            odor.label = is_cs_plus ? "odor_A" : "odor_B";
            odor.target_neurons = is_cs_plus ? orn_odor_a : orn_odor_b;
            ph.stimuli.push_back(odor);

            // Reward (DAN activation) only for CS+ trials, delayed 200ms
            // after odor onset (mimics US timing in olfactory conditioning)
            if (is_cs_plus) {
                StimulusEvent reward;
                reward.start_ms = 200.0f;
                reward.end_ms = trial_ms * 0.6f;
                reward.intensity = 1.0f;
                reward.label = "reward_DAN";
                reward.target_neurons = dan_reward;
                ph.stimuli.push_back(reward);
            }

            p.phases.push_back(std::move(ph));

            // Inter-trial interval (no stimulus, STDP off)
            if (trial < n_trials - 1) {
                ExperimentPhase iti;
                iti.name = "ITI " + std::to_string(trial + 1);
                iti.duration_ms = iti_ms;
                iti.bridge_mode = 0;
                iti.enable_stdp = false;
                iti.weight_scale = 1.0f;
                p.phases.push_back(std::move(iti));
            }

            t += trial_ms + iti_ms;
        }

        return p;
    }

    // Ablation study: baseline -> silence neurons -> observe -> restore ->
    // recovery. Compares dynamics before and after ablation to quantify the
    // functional role of targeted neurons.
    static ExperimentProtocol AblationStudy(
            const std::vector<uint32_t>& ablate_neurons,
            float duration_ms) {
        ExperimentProtocol p;
        p.name = "Ablation Study";
        p.description = "Run baseline, silence specified neurons (simulated "
                         "ablation), observe altered dynamics, then restore "
                         "and measure recovery.";

        float phase_dur = duration_ms / 4.0f;

        // Phase 1: pre-ablation baseline
        {
            ExperimentPhase ph;
            ph.name = "Pre-Ablation Baseline";
            ph.duration_ms = phase_dur;
            ph.bridge_mode = 0;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            p.phases.push_back(std::move(ph));
        }

        // Phase 2: ablation; inject zero-current stimulus to silence
        // target neurons (override with negative current to clamp below
        // threshold)
        {
            ExperimentPhase ph;
            ph.name = "Ablation Active";
            ph.duration_ms = phase_dur;
            ph.bridge_mode = 0;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;

            StimulusEvent silence;
            silence.start_ms = 0.0f;
            silence.end_ms = phase_dur;
            silence.intensity = -1.0f;  // strong hyperpolarizing current
            silence.label = "ablation_silence";
            silence.target_neurons = ablate_neurons;
            ph.stimuli.push_back(silence);

            p.phases.push_back(std::move(ph));
        }

        // Phase 3: ablation observation (continued silencing); monitor
        // downstream effects with shadow tracking
        {
            ExperimentPhase ph;
            ph.name = "Ablation Observation";
            ph.duration_ms = phase_dur;
            ph.bridge_mode = 1;  // shadow to compare
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;

            StimulusEvent silence;
            silence.start_ms = 0.0f;
            silence.end_ms = phase_dur;
            silence.intensity = -1.0f;
            silence.label = "ablation_silence";
            silence.target_neurons = ablate_neurons;
            ph.stimuli.push_back(silence);

            p.phases.push_back(std::move(ph));
        }

        // Phase 4: recovery; release silencing, observe restoration
        {
            ExperimentPhase ph;
            ph.name = "Recovery";
            ph.duration_ms = phase_dur;
            ph.bridge_mode = 0;
            ph.enable_stdp = false;
            ph.weight_scale = 1.0f;
            p.phases.push_back(std::move(ph));
        }

        return p;
    }

    // ---------------------------------------------------------------
    // INI-like serialization
    // ---------------------------------------------------------------

    // Save protocol to .protocol file
    void SaveToFile(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) {
            Log(LogLevel::kError, "Failed to open %s for writing", path.c_str());
            return;
        }

        fprintf(f, "[protocol]\n");
        fprintf(f, "name = %s\n", name.c_str());
        fprintf(f, "description = %s\n", description.c_str());
        fprintf(f, "seed = %u\n", random_seed);
        if (!config_hash.empty())
            fprintf(f, "config_hash = %s\n", config_hash.c_str());
        fprintf(f, "\n");

        for (size_t i = 0; i < phases.size(); ++i) {
            const auto& ph = phases[i];
            // Generate a section key from the phase name (lowercase, underscores)
            std::string key;
            for (char c : ph.name) {
                if (c == ' ') key += '_';
                else key += static_cast<char>(tolower(c));
            }
            fprintf(f, "[phase:%s]\n", key.c_str());
            fprintf(f, "name = %s\n", ph.name.c_str());
            fprintf(f, "duration_ms = %g\n", ph.duration_ms);
            fprintf(f, "bridge_mode = %d\n", ph.bridge_mode);
            fprintf(f, "enable_stdp = %s\n", ph.enable_stdp ? "true" : "false");
            fprintf(f, "weight_scale = %g\n", ph.weight_scale);

            if (ph.dt_ms.has_value())
                fprintf(f, "dt_ms = %g\n", ph.dt_ms.value());
            if (ph.monitor_threshold.has_value())
                fprintf(f, "monitor_threshold = %g\n", ph.monitor_threshold.value());
            if (ph.bridge_threshold.has_value())
                fprintf(f, "bridge_threshold = %g\n", ph.bridge_threshold.value());

            switch (ph.transition) {
                case ExperimentPhase::TransitionType::kTimeOnly:
                    break;  // default, no line needed
                case ExperimentPhase::TransitionType::kCorrelationAbove:
                    fprintf(f, "transition = correlation_above %g\n",
                            ph.transition_threshold);
                    break;
                case ExperimentPhase::TransitionType::kSpikeRateBelow:
                    fprintf(f, "transition = spike_rate_below %g\n",
                            ph.transition_threshold);
                    break;
                case ExperimentPhase::TransitionType::kCustom:
                    fprintf(f, "transition = custom %g\n",
                            ph.transition_threshold);
                    break;
            }

            // Write stimuli
            for (const auto& stim : ph.stimuli) {
                fprintf(f, "stimulus = %s %g %g %g",
                        stim.label.c_str(), stim.start_ms, stim.end_ms,
                        stim.intensity);
                for (size_t j = 0; j < stim.target_neurons.size(); ++j) {
                    fprintf(f, "%s%u", (j == 0) ? " " : ",",
                            stim.target_neurons[j]);
                }
                fprintf(f, "\n");
            }

            fprintf(f, "\n");
        }

        fclose(f);
        Log(LogLevel::kInfo, "Protocol saved to %s (%zu phases)",
            path.c_str(), phases.size());
    }

    // Load protocol from .protocol file (INI-like format)
    static ExperimentProtocol LoadFromFile(const std::string& path) {
        ExperimentProtocol p;

        FILE* f = fopen(path.c_str(), "r");
        if (!f) {
            Log(LogLevel::kError, "Failed to open %s for reading", path.c_str());
            return p;
        }

        char line_buf[1024];
        ExperimentPhase* current_phase = nullptr;

        while (fgets(line_buf, sizeof(line_buf), f)) {
            std::string line(line_buf);
            // Strip trailing whitespace / newline
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'
                                     || line.back() == ' '))
                line.pop_back();

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            // Section header
            if (line[0] == '[') {
                auto close = line.find(']');
                if (close == std::string::npos) continue;
                std::string section = line.substr(1, close - 1);

                if (section == "protocol") {
                    current_phase = nullptr;
                } else if (section.rfind("phase:", 0) == 0) {
                    p.phases.push_back(ExperimentPhase{});
                    current_phase = &p.phases.back();
                    // defaults
                    current_phase->duration_ms = 1000.0f;
                    current_phase->bridge_mode = 0;
                    current_phase->enable_stdp = false;
                    current_phase->weight_scale = 1.0f;
                }
                continue;
            }

            // Key = value
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            // Trim
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());

            if (!current_phase) {
                // Protocol-level keys
                if (key == "name") p.name = val;
                else if (key == "description") p.description = val;
                else if (key == "seed") p.random_seed = static_cast<uint32_t>(std::stoul(val));
                else if (key == "config_hash") p.config_hash = val;
            } else {
                // Phase-level keys
                if (key == "name") current_phase->name = val;
                else if (key == "duration_ms") current_phase->duration_ms = std::stof(val);
                else if (key == "bridge_mode") current_phase->bridge_mode = std::stoi(val);
                else if (key == "enable_stdp") current_phase->enable_stdp = (val == "true");
                else if (key == "weight_scale") current_phase->weight_scale = std::stof(val);
                else if (key == "dt_ms") current_phase->dt_ms = std::stof(val);
                else if (key == "monitor_threshold") current_phase->monitor_threshold = std::stof(val);
                else if (key == "bridge_threshold") current_phase->bridge_threshold = std::stof(val);
                else if (key == "transition") {
                    ParseTransition(val, *current_phase);
                }
                else if (key == "stimulus") {
                    auto stim = ParseStimulus(val);
                    if (!stim.label.empty())
                        current_phase->stimuli.push_back(std::move(stim));
                }
            }
        }

        fclose(f);
        Log(LogLevel::kInfo, "Protocol loaded from %s: \"%s\" (%zu phases)",
            path.c_str(), p.name.c_str(), p.phases.size());
        return p;
    }

    // ---------------------------------------------------------------
    // Helpers: convert to a sequence of ExperimentConfig objects
    // ---------------------------------------------------------------

    // Build an ExperimentConfig for the given phase, inheriting base settings.
    ExperimentConfig ConfigForPhase(size_t phase_idx,
                                     const ExperimentConfig& base) const {
        if (phase_idx >= phases.size()) return base;
        const auto& ph = phases[phase_idx];

        ExperimentConfig cfg = base;
        cfg.name = name + " / " + ph.name;
        cfg.duration_ms = ph.duration_ms;
        cfg.bridge_mode = ph.bridge_mode;
        cfg.enable_stdp = ph.enable_stdp;
        cfg.weight_scale = ph.weight_scale;
        cfg.stimulus_protocol = ph.stimuli;

        if (ph.dt_ms.has_value()) cfg.dt_ms = ph.dt_ms.value();
        if (ph.monitor_threshold.has_value())
            cfg.monitor_threshold = ph.monitor_threshold.value();
        if (ph.bridge_threshold.has_value())
            cfg.bridge_threshold = ph.bridge_threshold.value();

        return cfg;
    }

    // Total duration across all phases (nominal, ignoring early transitions)
    float TotalDuration() const {
        float total = 0.0f;
        for (const auto& ph : phases) total += ph.duration_ms;
        return total;
    }

private:
    static void ParseTransition(const std::string& val,
                                 ExperimentPhase& ph) {
        std::istringstream ss(val);
        std::string type_str;
        ss >> type_str;
        float thresh = 0.0f;
        ss >> thresh;

        if (type_str == "correlation_above") {
            ph.transition = ExperimentPhase::TransitionType::kCorrelationAbove;
            ph.transition_threshold = thresh;
        } else if (type_str == "spike_rate_below") {
            ph.transition = ExperimentPhase::TransitionType::kSpikeRateBelow;
            ph.transition_threshold = thresh;
        } else if (type_str == "custom") {
            ph.transition = ExperimentPhase::TransitionType::kCustom;
            ph.transition_threshold = thresh;
        }
        // else leave as kTimeOnly
    }

    static StimulusEvent ParseStimulus(const std::string& val) {
        StimulusEvent ev;
        std::istringstream ss(val);

        ss >> ev.label >> ev.start_ms >> ev.end_ms >> ev.intensity;

        // Remaining tokens are comma-separated neuron indices
        std::string neurons_str;
        if (ss >> neurons_str) {
            std::istringstream ns(neurons_str);
            std::string tok;
            while (std::getline(ns, tok, ',')) {
                if (!tok.empty())
                    ev.target_neurons.push_back(
                        static_cast<uint32_t>(std::stoul(tok)));
            }
        }

        return ev;
    }
};

}  // namespace mechabrain

#endif  // FWMC_EXPERIMENT_PROTOCOL_H_
