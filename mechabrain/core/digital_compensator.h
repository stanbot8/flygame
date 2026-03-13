#ifndef PARABRAIN_DIGITAL_COMPENSATOR_H_
#define PARABRAIN_DIGITAL_COMPENSATOR_H_

// General-purpose digital twin compensation for neural prosthesis.
//
// Given a circuit with some neurons silenced (ablated/dead), the compensator
// runs a digital twin for those neurons and injects twin output spikes
// back into the main circuit. This allows the circuit to maintain function
// despite neuron loss.
//
// Usage:
//   DigitalCompensator comp;
//   comp.Init(neurons, synapses, types);
//   comp.SetSilenced(silenced_mask);
//   // In simulation loop:
//   comp.Step(neurons, synapses, types, dt_ms, sim_time);
//   // After Step(), neurons.spiked and neurons.i_syn reflect combined
//   // bio + twin activity for non-silenced neurons.

#include <cstdint>
#include <vector>

#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/cell_types.h"

namespace mechabrain {

struct DigitalCompensator {
  NeuronArray twin;
  std::vector<bool> silenced;
  std::vector<uint8_t> combined_spikes;
  bool initialized = false;

  // Initialize the twin as a copy of the main circuit's neuron parameters.
  void Init(const NeuronArray& neurons, const CellTypeManager& /*types*/) {
    uint32_t n = static_cast<uint32_t>(neurons.n);
    twin.Resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      twin.type[i] = neurons.type[i];
      twin.region[i] = neurons.region[i];
    }
    silenced.assign(n, false);
    combined_spikes.resize(n, 0);
    ResetTwin();
    initialized = true;
  }

  // Reset twin state to resting potential.
  void ResetTwin() {
    for (uint32_t i = 0; i < twin.n; ++i) {
      twin.v[i] = -65.0f;
      twin.u[i] = -13.0f;
      twin.i_syn[i] = 0.0f;
      twin.i_ext[i] = 0.0f;
      twin.spiked[i] = 0;
    }
  }

  // Set which neurons are silenced (ablated). The twin will fill in for these.
  void SetSilenced(const std::vector<bool>& mask) {
    silenced = mask;
  }

  // Run one compensation step. Call AFTER setting external input on `neurons`
  // but BEFORE propagating spikes and stepping dynamics.
  //
  // This method:
  //   1. Copies external input to twin for silenced neurons
  //   2. Builds combined spike vector (bio for alive, twin for silenced)
  //   3. Propagates combined spikes to both bio and twin synaptic input
  //   4. Steps twin dynamics for silenced neurons
  //   5. Zeros bio state for silenced neurons (keeps them silent)
  //
  // After calling this, the caller should:
  //   - Step bio dynamics (IzhikevichStep); silenced neurons will be clamped
  //   - The synaptic input to non-silenced neurons includes twin output
  void PreStep(NeuronArray& neurons, SynapseTable& synapses,
               const CellTypeManager& /*types*/, float dt_ms) {
    uint32_t n = static_cast<uint32_t>(neurons.n);

    // Give twin the same external input as bio would get for silenced neurons
    for (uint32_t i = 0; i < n; ++i) {
      twin.i_ext[i] = silenced[i] ? neurons.i_ext[i] : 0.0f;
    }

    // Build combined spike vector
    for (uint32_t i = 0; i < n; ++i) {
      combined_spikes[i] = silenced[i] ? twin.spiked[i] : neurons.spiked[i];
    }

    // Propagate combined spikes to twin's synaptic input
    twin.DecaySynapticInput(dt_ms, 3.0f);
    synapses.PropagateSpikes(combined_spikes.data(), twin.i_syn.data(), 1.0f);
    // Zero twin synaptic input for non-silenced neurons (only twin runs silenced ones)
    for (uint32_t i = 0; i < n; ++i) {
      if (!silenced[i]) {
        twin.i_syn[i] = 0.0f;
        twin.i_ext[i] = 0.0f;
      }
    }

    // Propagate combined spikes to bio circuit
    neurons.DecaySynapticInput(dt_ms, 3.0f);
    synapses.PropagateSpikes(combined_spikes.data(), neurons.i_syn.data(), 1.0f);

    // Silence ablated bio neurons
    for (uint32_t i = 0; i < n; ++i) {
      if (silenced[i]) {
        neurons.i_ext[i] = 0.0f;
        neurons.v[i] = -65.0f;
        neurons.u[i] = -13.0f;
        neurons.spiked[i] = 0;
        neurons.i_syn[i] = 0.0f;
      }
    }
  }

  // Step twin dynamics. Call AFTER bio IzhikevichStep.
  void PostStep(const CellTypeManager& types, float dt_ms, float sim_time) {
    IzhikevichStepHeterogeneousFast(twin, dt_ms, sim_time, types);
  }
};

}  // namespace mechabrain

#endif  // PARABRAIN_DIGITAL_COMPENSATOR_H_
