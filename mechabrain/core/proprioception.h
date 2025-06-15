#ifndef FWMC_PROPRIOCEPTION_H_
#define FWMC_PROPRIOCEPTION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"

namespace mechabrain {

// Proprioceptive feedback: maps body state to sensory currents injected
// into VNC (ventral nerve cord) neurons, closing the sensorimotor loop.
//
// In Drosophila, leg proprioceptors (femoral chordotonal organs) encode
// joint angles and velocities. Campaniform sensilla detect load (ground
// contact). Halteres sense body rotation (gyroscopic feedback).
//
// This module converts MuJoCo body state into excitatory currents on
// VNC sensory neurons, split into functional channels:
//   - Joint angle sensors (6 legs x 7 joints = 42 channels)
//   - Contact sensors (6 legs)
//   - Body velocity (forward, lateral, yaw)
//   - Haltere-like rotation sensing (L/R asymmetric)
struct ProprioConfig {
  float angle_gain = 4.0f;      // joint angle -> current scaling
  float velocity_gain = 2.0f;   // joint velocity -> current scaling
  float contact_gain = 8.0f;    // ground contact -> current (strong signal)
  float body_vel_gain = 3.0f;   // body velocity -> current
  float haltere_gain = 5.0f;    // yaw rate -> asymmetric L/R current
};

// Body state from MuJoCo (matches flygame BodyState layout).
// Can be populated from either local BodyViewport or remote TCP.
struct ProprioState {
  float joint_angles[42] = {};    // 6 legs x 7 joints (radians)
  float joint_velocities[42] = {};
  float contacts[6] = {};         // per-leg ground contact [0,1]
  float body_velocity[3] = {};    // [fwd mm/s, lat mm/s, yaw rad/s]
};

// Assigns VNC neurons to proprioceptive channels.
// Neurons are distributed across channels proportionally.
struct ProprioMap {
  // Neuron index ranges for each sensory channel
  std::vector<uint32_t> joint_angle_neurons[42];   // per joint
  std::vector<uint32_t> contact_neurons[6];         // per leg
  std::vector<uint32_t> body_vel_neurons;           // forward/lateral
  std::vector<uint32_t> haltere_left;               // left-side rotation
  std::vector<uint32_t> haltere_right;              // right-side rotation
  bool initialized = false;

  // Auto-assign VNC neurons to proprioceptive channels.
  // vnc_region: region index for VNC (typically 5 in drosophila_full.brain)
  // Splits VNC neurons into sensory (30%), motor (rest).
  // Sensory neurons are distributed across channels by position.
  void Init(const NeuronArray& neurons, uint8_t vnc_region,
            float midline_x = 250.0f) {
    // Collect VNC neuron indices
    std::vector<uint32_t> vnc;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == vnc_region)
        vnc.push_back(static_cast<uint32_t>(i));
    }
    if (vnc.empty()) return;

    // Use first 30% as sensory afferents (the rest are motor/interneurons)
    size_t n_sensory = vnc.size() * 3 / 10;
    if (n_sensory < 100) n_sensory = std::min(vnc.size(), size_t(100));

    // Distribute sensory neurons across channels:
    //   42 joint angle channels (1 neuron each minimum)
    //   6 contact channels (larger populations for strong signal)
    //   3 body velocity channels
    //   2 haltere channels (L/R)
    // Total: 53 channels. Distribute proportionally.
    size_t per_joint = std::max(size_t(1), n_sensory / 80);
    size_t per_contact = std::max(size_t(3), n_sensory / 20);
    size_t per_body = std::max(size_t(2), n_sensory / 30);

    size_t idx = 0;
    auto assign = [&](std::vector<uint32_t>& dest, size_t count) {
      for (size_t c = 0; c < count && idx < n_sensory; ++c, ++idx)
        dest.push_back(vnc[idx]);
    };

    for (int j = 0; j < 42; ++j) assign(joint_angle_neurons[j], per_joint);
    for (int l = 0; l < 6; ++l) assign(contact_neurons[l], per_contact);
    assign(body_vel_neurons, per_body * 3);

    // Halteres: split remaining sensory neurons L/R by x-coordinate
    for (; idx < n_sensory; ++idx) {
      uint32_t ni = vnc[idx];
      if (neurons.x[ni] < midline_x)
        haltere_left.push_back(ni);
      else
        haltere_right.push_back(ni);
    }

    initialized = true;
  }

  // Inject proprioceptive currents into neuron i_ext based on body state.
  void Inject(NeuronArray& neurons, const ProprioState& state,
              const ProprioConfig& cfg) const {
    if (!initialized) return;

    auto inject = [&](const std::vector<uint32_t>& indices, float current) {
      for (uint32_t i : indices)
        neurons.i_ext[i] += current;
    };

    // Joint angle sensors: sigmoid activation, scales with angle magnitude
    for (int j = 0; j < 42; ++j) {
      float activation = sigmoid(std::abs(state.joint_angles[j]) * 2.0f);
      inject(joint_angle_neurons[j], activation * cfg.angle_gain);
    }

    // Contact sensors: binary-ish (strong signal when foot on ground)
    for (int l = 0; l < 6; ++l) {
      float contact_current = state.contacts[l] * cfg.contact_gain;
      inject(contact_neurons[l], contact_current);
    }

    // Body velocity: forward and lateral
    float fwd_activation = sigmoid(std::abs(state.body_velocity[0]) * 0.1f);
    float lat_activation = sigmoid(std::abs(state.body_velocity[1]) * 0.1f);
    inject(body_vel_neurons, (fwd_activation + lat_activation) * cfg.body_vel_gain);

    // Haltere feedback: yaw rate drives asymmetric L/R excitation.
    // Positive yaw (turning left) excites right haltere, inhibits left.
    // This provides corrective feedback for straight-line walking.
    float yaw = state.body_velocity[2];  // rad/s
    float haltere_signal = std::clamp(yaw * cfg.haltere_gain, -10.0f, 10.0f);
    inject(haltere_left, std::max(0.0f, -haltere_signal));
    inject(haltere_right, std::max(0.0f, haltere_signal));
  }

 private:
  static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-(x - 2.0f)));
  }
};

// Read proprioceptive state from a local MuJoCo body sim.
// Template to avoid hard dependency on MuJoCo types.
template <typename MjModel, typename MjData>
ProprioState ReadProprioFromMuJoCo(const MjModel* model, const MjData* data) {
  ProprioState state;

  // Joint angles and velocities: iterate over actuated joints.
  // NMF model has 48 actuators, but we care about 42 leg joints (7 per leg).
  int n_joints = std::min(42, model->njnt - 1);  // skip freejoint
  for (int j = 0; j < n_joints; ++j) {
    int jnt_id = j + 1;  // skip freejoint (index 0)
    if (jnt_id >= model->njnt) break;
    int qa = model->jnt_qposadr[jnt_id];
    int va = model->jnt_dofadr[jnt_id];
    state.joint_angles[j] = static_cast<float>(data->qpos[qa]);
    state.joint_velocities[j] = static_cast<float>(data->qvel[va]);
  }

  // Ground contacts: check which leg geoms are touching the floor.
  // Leg tip geom names follow pattern: LF_tarsus5, LM_tarsus5, etc.
  // Simplified: check contacts by body z-position or contact array.
  for (int c = 0; c < data->ncon && c < 100; ++c) {
    int g1 = data->contact[c].geom1;
    int g2 = data->contact[c].geom2;
    // Map geom to leg index (simplified: use geom body parent)
    auto leg_from_geom = [&](int g) -> int {
      if (g < 0 || g >= model->ngeom) return -1;
      int body = model->geom_bodyid[g];
      // Leg bodies are typically ordered: LF, LM, LH, RF, RM, RH
      // Each leg has ~10 bodies. Rough mapping by body index range.
      if (body <= 0) return -1;
      int leg = (body - 1) / 10;  // approximate
      return (leg >= 0 && leg < 6) ? leg : -1;
    };
    int leg = leg_from_geom(g1);
    if (leg < 0) leg = leg_from_geom(g2);
    if (leg >= 0 && leg < 6) state.contacts[leg] = 1.0f;
  }

  // Body velocity from freejoint
  state.body_velocity[0] = static_cast<float>(data->qvel[0]) * 1000.0f; // m/s -> mm/s
  state.body_velocity[1] = static_cast<float>(data->qvel[1]) * 1000.0f;
  state.body_velocity[2] = static_cast<float>(data->qvel[5]);            // yaw rad/s

  return state;
}

}  // namespace mechabrain

#endif  // FWMC_PROPRIOCEPTION_H_
