#pragma once
// nmf_wing.h — Drosophila wing kinematics (stub).
//
// The NMF MJCF model does not have wing joints or actuators, so this
// controller is currently a no-op. Wing meshes exist for visualization
// only. To enable flight, the MJCF would need wing hinge joints and
// corresponding actuators added after the leg joints.

#include <cmath>

namespace nmfly {

// Actuator layout in the NMF MJCF: 42 leg joints + 6 adhesion = 48 total.
constexpr int kLegActuators     = 42;
constexpr int kAdhesionStart    = 42;
constexpr int kTotalActuators   = 48;

// Wing controller stub. No-op until wing actuators are added to the MJCF.
class NmfWing {
public:
    bool active = false;

    void Update(float) {}
    void GetCtrl(double*) const {}

private:
    static constexpr float kPi = 3.14159265358979323846f;
    float phase_ = 0.0f;
};

}  // namespace nmfly
