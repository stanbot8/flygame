#pragma once
// Forward kinematics for NeuroMechFly legs.
// Pure math — no MuJoCo dependency.

#include "nmf_leg_geometry.h"

namespace nmfly {

struct FkResult {
    Vec3 foot_pos;    // tarsus1 joint origin in thorax frame
    Mat3 foot_orient; // cumulative rotation at tarsus1
};

// Forward kinematics: joint angles -> foot position in thorax frame.
//
// angles[] is indexed by actuator order (see Joint enum):
//   [0]=Coxa(pitch), [1]=CoxaRoll, [2]=CoxaYaw,
//   [3]=Femur(pitch), [4]=FemurRoll, [5]=Tibia, [6]=Tarsus1
//
// Rotations are applied in MJCF body declaration order to match MuJoCo physics:
//   Coxa body: yaw(X)[2] -> pitch(Y)[0] -> roll(Z)[1]  (outermost first)
//   Femur body: pitch(Y)[3] -> roll(Z)[4]
//   Tibia body: pitch(Y)[5]
//   Tarsus1 body: pitch(Y)[6]
//
// Translation between joints is along local -Z (legs point downward from thorax).
inline FkResult FlyFK(int leg, const double angles[7]) {
    const auto& g = kLegGeom[leg];

    // Start at coxa attachment in thorax frame.
    Vec3 pos = g.attachment;
    Mat3 rot = {};  // identity

    // Coxa: 3 DOF in MJCF declaration order (yaw, pitch, roll).
    rot = rot * RotX(angles[kCoxaYaw]);   // yaw — outermost (declared first in XML)
    rot = rot * RotY(angles[kCoxa]);      // pitch
    rot = rot * RotZ(angles[kCoxaRoll]);  // roll — innermost (declared last)

    // Translate to femur joint along local -Z.
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[0]};

    // Femur: 2 DOF (pitch, roll — MJCF order matches actuator order here).
    rot = rot * RotY(angles[kFemur]);
    rot = rot * RotZ(angles[kFemurRoll]);

    // Translate to tibia joint.
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[1]};

    // Tibia: 1 DOF.
    rot = rot * RotY(angles[kTibia]);

    // Translate to tarsus1 joint.
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[2]};

    // Tarsus1: 1 DOF (included in orientation; foot_pos is at joint origin).
    rot = rot * RotY(angles[kTarsus1]);

    return {pos, rot};
}

// Extended FK returning intermediate joint positions (for fitting).
struct FkChain {
    Vec3 femur_pos;   // femur joint origin (= "knee")
    Vec3 tibia_pos;   // tibia joint origin
    Vec3 foot_pos;    // tarsus1 joint origin
};

inline FkChain FlyFKChain(int leg, const double angles[7]) {
    const auto& g = kLegGeom[leg];
    Vec3 pos = g.attachment;
    Mat3 rot = {};

    rot = rot * RotX(angles[kCoxaYaw]);
    rot = rot * RotY(angles[kCoxa]);
    rot = rot * RotZ(angles[kCoxaRoll]);
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[0]};
    Vec3 femur_pos = pos;

    rot = rot * RotY(angles[kFemur]);
    rot = rot * RotZ(angles[kFemurRoll]);
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[1]};
    Vec3 tibia_pos = pos;

    rot = rot * RotY(angles[kTibia]);
    pos = pos + rot * Vec3{0, 0, -g.inter_joint[2]};

    return {femur_pos, tibia_pos, pos};
}

}  // namespace nmfly
