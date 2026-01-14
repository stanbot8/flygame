#pragma once
// Fly-specific body state extraction from MuJoCo data.
// Knows the Drosophila joint layout: 6 legs x 7 joints = 42 DOF.

#include <algorithm>
#include <cmath>

#include "sim.h"    // mjgame::Sim
#include "types.h"  // mjgame::BodyState

namespace nmfly {

constexpr int kFlyJoints   = 42;  // 6 legs x 7 joints
constexpr int kFlyContacts = 6;   // 6 legs

// Extract fly body state from the simulation.
inline mjgame::BodyState ReadFlyState(const mjgame::Sim& sim) {
    mjgame::BodyState s = {};
    const mjModel* m = sim.model();
    const mjData*  d = sim.data();
    if (!m || !d) return s;

    s.n_joints   = kFlyJoints;
    s.n_contacts = kFlyContacts;

    int nq = std::min(kFlyJoints, static_cast<int>(m->nq));
    for (int i = 0; i < nq; ++i)
        s.joint_angles[i] = static_cast<float>(d->qpos[i]);

    int nv = std::min(kFlyJoints, static_cast<int>(m->nv));
    for (int i = 0; i < nv; ++i)
        s.joint_velocities[i] = static_cast<float>(d->qvel[i]);

    // Contact detection: check which leg geoms are in contact.
    for (int c = 0; c < d->ncon; ++c) {
        int g1 = d->contact[c].geom1;
        int g2 = d->contact[c].geom2;
        for (int leg = 0; leg < 6; ++leg) {
            int gs = leg * 4;
            int ge = gs + 4;
            if ((gs <= g1 && g1 < ge) || (gs <= g2 && g2 < ge))
                s.contacts[leg] = 1.0f;
        }
    }

    // Body velocity (first 6 DOF are the freejoint).
    if (m->nv >= 6) {
        s.body_velocity[0] = static_cast<float>(d->qvel[0]) * 1000.0f;
        s.body_velocity[1] = static_cast<float>(d->qvel[1]) * 1000.0f;
        s.body_velocity[2] = static_cast<float>(d->qvel[5]);
    }

    // Position (from freejoint qpos).
    if (m->nq >= 3) {
        s.position[0] = static_cast<float>(d->qpos[0]);
        s.position[1] = static_cast<float>(d->qpos[1]);
        s.position[2] = static_cast<float>(d->qpos[2]);
    }
    if (m->nq >= 7) {
        double qw = d->qpos[3], qz = d->qpos[6];
        double qx = d->qpos[4], qy = d->qpos[5];
        s.heading = static_cast<float>(
            std::atan2(2.0 * (qw * qz + qx * qy),
                       1.0 - 2.0 * (qy * qy + qz * qz)));
    }

    s.step     = sim.step_count();
    s.sim_time = sim.sim_time();
    return s;
}

}  // namespace nmfly
