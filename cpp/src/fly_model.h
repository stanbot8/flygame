#pragma once
// Embedded stick-figure Drosophila MJCF model.
//
// A minimal 6-legged fly for testing and prototyping without
// requiring NeuroMechFly. Matches the Python nmfly.fly_model output.

#include <string>

namespace nmfly {

inline std::string StickFlyMJCF() {
    // Leg names and positions.
    struct LegDef { const char* name; int side; float y_off; };
    constexpr LegDef legs[] = {
        {"LF", -1, 0.02f}, {"RF", 1, 0.02f},
        {"LM", -1, 0.00f}, {"RM", 1, 0.00f},
        {"LH", -1,-0.02f}, {"RH", 1,-0.02f},
    };

    std::string legs_xml;
    std::string actuators_xml;
    const char* joint_names[] = {"ThC_yaw", "ThC_pitch", "CTr_pitch", "FTi_pitch"};

    for (auto& L : legs) {
        float cx = L.side * 0.008f;
        float cy = L.y_off;
        char buf[2048];
        snprintf(buf, sizeof(buf), R"(
        <body name="%s_coxa" pos="%.4f %.4f 0">
          <joint name="%s_ThC_yaw" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
          <joint name="%s_ThC_pitch" type="hinge" axis="1 0 0" range="-1 1"/>
          <geom type="capsule" size="0.001" fromto="0 0 0 %.4f 0 -0.005" rgba="0.7 0.5 0.3 1"/>
          <body name="%s_femur" pos="%.4f 0 -0.005">
            <joint name="%s_CTr_pitch" type="hinge" axis="1 0 0" range="-1.5 0.5"/>
            <geom type="capsule" size="0.0008" fromto="0 0 0 0 0 -0.012" rgba="0.7 0.5 0.3 1"/>
            <body name="%s_tibia" pos="0 0 -0.012">
              <joint name="%s_FTi_pitch" type="hinge" axis="1 0 0" range="-0.5 2.0"/>
              <geom type="capsule" size="0.0006" fromto="0 0 0 0 0 -0.01" rgba="0.6 0.4 0.2 1"/>
              <body name="%s_tarsus" pos="0 0 -0.01">
                <geom type="sphere" size="0.0005" rgba="0.5 0.3 0.1 1"/>
              </body>
            </body>
          </body>
        </body>)",
            L.name, cx, cy,
            L.name, L.name,
            L.side * 0.01f,
            L.name, L.side * 0.01f,
            L.name,
            L.name, L.name,
            L.name);
        legs_xml += buf;

        for (auto jn : joint_names) {
            char abuf[256];
            snprintf(abuf, sizeof(abuf),
                R"(    <position name="%s_%s" joint="%s_%s" kp="0.001" ctrlrange="-2 2"/>
)",
                L.name, jn, L.name, jn);
            actuators_xml += abuf;
        }
    }

    return R"(<?xml version="1.0"?>
<mujoco model="nmfly_stick_fly">
  <option timestep="0.0002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.0001" armature="0.00001"/>
    <geom contype="1" conaffinity="1" friction="1.0 0.005 0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" rgba="0.2 0.3 0.2 1"/>

    <body name="thorax" pos="0 0 0.025">
      <freejoint name="root"/>
      <geom type="ellipsoid" size="0.004 0.012 0.003" rgba="0.8 0.6 0.3 1"/>

      <body name="head" pos="0 0.014 0.001">
        <geom type="sphere" size="0.004" rgba="0.7 0.5 0.2 1"/>
      </body>

      <body name="abdomen" pos="0 -0.015 -0.001">
        <geom type="ellipsoid" size="0.003 0.01 0.0025" rgba="0.6 0.4 0.2 1"/>
      </body>

      <body name="wing_L" pos="-0.004 0.005 0.003">
        <joint name="wing_L_stroke" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
        <geom type="ellipsoid" size="0.015 0.004 0.0003" rgba="0.9 0.9 1.0 0.3"/>
      </body>
      <body name="wing_R" pos="0.004 0.005 0.003">
        <joint name="wing_R_stroke" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
        <geom type="ellipsoid" size="0.015 0.004 0.0003" rgba="0.9 0.9 1.0 0.3"/>
      </body>

)" + legs_xml + R"(
    </body>
  </worldbody>

  <actuator>
)" + actuators_xml + R"(
    <position name="wing_L_act" joint="wing_L_stroke" kp="0.01" ctrlrange="-2.5 2.5"/>
    <position name="wing_R_act" joint="wing_R_stroke" kp="0.01" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>)";
}

}  // namespace nmfly
