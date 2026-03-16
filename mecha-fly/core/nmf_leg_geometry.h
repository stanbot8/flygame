#pragma once
// Kinematic chain data and math types for NeuroMechFly legs.
// Geometry extracted from nmf_complete.xml body pos attributes.

#include <cmath>

namespace nmfly {

// --- Math types ---

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3 operator+(Vec3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    double dot(Vec3 b) const { return x*b.x + y*b.y + z*b.z; }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

struct Mat3 {
    double m[3][3] = {{1,0,0},{0,1,0},{0,0,1}};  // identity default

    Vec3 operator*(Vec3 v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    Mat3 operator*(const Mat3& b) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r.m[i][j] = 0;
                for (int k = 0; k < 3; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
            }
        return r;
    }
};

// Rotation matrix around X axis by angle (radians).
inline Mat3 RotX(double a) {
    double c = std::cos(a), s = std::sin(a);
    return {{{1, 0, 0}, {0, c, -s}, {0, s, c}}};
}

// Rotation matrix around Y axis.
inline Mat3 RotY(double a) {
    double c = std::cos(a), s = std::sin(a);
    return {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}};
}

// Rotation matrix around Z axis.
inline Mat3 RotZ(double a) {
    double c = std::cos(a), s = std::sin(a);
    return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

// --- Leg geometry ---

// Leg indices (matches actuator order and walk data order).
enum Leg { kLF = 0, kLM = 1, kLH = 2, kRF = 3, kRM = 4, kRH = 5 };

// Joint indices within a leg (actuator order, walk data order).
// Note: FK applies coxa rotations in MJCF declaration order (yaw, pitch, roll)
// not actuator index order. See nmf_fk.h.
enum Joint {
    kCoxa = 0,       // Y axis (pitch)
    kCoxaRoll = 1,   // Z axis (roll)
    kCoxaYaw = 2,    // X axis (yaw)
    kFemur = 3,      // Y axis (pitch)
    kFemurRoll = 4,  // Z axis (roll)
    kTibia = 5,      // Y axis (pitch)
    kTarsus1 = 6     // Y axis (pitch)
};

struct LegGeometry {
    Vec3 attachment;       // coxa origin in thorax frame
    double inter_joint[3]; // distances: coxa->femur, femur->tibia, tibia->tarsus1
    double mirror_y;       // +1.0 for left legs, -1.0 for right legs
};

// Geometry data extracted from nmf_complete.xml body pos Euclidean norms.
// Front, Mid, Hind each have unique dimensions. Left/right differ only in Y sign.
// Inter-joint distances use the Euclidean norm of the child body pos vector
// (small off-axis components ~0.001 are present in the data but the norm is
// dominated by the Z component).
constexpr LegGeometry kLegGeom[6] = {
    // LF (Front left)
    {{-0.160596,  0.171588, -0.230215}, {0.365456, 0.705223, 0.518397},  1.0},
    // LM (Mid left)
    {{-0.556871,  0.124901, -0.482209}, {0.181447, 0.783670, 0.667326},  1.0},
    // LH (Hind left)
    {{-0.783722,  0.086993, -0.496612}, {0.199417, 0.836314, 0.684456},  1.0},
    // RF (Front right)
    {{-0.160596, -0.171588, -0.230215}, {0.365456, 0.705223, 0.518397}, -1.0},
    // RM (Mid right)
    {{-0.556871, -0.124901, -0.482209}, {0.181447, 0.783670, 0.667326}, -1.0},
    // RH (Hind right)
    {{-0.783722, -0.086993, -0.496612}, {0.199417, 0.836314, 0.684456}, -1.0},
};

}  // namespace nmfly
