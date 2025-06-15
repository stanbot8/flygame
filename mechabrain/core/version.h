#pragma once
// FWMC version constants (single source of truth)

namespace mechabrain {

inline constexpr int kVersionMajor = 1;
inline constexpr int kVersionMinor = 0;
inline constexpr int kVersionPatch = 0;
inline constexpr const char* kVersionString = "1.0.0";
inline constexpr const char* kProjectName = "fwmc";
inline constexpr const char* kProjectDescription =
    "FlyWire Mind Couple: C++23 spiking network simulator and "
    "bidirectional neural twinning engine";

}  // namespace mechabrain
