#ifndef FWMC_PLATFORM_H_
#define FWMC_PLATFORM_H_

// MSVC uses __restrict, GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define FWMC_RESTRICT __restrict
#else
  #define FWMC_RESTRICT __restrict__
#endif

// AVX2 SIMD support detection (unified macro, no header includes).
// Files that use SIMD intrinsics should include <immintrin.h> themselves.
#if defined(__AVX2__)
  #ifndef FWMC_HAS_AVX2
    #define FWMC_HAS_AVX2 1
  #endif
#endif

// Math constants -- avoids scattered hardcoded pi values and M_PI portability issues.
namespace mechabrain {
inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kTwoPi = 2.0f * kPi;
}  // namespace mechabrain

#endif  // FWMC_PLATFORM_H_
