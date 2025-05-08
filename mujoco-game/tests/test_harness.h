#ifndef MJGAME_TEST_HARNESS_H_
#define MJGAME_TEST_HARNESS_H_

// Shared test infrastructure for mujoco-game unit tests.

#ifdef _MSC_VER
  #pragma warning(disable: 4189)
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

using namespace mjgame;

struct TestEntry { const char* name; void (*fn)(); };
inline std::vector<TestEntry>& GetTests() {
  static std::vector<TestEntry> tests;
  return tests;
}

#define TEST(name) \
  static void test_##name(); \
  struct Register_##name { \
    Register_##name() { GetTests().push_back({#name, test_##name}); } \
  } reg_##name; \
  static void test_##name()

inline int RunAllTests() {
  int passed = 0, failed = 0;
  for (auto& t : GetTests()) {
    try {
      t.fn();
      printf("  PASS  %s\n", t.name);
      passed++;
    } catch (const std::exception& e) {
      printf("  FAIL  %s  (%s)\n", t.name, e.what());
      failed++;
    } catch (...) {
      printf("  FAIL  %s  (unknown exception)\n", t.name);
      failed++;
    }
  }
  printf("\n%d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}

#endif  // MJGAME_TEST_HARNESS_H_
