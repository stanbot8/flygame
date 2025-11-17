#ifndef MECHA_FLY_TEST_HARNESS_H_
#define MECHA_FLY_TEST_HARNESS_H_

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nmfly;

// CHECK: like assert() but never compiled out, and throws so the harness
// can catch and report which test failed.
#define CHECK(expr) \
  do { if (!(expr)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
    ": CHECK failed: " #expr); } while(0)

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

#endif  // MECHA_FLY_TEST_HARNESS_H_
