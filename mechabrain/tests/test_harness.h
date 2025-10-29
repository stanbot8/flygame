#ifndef PARABRAIN_TEST_HARNESS_H_
#define PARABRAIN_TEST_HARNESS_H_

// Shared test infrastructure for mechabrain unit tests.
// Each test file includes this header, defines TEST() functions,
// and calls RUN_TESTS() from main().

#ifdef _MSC_VER
  #pragma warning(disable: 4189)  // local variable initialized but not referenced (assert-only vars)
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
#ifdef _WIN32
#include <direct.h>
#endif

#include "core/log.h"

using namespace mechabrain;

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

// Helper: write binary data to a temp file and return the path
inline std::string WriteTempFile(const char* name, const void* data, size_t size) {
  std::string path = std::string("test_tmp_") + name;
  FILE* f = fopen(path.c_str(), "wb");
  assert(f);
  fwrite(data, 1, size, f);
  fclose(f);
  return path;
}

inline int RunAllTests() {
  // Suppress log output during tests
#ifdef _WIN32
  SetLogSink(fopen("NUL", "w"));
#else
  SetLogSink(fopen("/dev/null", "w"));
#endif

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

#endif  // PARABRAIN_TEST_HARNESS_H_
