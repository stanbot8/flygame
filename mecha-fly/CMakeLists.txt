cmake_minimum_required(VERSION 3.19)
project(mecha-fly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
else()
  add_compile_options(-Wall -Wextra)
endif()

# Header-only library
add_library(mecha-fly INTERFACE)
target_include_directories(mecha-fly INTERFACE core)

# Tests
option(MECHA_FLY_TESTS "Build mecha-fly tests" ON)
if(MECHA_FLY_TESTS)
  enable_testing()
  add_executable(mf_test_cpg tests/test_cpg.cc)
  target_include_directories(mf_test_cpg PRIVATE core tests)
  add_test(NAME mf_test_cpg COMMAND mf_test_cpg)
endif()
