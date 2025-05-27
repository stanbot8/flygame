cmake_minimum_required(VERSION 3.19)
project(mechabrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(mechabrain INTERFACE)
target_include_directories(mechabrain INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/mechabrain>
)

# Optional OpenMP for parallel neuron dynamics
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mechabrain INTERFACE OpenMP::OpenMP_CXX)
endif()

# Tests
option(MECHABRAIN_TESTS "Build mechabrain tests" OFF)
if(MECHABRAIN_TESTS)
  enable_testing()
  foreach(test_file
      tests/test_brain_sim.cc
      tests/test_core.cc
      tests/test_parametric.cc
      tests/test_tissue.cc)
    get_filename_component(test_name ${test_file} NAME_WE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
      add_executable(mb_${test_name} ${test_file})
      target_link_libraries(mb_${test_name} PRIVATE mechabrain)
      target_include_directories(mb_${test_name} PRIVATE tests)
      if(NOT MSVC)
        target_link_libraries(mb_${test_name} PRIVATE pthread)
      endif()
      if(OpenMP_CXX_FOUND)
        target_link_libraries(mb_${test_name} PRIVATE OpenMP::OpenMP_CXX)
      endif()
      add_test(NAME mb_${test_name} COMMAND mb_${test_name})
    endif()
  endforeach()
endif()
