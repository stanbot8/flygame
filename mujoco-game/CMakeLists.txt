cmake_minimum_required(VERSION 3.20)
project(mujoco-game LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find MuJoCo ---
# Set MUJOCO_DIR to the root of a MuJoCo installation containing:
#   include/mujoco/mujoco.h
#   mujoco.dll (or lib/libmujoco.so)
#
# Auto-detection: tries pip-installed mujoco package first.

if(NOT MUJOCO_DIR)
    find_program(_PYTHON_CMD NAMES python3 python)
    if(_PYTHON_CMD)
        execute_process(
            COMMAND "${_PYTHON_CMD}" -c "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)"
            OUTPUT_VARIABLE _MUJOCO_PIP_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(_MUJOCO_PIP_DIR AND EXISTS "${_MUJOCO_PIP_DIR}/include/mujoco/mujoco.h")
        set(MUJOCO_DIR "${_MUJOCO_PIP_DIR}" CACHE PATH "MuJoCo root directory")
        message(STATUS "Found MuJoCo via pip: ${MUJOCO_DIR}")
    endif()
endif()

if(NOT MUJOCO_DIR)
    message(FATAL_ERROR
        "MuJoCo not found. Set -DMUJOCO_DIR=<path> or install: pip install mujoco")
endif()

set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")

if(WIN32)
    set(MUJOCO_DLL "${MUJOCO_DIR}/mujoco.dll")

    set(MUJOCO_LIB "${MUJOCO_DIR}/mujoco.lib")
    if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${MUJOCO_DIR}/lib/mujoco.lib")
    endif()
    if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/mujoco.lib")
    endif()
    if(NOT EXISTS "${MUJOCO_LIB}")
        message(STATUS "Generating MuJoCo import library...")
        find_program(_DUMPBIN dumpbin)
        find_program(_LIB lib)
        if(_DUMPBIN AND _LIB)
            execute_process(
                COMMAND "${_DUMPBIN}" /exports "${MUJOCO_DLL}"
                OUTPUT_VARIABLE _DUMP
            )
            set(_DEF "LIBRARY mujoco\nEXPORTS\n")
            string(REGEX MATCHALL "[0-9]+[ \t]+[0-9A-Fa-f]+[ \t]+[0-9A-Fa-f]+[ \t]+([a-zA-Z_][a-zA-Z0-9_]+)" _MATCHES "${_DUMP}")
            foreach(_M ${_MATCHES})
                string(REGEX REPLACE ".*[ \t]" "" _SYM "${_M}")
                string(APPEND _DEF "    ${_SYM}\n")
            endforeach()
            file(WRITE "${CMAKE_BINARY_DIR}/mujoco.def" "${_DEF}")
            execute_process(
                COMMAND "${_LIB}" "/def:${CMAKE_BINARY_DIR}/mujoco.def"
                        "/out:${CMAKE_BINARY_DIR}/mujoco.lib" /machine:x64
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            )
            set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/mujoco.lib")
        endif()
    endif()

    if(NOT EXISTS "${MUJOCO_LIB}")
        message(FATAL_ERROR
            "MuJoCo import library not found. Build from VS Developer Command Prompt,\n"
            "or set -DMUJOCO_DIR to a release with mujoco.lib.")
    endif()

    add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_DLL}"
        IMPORTED_IMPLIB "${MUJOCO_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
else()
    find_library(MUJOCO_LIB NAMES mujoco
        PATHS "${MUJOCO_DIR}" "${MUJOCO_DIR}/lib"
        NO_DEFAULT_PATH)
    add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
endif()

message(STATUS "MuJoCo: ${MUJOCO_DIR} (lib: ${MUJOCO_LIB})")

# --- GLFW (optional viewer) ---
option(MJGAME_VIEWER "Build with GLFW viewer" ON)

if(MJGAME_VIEWER)
    include(FetchContent)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# --- Header-only interface library ---
add_library(mujoco-game INTERFACE)
target_include_directories(mujoco-game INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/mujoco-game>
)
target_link_libraries(mujoco-game INTERFACE mujoco::mujoco)

if(MJGAME_VIEWER)
    target_link_libraries(mujoco-game INTERFACE glfw)
    target_compile_definitions(mujoco-game INTERFACE MJGAME_VIEWER=1)
endif()

if(WIN32)
    target_link_libraries(mujoco-game INTERFACE ws2_32)
    target_compile_definitions(mujoco-game INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Export MuJoCo DLL path for consumers to copy.
set(MUJOCO_DLL "${MUJOCO_DLL}" CACHE FILEPATH "MuJoCo DLL path" FORCE)

# Tests (pure functions only, no MuJoCo runtime needed)
option(MJGAME_TESTS "Build mujoco-game tests" OFF)
if(MJGAME_TESTS)
  enable_testing()
  foreach(test_file tests/test_protocol.cc)
    get_filename_component(test_name ${test_file} NAME_WE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
      add_executable(mj_${test_name} ${test_file})
      target_include_directories(mj_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        tests
      )
      if(WIN32)
        target_link_libraries(mj_${test_name} PRIVATE ws2_32)
        target_compile_definitions(mj_${test_name} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
      endif()
      add_test(NAME mj_${test_name} COMMAND mj_${test_name})
    endif()
  endforeach()
endif()
