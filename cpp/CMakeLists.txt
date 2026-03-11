cmake_minimum_required(VERSION 3.20)
project(nmfly-sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find MuJoCo ---
# Set MUJOCO_DIR to the root of a MuJoCo installation containing:
#   include/mujoco/mujoco.h
#   mujoco.dll (or lib/libmujoco.so)
#
# Auto-detection: tries pip-installed mujoco package first.

if(NOT MUJOCO_DIR)
    # Try python3 first (avoids Windows Store alias for bare "python").
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

    # Check for pre-existing import lib (from official release or manual generation).
    # If not found, check the build directory (user may have run gen_implib.bat).
    set(MUJOCO_LIB "${MUJOCO_DIR}/mujoco.lib")
    if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${MUJOCO_DIR}/lib/mujoco.lib")
    endif()
    if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/mujoco.lib")
    endif()
    if(NOT EXISTS "${MUJOCO_LIB}")
        # Auto-generate import lib using MSVC tools (available during VS generator build).
        message(STATUS "Generating MuJoCo import library...")
        find_program(_DUMPBIN dumpbin)
        find_program(_LIB lib)
        if(_DUMPBIN AND _LIB)
            # Extract exports.
            execute_process(
                COMMAND "${_DUMPBIN}" /exports "${MUJOCO_DLL}"
                OUTPUT_VARIABLE _DUMP
            )
            # Parse to .def.
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
            "MuJoCo import library not found. Run from VS Developer Command Prompt,\n"
            "or manually generate: cmake\\gen_implib.bat \"${MUJOCO_DLL}\" \"${CMAKE_BINARY_DIR}\"")
    endif()

    add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_DLL}"
        IMPORTED_IMPLIB "${MUJOCO_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
else()
    # Linux/macOS: find shared library.
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
option(NMFLY_VIEWER "Build with GLFW viewer" ON)

if(NMFLY_VIEWER)
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

# --- nmfly-sim ---
add_executable(nmfly-sim src/main.cc)
target_link_libraries(nmfly-sim PRIVATE mujoco::mujoco)
target_compile_definitions(nmfly-sim PRIVATE _CRT_SECURE_NO_WARNINGS)

if(NMFLY_VIEWER)
    target_link_libraries(nmfly-sim PRIVATE glfw)
    target_compile_definitions(nmfly-sim PRIVATE NMFLY_VIEWER=1)
endif()

if(WIN32)
    target_link_libraries(nmfly-sim PRIVATE ws2_32)
    target_compile_definitions(nmfly-sim PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Copy MuJoCo DLL next to executable.
if(WIN32)
    add_custom_command(TARGET nmfly-sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:nmfly-sim>"
    )
endif()
