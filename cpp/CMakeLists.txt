cmake_minimum_required(VERSION 3.20)
project(nmfly-sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- mujoco-game (shared MuJoCo engine) ---
add_subdirectory(${CMAKE_SOURCE_DIR}/../../mujoco-game
                 ${CMAKE_BINARY_DIR}/mujoco-game)

# --- GLFW viewer extras (ImGui, glad) ---
option(NMFLY_VIEWER "Build with viewer (ImGui overlay)" ON)

if(NMFLY_VIEWER)
    # glad (OpenGL loader)
    add_library(glad STATIC vendor/glad/src/gl.c)
    target_include_directories(glad PUBLIC vendor/glad/include)

    # Dear ImGui (vendored)
    add_library(imgui STATIC
        vendor/imgui-src/imgui.cpp
        vendor/imgui-src/imgui_draw.cpp
        vendor/imgui-src/imgui_widgets.cpp
        vendor/imgui-src/imgui_tables.cpp
        vendor/imgui-src/backends/imgui_impl_glfw.cpp
        vendor/imgui-src/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC
        vendor/imgui-src
        vendor/imgui-src/backends
    )
    target_link_libraries(imgui PUBLIC glfw glad)
    target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD2)
endif()

# --- Embedded brain (optional mechabrain) ---
option(NMFLY_BRAIN "Build with embedded mechabrain simulation" ON)
set(MECHABRAIN_DIR "" CACHE PATH "Path to mechabrain directory")

# --- mecha-fly (NMF fly body) ---
set(MECHA_FLY_DIR "${CMAKE_SOURCE_DIR}/../mecha-fly")

# --- nmfly-sim ---
add_executable(nmfly-sim src/main.cc)
target_link_libraries(nmfly-sim PRIVATE mujoco-game)
target_include_directories(nmfly-sim PRIVATE "${MECHA_FLY_DIR}/core")
target_compile_definitions(nmfly-sim PRIVATE _CRT_SECURE_NO_WARNINGS)

if(NMFLY_VIEWER)
    target_link_libraries(nmfly-sim PRIVATE glad imgui)
    target_compile_definitions(nmfly-sim PRIVATE NMFLY_VIEWER=1)
endif()

if(NMFLY_BRAIN)
    if(NOT MECHABRAIN_DIR OR NOT EXISTS "${MECHABRAIN_DIR}/core/neuron_array.h")
        # Auto-detect: sibling mechabrain repo
        set(_BRAIN_GUESS "${CMAKE_SOURCE_DIR}/../../mechabrain")
        if(EXISTS "${_BRAIN_GUESS}/core/neuron_array.h")
            set(MECHABRAIN_DIR "${_BRAIN_GUESS}" CACHE PATH "" FORCE)
            message(STATUS "Auto-detected mechabrain: ${MECHABRAIN_DIR}")
        else()
            message(FATAL_ERROR
                "NMFLY_BRAIN=ON but mechabrain not found.\n"
                "Set -DMECHABRAIN_DIR=<path-to-mechabrain>")
        endif()
    endif()
    target_include_directories(nmfly-sim PRIVATE "${MECHABRAIN_DIR}")
    target_compile_definitions(nmfly-sim PRIVATE NMFLY_BRAIN=1)
    # mechabrain uses C++23 features (std::expected)
    set_target_properties(nmfly-sim PROPERTIES CXX_STANDARD 23)
    message(STATUS "Embedded brain: ${MECHABRAIN_DIR}")

    # Copy .brain spec files to exe_dir/data/ if available
    set(_BRAIN_EXAMPLES "${CMAKE_SOURCE_DIR}/../../fwmc/examples")
    if(EXISTS "${_BRAIN_EXAMPLES}/drosophila_full.brain")
        add_custom_command(TARGET nmfly-sim POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nmfly-sim>/data"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_BRAIN_EXAMPLES}/drosophila_full.brain"
                "$<TARGET_FILE_DIR:nmfly-sim>/data/drosophila_full.brain"
        )
    endif()

    # Optional OpenMP for mechabrain parallelism
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nmfly-sim PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Copy MuJoCo DLL next to executable.
if(WIN32 AND MUJOCO_DLL)
    add_custom_command(TARGET nmfly-sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:nmfly-sim>"
    )
endif()

# Copy NMF model and mesh data to exe_dir/data/ for runtime discovery.
# Prefer mecha-fly data dir, fall back to local data/.
set(_NMF_DATA "${MECHA_FLY_DIR}/data")
if(NOT EXISTS "${_NMF_DATA}/nmf_complete.xml")
    set(_NMF_DATA "${CMAKE_SOURCE_DIR}/data")
endif()
if(EXISTS "${_NMF_DATA}/nmf_complete.xml")
    add_custom_command(TARGET nmfly-sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nmfly-sim>/data"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_NMF_DATA}" "$<TARGET_FILE_DIR:nmfly-sim>/data"
    )
endif()
