# mujoco-game

Header-only C++20 MuJoCo simulation framework for brain-body experiments. Provides the shared infrastructure that any animal body simulation needs:

- **sim.h** — MuJoCo physics stepping, GLFW viewer with mouse/keyboard controls, camera follow, HUD overlay
- **tcp.h** — FWMC bridge protocol v1 TCP client for connecting to spiking brain simulators
- **types.h** — Framework-agnostic types: MotorCommand, SensoryReading, BodyState

Animal-specific code (gait generation, skeleton/model building, proprioception encoding) lives in each animal's own project.

## Usage

```cmake
add_subdirectory(../mujoco-game ${CMAKE_BINARY_DIR}/mujoco-game)
target_link_libraries(my-animal PRIVATE mujoco-game)
```

Then in C++:
```cpp
#include "sim.h"
#include "tcp.h"

mjgame::Sim sim;
mjgame::SimConfig cfg;
cfg.window_title = "my-animal";
sim.Setup(cfg, mjcf_xml);
```

No external dependencies beyond MuJoCo. Optional GLFW for viewer (auto-fetched by CMake).
