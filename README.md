# flygame

**C++ reimplementation of [flygym](https://github.com/NeLy-EPFL/flygym)'s locomotion core on MuJoCo, with spiking connectome integration.**

**flygym** is a Python framework (~16k lines) for simulating a biomechanical *Drosophila* body on MuJoCo. It runs at ~0.03x realtime. **flygame** rewrites the simulation loop, CPG, and gait trajectory system in C++, hitting **7x realtime** headless (~230x faster). Fast enough to close the loop with a spiking brain in real time. It does not reimplement flygym's vision, olfaction, terrain, or RL interfaces.

For the Python bridge and adapter framework, see [pyflygame](https://github.com/stanbot8/pyflygame).

## Architecture

```
Your Controller                 flygame              MuJoCo
 (any framework)                  Bridge                  Physics
       |                            |                        |
       +-- MotorCommand ---------> | --> joint targets ----> mj_step()
       |   (fwd, turn,             |                        |
       |    approach, freeze)      |                        |
       |                            |                        |
       <-- SensoryReading[] ------ | <-- proprioception --+ |
           (activation,            |     (angles, contacts,  |
            raw_value)             |      body velocity)     |
```

## Build

Requires MuJoCo (pip or system install) and CMake 3.20+.

```bash
# Configure (auto-detects MuJoCo from pip)
cmake -G "Visual Studio 17 2022" -S cpp -B build

# Or specify MuJoCo location
cmake -S cpp -B build -DMUJOCO_DIR=/path/to/mujoco

# Build
cmake --build build --config Release
```

On Windows with pip-installed MuJoCo, the build auto-generates an import library from `mujoco.dll`. If that fails, run manually from a VS Developer Command Prompt:
```
cpp\cmake\gen_implib.bat "path\to\mujoco.dll" build
```

## Usage

```bash
# Standalone: walk forward with NeuroMechFly
nmfly-sim

# Connect to FWMC spiking brain
nmfly-sim --brain 127.0.0.1:9100

# Custom model, headless benchmark
nmfly-sim --model my_fly.xml --headless --duration 60

# All options
nmfly-sim --help
```

## Performance

| Mode | Speed | Steps/sec |
|------|-------|-----------|
| Headless | 7x realtime | ~35,000 |
| With viewer | 5.7x realtime | ~29,000 |

Compared to Python flygym (~0.03x realtime), the C++ sim is ~230x faster.

## What's reimplemented from flygym

| flygym (Python) | flygame (C++) | Notes |
|---|---|---|
| `PreprogrammedSteps` | `mecha-fly` `nmf_walk_data.h` + `WalkLerp()` | Same cubic-spline walk trajectories, vendored as lookup table |
| `CPGNetwork` (Kuramoto) | `mecha-fly` `nmf_cpg.h` `NmfCpg` | Same phase biases, coupling weights, amplitude convergence |
| `get_joint_angles()` | `NmfCpg::GetCtrl()` | `neutral + amplitude * (trajectory(phase) - neutral)` |
| `get_adhesion_onoff()` | Boolean ctrl[42..47] | Stance/swing from CPG phase |
| `HybridTurningController` | `NmfCpg::SetDrive()` | Magnitude + turn mapped to L/R amplitude |
| `NeuroMechFly` sim loop | `mujoco-game` `Sim` class | MuJoCo wrapper with GLFW viewer |
| Proprioceptive obs | `proprio.h` | 95-channel encoder (joint angles, velocities, contacts, body vel) |

Physics parameters match flygym defaults: timestep 0.0001s, position actuators with gainprm=45, NeuroMechFly v2 MJCF model.

## Project structure

```
cpp/
  CMakeLists.txt          Build system (auto-detects MuJoCo, optional GLFW viewer)
  cmake/gen_implib.bat    Windows import library generator for pip MuJoCo
  src/
    main.cc               Standalone sim entry point
    brain.h               Embedded spiking brain module (on by default)
    brain_viewport.h      Brain point-cloud renderer for ImGui panel
    fly_state.h           Fly body state extraction (6 legs x 7 joints = 42 DOF)
    fly_tcp.h             Fly-specific TCP wire format for FWMC protocol
    proprio.h             95-channel proprioceptive encoder
  data/
    nmf_complete.xml      Pre-exported NeuroMechFly MJCF (with actuators)
    *.stl                 NeuroMechFly mesh files
mujoco-game/              Shared MuJoCo engine (sim.h, tcp.h, types.h)
mechabrain/               Spiking neural network engine (used by brain.h)
```

## Brain integration

flygame includes an embedded spiking brain (`-DNMFLY_BRAIN=ON`, default). When a `.brain` spec file is found next to the executable, the brain auto-enables — no flags needed.

**Controller modes:**

1. **Embedded brain** (default): WASD injects currents into descending neurons; spiking network drives motor output via `brain.h`
2. **TCP bridge** (`--brain host:port`): Brain runs in a separate process via `fly_tcp.h`
3. **Standalone CPG** (`-DNMFLY_BRAIN=OFF`): WASD drives the CPG directly, no brain

See `fly_tcp.h` for the wire protocol details.

## Requirements

- C++20 compiler, CMake 3.20+, MuJoCo 3.0+
- Optional: GLFW (fetched automatically by CMake)

## License

MIT License. See [LICENSE](LICENSE).

## Related

- [pyflygame](https://github.com/stanbot8/pyflygame): Python bridge and adapter framework
- [flygym](https://github.com/NeLy-EPFL/flygym) (NeLy-EPFL): Original NeuroMechFly framework
- [MuJoCo](https://mujoco.org/) (DeepMind): Physics engine
- [FWMC](https://github.com/stanbot8/fwmc): Spiking connectome simulator
