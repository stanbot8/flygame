# flygame

Real-time *Drosophila* simulation with a spiking brain. C++ on MuJoCo, ~230x faster than [flygym](https://github.com/NeLy-EPFL/flygym).

## Quick start

```bash
cmake -G "Visual Studio 17 2022" -S cpp -B build
cmake --build build --config Release
launch.bat
```

WASD to walk, Space to run, Shift to fly (experimental). The brain drives motor output through descending neurons, not scripted gaits.

## Packages

flygame bundles four independent packages:

```
flygame/
  cpp/src/         main app: viewer, brain module, proprioception
  mecha-fly/       locomotion controllers (CPG, hybrid, IK)
  mujoco-game/     MuJoCo physics wrapper + TCP protocol
  mechabrain/      spiking neural network engine (159k neurons, 6.7M synapses)
```

**mecha-fly**: Kuramoto CPG oscillator and hybrid controller (cubic spline + correction vectors). Ported from flygym's `PreprogrammedSteps` and `HybridTurningController`. Source of truth is `mujoco-zoo/mecha-fly`.

**mujoco-game**: Header-only MuJoCo wrapper. Handles the sim loop, GLFW viewer, and a simple TCP protocol (`MotorCommand` / `SensoryReading`) for remote brains.

**mechabrain**: Pure brain engine. Izhikevich neurons, STDP, gap junctions, homeostasis, temperature dynamics, parametric connectome generation from `.brain` spec files. Header-only C++23.

## Controller modes

| Mode | How | Description |
|------|-----|-------------|
| Embedded brain | *(default)* | Spiking network drives motor output in-process |
| TCP bridge | `--brain host:port` | Brain runs in a separate process |
| Standalone CPG | build with `-DNMFLY_BRAIN=OFF` | WASD drives CPG directly, no brain |
| Flight | hold Shift | Wing oscillation with MuJoCo fluid aero (experimental) |

## Performance

| Mode | Speed |
|------|-------|
| Headless | 7x realtime (~35k steps/sec) |
| With viewer | 5.7x realtime (~29k steps/sec) |
| flygym (Python) | 0.03x realtime |

## Build options

```bash
# Auto-detects MuJoCo from pip
cmake -S cpp -B build

# Specify MuJoCo location
cmake -S cpp -B build -DMUJOCO_DIR=/path/to/mujoco

# Disable brain (CPG-only mode)
cmake -S cpp -B build -DNMFLY_BRAIN=OFF
```

Requires C++20, CMake 3.20+, MuJoCo 3.0+. GLFW is fetched automatically.

On Windows with pip-installed MuJoCo, the build auto-generates an import library. If that fails:
```
cpp\cmake\gen_implib.bat "path\to\mujoco.dll" build
```

## What's ported from flygym

| flygym | flygame | Notes |
|--------|---------|-------|
| `PreprogrammedSteps` | `nmf_walk_spline.h` | Periodic cubic spline (C2 smooth) |
| `CPGNetwork` | `nmf_cpg.h` | Same phase biases, coupling weights |
| `HybridTurningController` | `nmf_hybrid.h` | Magnitude + turn mapped to L/R amplitude |
| Retraction/stumbling corrections | `NmfHybrid::UpdateCorrections()` | Lifts on low foot z, retracts on swing contact |
| Proprioceptive obs | `proprio.h` | 95-channel encoder (angles, velocities, contacts) |

Physics parameters match flygym: timestep 0.0001s, position actuators with gainprm=45, NeuroMechFly v2 MJCF.

## License

MIT. See [LICENSE](LICENSE).

## Related

- [pyflygame](https://github.com/stanbot8/pyflygame): Python bridge and adapter framework
- [flygym](https://github.com/NeLy-EPFL/flygym): Original NeuroMechFly framework (NeLy-EPFL)
- [FWMC](https://github.com/stanbot8/fwmc): CLI wrapper for spiking connectome experiments
- [MuJoCo](https://mujoco.org/): Physics engine (DeepMind)
