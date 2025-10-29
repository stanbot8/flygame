# mechabrain

Header-only C++23 spiking neural network engine for whole-brain scale simulation. Designed around the *Drosophila melanogaster* connectome (~140k neurons, ~50M synapses) but general enough for any spiking network.

All mechanistic brain code lives here: neuron dynamics, synapses, plasticity, sensorimotor integration, parametric generation, neural twinning bridge, GPU kernels, optogenetics, and structured experiments.

No external dependencies. Optional OpenMP and CUDA acceleration.

## Core modules

### Neuron dynamics

| File | Description |
|------|-------------|
| [neuron_array.h](core/neuron_array.h) | SoA (structure-of-arrays) neuron state: membrane voltage, recovery variable, synaptic current, spike flags, neuromodulator concentrations |
| [izhikevich.h](core/izhikevich.h) | Izhikevich model (2 coupled ODEs, 20+ firing patterns) and leaky integrate-and-fire. AVX2 SIMD path, half-step integration, NaN recovery, OpenMP parallel |
| [cell_types.h](core/cell_types.h) | Per-neuron heterogeneous dynamics. Biologically tuned parameters for KC, PN, LN, MBON, DAN, ORN, CX types (EPG, PEN, PEG, PFL, Delta7), mouse L4/L5, zebrafish tectal |
| [cell_type_defs.h](core/cell_type_defs.h) | Cell type parameter library: 16+ Izhikevich parameter sets from literature |
| [compartmental_neuron.h](core/compartmental_neuron.h) | Multi-compartment neuron model |

### Connectivity

| File | Description |
|------|-------------|
| [synapse_table.h](core/synapse_table.h) | CSR (compressed sparse row) synapse graph. 6 NT types (ACh, GABA, Glut, DA, 5HT, OA), stochastic vesicle release, Tsodyks-Markram STP. OpenMP spike propagation with atomic accumulation |
| [connectome_loader.h](core/connectome_loader.h) | Binary connectome reader with `std::expected` error handling |
| [connectome_export.h](core/connectome_export.h) | Export parametric brains to binary format |
| [connectome_stats.h](core/connectome_stats.h) | Degree distributions, NT ratios, weight statistics, integrity checks |

### Plasticity and synaptic dynamics

| File | Description |
|------|-------------|
| [stdp.h](core/stdp.h) | Spike-timing-dependent plasticity with dopamine-gated modulation (Izhikevich 2007). Exponential timing windows, bounded weights |
| [short_term_plasticity.h](core/short_term_plasticity.h) | Tsodyks-Markram STP update, preset factories (facilitating, depressing, combined), diagnostics |
| [structural_plasticity.h](core/structural_plasticity.h) | Synapse pruning (weak weights) and sprouting (correlated co-firing) |
| [calcium_plasticity.h](core/calcium_plasticity.h) | Calcium-dependent plasticity |
| [inhibitory_plasticity.h](core/inhibitory_plasticity.h) | GABAergic weight adaptation via synaptic scaling |
| [gap_junctions.h](core/gap_junctions.h) | Electrical gap junctions: bidirectional current, region-based construction, OpenMP parallel |
| [nmda.h](core/nmda.h) | NMDA receptor dynamics with voltage-dependent Mg2+ block and calcium influx |
| [neuromodulator_effects.h](core/neuromodulator_effects.h) | DA/5HT/OA modulation of synaptic transmission |
| [spike_frequency_adaptation.h](core/spike_frequency_adaptation.h) | Calcium-dependent SFA for realistic spiking diversity |
| [intrinsic_homeostasis.h](core/intrinsic_homeostasis.h) | Target firing rate maintenance via excitability adaptation |

### Parametric generation

| File | Description |
|------|-------------|
| [parametric_gen.h](core/parametric_gen.h) | Generate connectomes from high-level brain specs: regions, cell type distributions, NT ratios, inter-region projections |
| [brain_spec_loader.h](core/brain_spec_loader.h) | Parser for `.brain` config files |
| [param_sweep.h](core/param_sweep.h) | Grid search + stochastic hill-climbing for neuron parameter tuning |
| [parametric_sync.h](core/parametric_sync.h) | Three-timescale sync engine: corrective current (fast), weight SGD (medium), parameter nudge (slow) |
| [experiment_optimizer.h](core/experiment_optimizer.h) | General-purpose random search with refinement and multi-seed robustness |
| [region_metrics.h](core/region_metrics.h) | Per-region spike counts, firing rates, fraction active, mean voltage |

### Sensorimotor

| File | Description |
|------|-------------|
| [motor_output.h](core/motor_output.h) | Maps descending neuron spike rates to fictive locomotion commands (forward velocity, turning, approach/avoid) |
| [proprioception.h](core/proprioception.h) | Maps MuJoCo body state (joint angles, contacts, velocity, haltere rotation) to VNC sensory neuron currents |
| [cpg.h](core/cpg.h) | Central pattern generator: anti-phase oscillatory drive to VNC motor neurons for tripod gait locomotion |
| [digital_compensator.h](core/digital_compensator.h) | Reusable digital twin compensation module for neural prosthesis experiments |

### Analysis and profiling

| File | Description |
|------|-------------|
| [spike_analysis.h](core/spike_analysis.h) | ISI statistics, burst detection, oscillation detection, Fano factor |
| [behavioral_fingerprint.h](core/behavioral_fingerprint.h) | Learning index, discrimination, approach/avoid readouts |
| [rate_monitor.h](core/rate_monitor.h) | Firing rate validation against literature ranges |
| [scoped_timer.h](core/scoped_timer.h) | Per-function performance timing |
| [memory_tracker.h](core/memory_tracker.h) | Allocation tracking |

### I/O and infrastructure

| File | Description |
|------|-------------|
| [experiment_config.h](core/experiment_config.h) | Experiment parameters, cell types, regions, stimulus events |
| [experiment_protocol.h](core/experiment_protocol.h) | Multi-phase protocols with transition conditions |
| [config_loader.h](core/config_loader.h) | Key-value config file parser |
| [checkpoint.h](core/checkpoint.h) | Binary save/load of full simulation state with extension blobs |
| [recorder.h](core/recorder.h) | Binary and CSV recording of spikes, voltages, drift metrics |
| [nwb_export.h](core/nwb_export.h) | NWB-compatible export: spike CSV, voltage CSV, session metadata JSON |
| [error.h](core/error.h) | `Result<T> = std::expected<T, Error>` with error codes |
| [log.h](core/log.h) | Minimal `std::format`-based logger with timestamp and level tags |

## Bridge: neural twinning pipeline

The [bridge/](bridge/) subdirectory implements the bidirectional interface for neural twinning: reading biological activity, running a digital twin, and writing optogenetic commands.

| File | Description |
|------|-------------|
| [protocol.h](bridge/protocol.h) | Binary TCP wire protocol (v1) for brain-body communication |
| [bridge_channel.h](bridge/bridge_channel.h) | Abstract read/write channel interfaces, simulated channels for testing |
| [spike_decoder.h](bridge/spike_decoder.h) | Calcium-to-spike deconvolution with adaptive thresholding |
| [shadow_tracker.h](bridge/shadow_tracker.h) | Digital twin shadowing with prediction error and correlation tracking |
| [neuron_replacer.h](bridge/neuron_replacer.h) | Per-neuron state machine (BIOLOGICAL/MONITORED/BRIDGED/REPLACED) |
| [calibrator.h](bridge/calibrator.h) | Perturbation-based gradient-free weight calibration |
| [validation.h](bridge/validation.h) | Single-neuron (F1, van Rossum) and population spike train metrics |
| [optogenetic_writer.h](bridge/optogenetic_writer.h) | Holographic two-photon SLM stimulation with thermal safety |
| [opsin_model.h](bridge/opsin_model.h) | ChR2, ChRmine, stGtACR2 three-state channel kinetics |
| [light_model.h](bridge/light_model.h) | Beer-Lambert tissue light propagation |
| [twin_bridge.h](bridge/twin_bridge.h) | Main orchestrator: state machine with hysteresis and adaptive boundary expansion |
| [tcp_bridge.h](bridge/tcp_bridge.h) | TCP socket interface for remote brain process |

## CUDA GPU acceleration

The [cuda/](cuda/) subdirectory provides GPU kernels for the hot path:

| File | Description |
|------|-------------|
| [gpu_manager.h](cuda/gpu_manager.h) | Device memory lifecycle, upload/download, async streams |
| [izhikevich_kernel.cu](cuda/izhikevich_kernel.cu) | 1-thread-per-neuron Izhikevich integration |
| [spike_propagation_kernel.cu](cuda/spike_propagation_kernel.cu) | CSR row-parallel spike routing with atomicAdd |
| [stdp_kernel.cu](cuda/stdp_kernel.cu) | Dopamine-gated STDP per synapse |

## Optogenetics

The [optogenetics/](optogenetics/) subdirectory provides standalone optogenetic simulation tools:

| File | Description |
|------|-------------|
| [optogenetics.h](optogenetics/optogenetics.h) | Opsin kinetics, light models, stimulation patterns |
| [optimizer.h](optogenetics/optimizer.h) | Compute optimal stimulation patterns for desired neuron sets |
| [opto_io.h](optogenetics/opto_io.h) | SLM and galvo mirror hardware abstraction |

## Experiments

The [experiments/](experiments/) subdirectory contains 9 self-contained neuroscience demonstrations:

| File | Circuit | Key result |
|------|---------|------------|
| [01_conditioning.h](experiments/01_conditioning.h) | ORN-PN-KC-MBON olfactory conditioning | LI=4.9 via dopamine-gated STDP |
| [02_visual_escape.h](experiments/02_visual_escape.h) | Visual lobe looming escape | Latency 88.5ms (lit: 30-150ms) |
| [03_navigation.h](experiments/03_navigation.h) | CX ring attractor heading | Heading error 2.5deg, R2=0.998 |
| [04_whisker.h](experiments/04_whisker.h) | Mouse barrel cortex L4-L5 | 2380 neurons, L23/L4 ratio=0.56 |
| [05_prey_capture.h](experiments/05_prey_capture.h) | Zebrafish tectal prey capture | Motor onset 153.4ms |
| [06_courtship.h](experiments/06_courtship.h) | Drosophila courtship song | IPI=44.5ms (lit: 30-45ms) |
| [07_twinning.h](experiments/07_twinning.h) | Full neural twinning demo | Behavioral continuity=1.000 |
| [08_ablation.h](experiments/08_ablation.h) | Progressive KC silencing | Graceful degradation score=0.86 |
| [09_compensated_ablation.h](experiments/09_compensated_ablation.h) | Digital twin prosthesis | +0.29 compensation benefit at 90% ablation |

## Tissue volume

The [tissue/](tissue/) subdirectory provides a procedural 3D brain volume with Wilson-Cowan neural field dynamics, voxel-based neuromodulator diffusion (DA/5HT/OA with literature-grounded coefficients), and a multi-level LOD manager.

## Tests

```bash
# From the fwmc build directory (mechabrain tests are built by fwmc's CMakeLists):
ctest --build-config Release
# test_core, test_parametric, test_tissue, test_bridge -- all 4 suites pass
```

## Documentation

- [docs/methods.md](docs/methods.md) -- Neuron models, synapses, plasticity, connectome, parametric generation
- [docs/api_reference.md](docs/api_reference.md) -- API reference for all mechabrain types and functions

## Design principles

- **Header-only**: every module is a single `.h` file. Include what you need, no link step.
- **Flat arrays**: SoA layout for cache-friendly iteration and easy SIMD/GPU porting. No virtual dispatch on hot paths.
- **No framework dependencies**: standard C++23 only. OpenMP and CUDA are optional.
- **Biologically grounded**: neuron parameters, NT classifications, and circuit motifs drawn from FlyWire data (Dorkenwald et al. 2024) and published electrophysiology.
