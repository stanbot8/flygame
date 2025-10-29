# Parabrain API Reference

All public types and functions reside in the `mechabrain` namespace. The project is header-only; include the relevant header to access each component.

> For bridge and experiment APIs, see [fwmc/docs/api_reference.md](../../fwmc/docs/api_reference.md).

---

## Core Data Structures

### NeuronArray
**Header**: `core/neuron_array.h`

Structure-of-arrays neuron storage. Each array index `i` refers to the same neuron. Fields include membrane potential `v` (mV), recovery variable `u`, synaptic input `i_syn`, external current `i_ext`, spike flag `spiked` (uint8), FlyWire root ID, 3D position `(x, y, z)`, cell type and region indices, neuromodulator concentrations `(dopamine, serotonin, octopamine)`, and `last_spike_time`.

- `void Resize(size_t count)`:Allocate all arrays to `count` neurons with default initial values (v=-65, u=-13, last_spike_time=-1e9).
- `void ClearSynapticInput()`:Zero all `i_syn` entries.
- `int CountSpikes() const`:Return the number of neurons with `spiked == 1`.

```cpp
NeuronArray neurons;
neurons.Resize(10000);
neurons.ClearSynapticInput();
int n = neurons.CountSpikes();
```

### SynapseTable
**Header**: `core/synapse_table.h`

CSR (Compressed Sparse Row) synapse storage. Sorted by pre-synaptic neuron for cache-friendly spike propagation. Fields: `row_ptr` (CSR index), `post` (post-synaptic indices), `weight`, `nt_type` (NTType enum).

- `size_t Size() const`:Return total synapse count.
- `static float Sign(uint8_t nt)`:Return -1.0 for GABA and Glutamate (GluCl in *Drosophila*), +1.0 for all others.
- `void BuildFromCOO(size_t num_neurons, pre, post, weight, nt)`:Sort COO data and build CSR index.
- `void PropagateSpikes(const uint8_t* spiked, float* i_syn, float weight_scale) const`:Deliver weighted synaptic current from all spiking pre-neurons to their post-synaptic targets.

```cpp
SynapseTable synapses;
synapses.BuildFromCOO(n, pre_vec, post_vec, weight_vec, nt_vec);
synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
```

### IzhikevichParams
**Header**: `core/izhikevich.h`

Parameters for the Izhikevich neuron model: `a` (recovery rate, default 0.02), `b` (recovery sensitivity, default 0.2), `c` (reset voltage, default -65 mV), `d` (after-spike reset increment, default 8), `v_thresh` (spike threshold, default 30 mV).

### LIFParams
**Header**: `core/izhikevich.h`

Parameters for the leaky integrate-and-fire model: `tau_ms` (membrane time constant, default 20), `v_rest` (resting potential, default -70 mV), `v_thresh` (spike threshold, default -55 mV), `v_reset` (reset voltage, default -70 mV), `r_membrane` (membrane resistance, default 10).

### STDPParams
**Header**: `core/stdp.h`

STDP rule parameters: `a_plus` (potentiation amplitude, default 0.01), `a_minus` (depression amplitude, default 0.012), `tau_plus` / `tau_minus` (time constants, default 20 ms each), `w_min` / `w_max` (weight bounds, default 0 / 10), `dopamine_gated` (enable DA modulation, default false), `da_scale` (DA modulation strength, default 5.0).

### ExperimentConfig
**Header**: `core/experiment_config.h`

Complete experiment specification: metadata (name, fly strain, date, notes), simulation parameters (dt_ms, duration_ms, weight_scale, metrics_interval, enable_stdp), bridge mode (0=open-loop, 1=shadow, 2=closed-loop), replacement thresholds (monitor, bridge, resync), calibration settings (interval, learning rate), stimulus protocol (ordered list of `StimulusEvent`), data paths (connectome_dir, recording_input, output_dir), and recording options (record_spikes, record_voltages, record_shadow_metrics, record_per_neuron_error, recording_interval).

### StimulusEvent
**Header**: `core/experiment_config.h`

A timed stimulus event: `start_ms`, `end_ms`, `intensity` (normalized [0,1]), `label` (string), and `target_neurons` (vector of neuron indices).

---

## Core Functions

### IzhikevichStep
**Header**: `core/izhikevich.h`

```cpp
void IzhikevichStep(NeuronArray& neurons, float dt_ms,
                    float sim_time_ms, const IzhikevichParams& p);
```

Advance all neurons by one timestep using the Izhikevich model with uniform parameters. Uses two half-step integration for stability. NaN/Inf guard resets divergent neurons. OpenMP parallelized for neuron counts > 10,000.

### IzhikevichStepHeterogeneous
**Header**: `core/cell_types.h`

```cpp
void IzhikevichStepHeterogeneous(NeuronArray& neurons, float dt_ms,
                                  float sim_time_ms,
                                  const CellTypeManager& types);
```

Same as `IzhikevichStep` but uses per-neuron parameters from a `CellTypeManager`. Required for parametric brains with multiple cell types.

### LIFStep
**Header**: `core/izhikevich.h`

```cpp
void LIFStep(NeuronArray& neurons, float dt_ms,
             float sim_time_ms, const LIFParams& p);
```

Advance all neurons using the leaky integrate-and-fire model. Faster than Izhikevich but captures fewer firing patterns. NaN/Inf guard and OpenMP parallelization included.

### STDPUpdate
**Header**: `core/stdp.h`

```cpp
void STDPUpdate(SynapseTable& synapses, const NeuronArray& neurons,
                float sim_time_ms, const STDPParams& p);
```

Update synaptic weights based on spike timing. For each synapse, compute timing difference between pre- and post-synaptic spikes and apply exponential potentiation or depression. If `p.dopamine_gated` is true, weight changes are scaled by the post-synaptic dopamine concentration. OpenMP parallelized over pre-synaptic neurons for >10k neurons (no write conflicts since CSR layout assigns each synapse to exactly one pre-neuron).

### NeuromodulatorUpdate
**Header**: `core/stdp.h`

```cpp
void NeuromodulatorUpdate(NeuronArray& neurons,
                          const SynapseTable& synapses, float dt_ms);
```

Decay existing neuromodulator concentrations and release from spiking neuromodulatory neurons. DAN neurons release dopamine to post-synaptic targets; fast-spiking interneurons release octopamine. All concentrations clamped to [0, 1].

### PropagateSpikes
**Header**: `core/synapse_table.h` (member of `SynapseTable`)

```cpp
void PropagateSpikes(const uint8_t* spiked, float* i_syn,
                     float weight_scale) const;
```

For each pre-synaptic neuron that spiked, traverse its outgoing CSR synapses and add `Sign(nt) * weight * weight_scale` to the post-synaptic `i_syn` accumulator. This is the computational hot loop. OpenMP parallelized over pre-synaptic neurons for >10k neurons, with `#pragma omp atomic` for i_syn accumulation.

### IzhikevichStepFast
**Header**: `core/izhikevich.h`

```cpp
void IzhikevichStepFast(NeuronArray& neurons, float dt_ms,
                        float sim_time_ms, const IzhikevichParams& p);
```

Dispatch function that selects AVX2-vectorized (`IzhikevichStepAVX2`, 8 neurons per iteration using `__m256` intrinsics with FMA) or scalar `IzhikevichStep` based on compile-time detection. ~87M neurons/sec on AVX2 hardware.

### ProprioMap
**Header**: `core/proprioception.h`

Assigns VNC neurons to proprioceptive sensory channels and injects body state as excitatory currents.

- `void Init(const NeuronArray& neurons, uint8_t vnc_region, float midline_x)`: Auto-assign first 30% of VNC neurons across 53 sensory channels (42 joint angle, 6 contact, 3 body velocity, 2 haltere L/R).
- `void Inject(NeuronArray& neurons, const ProprioState& state, const ProprioConfig& cfg) const`: Convert body state to currents and inject into assigned neurons. Joint angles use sigmoid activation; contacts are strong binary signals; haltere feedback is asymmetric L/R based on yaw rate.

Supporting types: `ProprioConfig` (gain parameters), `ProprioState` (42 joint angles, 42 velocities, 6 contacts, 3 body velocity), `ReadProprioFromMuJoCo<MjModel, MjData>()` (template function extracting state from MuJoCo qpos/qvel/contacts).

### CPGOscillator
**Header**: `core/cpg.h`

Central pattern generator injecting oscillatory current into VNC motor neurons for spontaneous rhythmic locomotion.

- `void Init(const NeuronArray& neurons, uint8_t vnc_region, float midline_x, float sensory_fraction)`: Split VNC motor neurons (after sensory fraction) into two anti-phase groups by x-coordinate.
- `void Step(NeuronArray& neurons, float dt_ms, float descending_drive)`: Advance phase at `frequency_hz` (default 8 Hz), inject `tonic_drive + sin(phase) * amplitude * drive_scale` to group A and anti-phase to group B. `drive_scale` smoothly tracks `descending_drive` with a 50ms time constant.

---

## Connectome I/O

### ConnectomeLoader
**Header**: `core/connectome_loader.h`

Static methods for loading binary connectome files.

- `static Result<size_t> LoadNeurons(const string& path, NeuronArray& neurons)`:Read `neurons.bin`. Returns neuron count or error. Validates count bounds (max 10M neurons).
- `static Result<size_t> LoadSynapses(const string& path, size_t n_neurons, SynapseTable& table)`:Read `synapses.bin`, validate index bounds, and build CSR. Returns synapse count or error.

```cpp
NeuronArray neurons;
SynapseTable synapses;
auto nr = ConnectomeLoader::LoadNeurons("data/neurons.bin", neurons);
auto sr = ConnectomeLoader::LoadSynapses("data/synapses.bin", neurons.n, synapses);
```

### ConnectomeExport
**Header**: `core/connectome_export.h`

Static methods for writing binary connectome files.

- `static Result<size_t> ExportNeurons(const string& path, const NeuronArray& neurons)`:Write `neurons.bin` from a NeuronArray.
- `static Result<size_t> ExportSynapses(const string& path, const SynapseTable& table)`:Reconstruct COO from CSR and write `synapses.bin`.

---

## Parametric Generation

### ParametricGenerator
**Header**: `core/parametric_gen.h`

Generates NeuronArray and SynapseTable from a BrainSpec.

- `uint32_t Generate(const BrainSpec& spec, NeuronArray& neurons, SynapseTable& synapses, CellTypeManager& types)`:Create all neurons, assign cell types and regions, generate intra-region and inter-region synapses, build CSR, assign per-neuron Izhikevich parameters. Returns total neuron count.
- `vector<RegionRange> region_ranges`:After generation, maps region names to contiguous neuron index ranges.

```cpp
BrainSpec spec;
spec.name = "test";
spec.regions.push_back({"AL", 500, 0.1f, kACh, {{CellType::kPN_excitatory, 0.5f}}});

ParametricGenerator gen;
NeuronArray neurons;
SynapseTable synapses;
CellTypeManager types;
gen.Generate(spec, neurons, synapses, types);
```

### BrainSpecLoader
**Header**: `core/brain_spec_loader.h`

Parses `.brain` specification files into a `BrainSpec` struct. Supports region definitions, projections, stimuli, background noise (`background_mean`, `background_std`), and global parameters.

- `static Result<BrainSpec> Load(const string& path)`:Parse a `.brain` file and return a BrainSpec.

### ParamSweep
**Header**: `core/param_sweep.h`

Parameter sweep engine for auto-tuning Izhikevich parameters.

- `void GridSweep(CellType target, const NeuronArray& base, const SynapseTable& synapses, ScoreFn score_fn)`:Exhaustive grid search over (a, b, c, d) space.
- `void RandomSweep(...)`:Uniform random sampling of parameter space.
- `void Refine(CellType target, ..., int iterations, float step_size)`:Stochastic hill-climbing from the best point found.
- `IzhikevichParams BestParams() const`:Return the highest-scoring parameter set.

Built-in scoring functions (in `fwmc::scoring` namespace):
- `ScoreFn TargetFiringRate(float target_hz, float dt_ms)`:Score inversely proportional to firing rate error.
- `ScoreFn ActivityInRange(float min_fraction, float max_fraction)`:Score based on fraction of active neurons being within bounds.
- `ScoreFn RealisticCV(float target_cv)`:Score based on coefficient of variation of spike timing.

### ParametricSync
**Header**: `core/parametric_sync.h`

Adaptive sync engine for tuning a parametric model to match a reference brain.

- `void Init(size_t n_neurons, size_t n_synapses)`:Allocate per-neuron and per-synapse state.
- `void Step(NeuronArray& model, SynapseTable& synapses, const NeuronArray& ref, CellTypeManager& types)`:Run one sync step: corrective current (fast), weight error accumulation (medium), parameter nudges (slow).
- `bool HasConverged() const`:Return true if the target fraction of neurons have converged.
- `const SyncSnapshot& Latest() const`:Get the most recent sync metrics.

---

## Metrics and Plasticity

### RegionMetrics
**Header**: `core/region_metrics.h`

Per-region activity tracking during parametric brain simulation.

- `void Init(const ParametricGenerator& gen)`:Initialize with region ranges from the generator.
- `void Record(const NeuronArray& neurons, float sim_time_ms, float dt_ms, int window_steps)`:Record a snapshot of per-region spike counts, firing rates, fraction active, and mean voltage.
- `void LogLatest() const`:Log the most recent snapshot.
- `void LogSummary() const`:Log cumulative statistics (total spikes, peak rate, mean fraction active) for all regions.

### ApplyStimuli (free function)
**Header**: `core/region_metrics.h`

```cpp
void ApplyStimuli(const vector<StimulusSpec>& stimuli,
                  const vector<RegionRange>& regions,
                  NeuronArray& neurons, float sim_time_ms, uint32_t seed);
```

Apply parametric stimulus specifications to neurons based on region ranges. Only active stimuli (within their time window) inject current.

### StructuralPlasticity
**Header**: `core/structural_plasticity.h`

Synapse pruning and sprouting.

- `size_t PruneWeak(SynapseTable& syn)`:Zero out synapses with weight below threshold. Returns count pruned.
- `size_t SproutNew(SynapseTable& syn, NeuronArray& neurons, mt19937& rng)`:Create new excitatory synapses between co-active neurons. Rebuilds CSR. Returns count sprouted.
- `void Update(SynapseTable& syn, NeuronArray& neurons, int step, mt19937& rng)`:Called each step; only acts at update_interval boundaries.

Configuration via `StructuralPlasticity::Config`: `prune_threshold` (default 0.05), `sprout_rate` (default 0.001), `update_interval` (default 5000 steps), `max_synapses_per_neuron` (default 100).

### GapJunctionTable
**Header**: `core/gap_junctions.h`

Electrical gap junction storage. Bidirectional current `I = g * (Vb - Va)` between connected neurons.

- `void AddJunction(uint32_t a, uint32_t b, float g)`: Add a single gap junction with conductance `g`.
- `void PropagateGapCurrents(NeuronArray& neurons) const`: Inject gap currents into `i_ext`. OpenMP parallelized with atomic accumulation for >10k junctions.
- `void BuildFromRegion(const NeuronArray& neurons, uint8_t region, float density, float g_default, uint32_t seed)`: Connect all neuron pairs within a region with probability `density`.

### UpdateSTP
**Header**: `core/short_term_plasticity.h`

Tsodyks-Markram short-term plasticity update for SynapseTable.

- `void UpdateSTP(SynapseTable& synapses, const NeuronArray& neurons, float dt_ms)`: Relax u and x toward resting values, then apply spike updates for firing pre-neurons. Requires `SynapseTable::InitSTP()`.
- `void ResetSTP(SynapseTable& synapses)`: Restore resting state (u=U_se, x=1).
- `STPParams STPFacilitating()`, `STPDepressing()`, `STPCombined()`: Preset factories.
- `float MeanSTPUtilization(const SynapseTable&)`, `float MeanSTPResources(const SynapseTable&)`: Diagnostics.

### NWBExporter
**Header**: `core/nwb_export.h`

Lightweight NWB-compatible exporter producing CSV spike/voltage files and JSON session metadata.

- `bool BeginSession(const string& dir, const string& description, const NeuronArray& neurons)`: Create output directory, open CSVs, snapshot neuron metadata.
- `void SetVoltageSubset(const vector<uint32_t>& neuron_indices)`: Configure which neurons get voltage traces.
- `void RecordTimestep(float time_ms, const NeuronArray& neurons)`: Record spikes and voltages for one step.
- `void AddStimulus(float start_ms, float stop_ms, const string& name, const string& desc)`: Register stimulus for metadata.
- `void EndSession()`: Flush CSVs, write `session.nwb.json` with NWB 2.7 schema.

### CellTypeManager
**Header**: `core/cell_types.h`

Manages per-neuron Izhikevich parameters based on cell type assignments.

- `void AssignFromTypes(const NeuronArray& neurons)`:Populate `neuron_params` from each neuron's type field, applying overrides where set.
- `const IzhikevichParams& Get(size_t idx) const`:Get parameters for a single neuron.
- `void SetOverride(CellType ct, const IzhikevichParams& p)`:Override default parameters for a cell type. Call `AssignFromTypes()` after to apply.

---

