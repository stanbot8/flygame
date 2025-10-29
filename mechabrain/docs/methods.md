# Parabrain Methods

> For bridge/twinning methods, see [fwmc/docs/methods.md](../../fwmc/docs/methods.md).

## 1. Neuron Model

### 1.1 Izhikevich Model

Individual neuron dynamics are simulated using the Izhikevich (2003) model, a two-dimensional system of ordinary differential equations that captures the qualitative features of biological spiking neurons while remaining computationally tractable for large-scale simulation:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

with the after-spike reset rule: if $v \geq v_{\text{thresh}}$ (default 30 mV), then $v \leftarrow c$ and $u \leftarrow u + d$. Here $v$ represents the membrane potential (mV), $u$ is a recovery variable that accounts for the activation of K+ ionic currents and inactivation of Na+ currents, and $I = I_{\text{syn}} + I_{\text{ext}}$ is the total input current comprising synaptic and external stimulus components.

The four parameters $(a, b, c, d)$ are assigned per cell type based on known *Drosophila* neuronal physiology (Table 1). This parameterization enables heterogeneous dynamics within the same network, where Kenyon cells exhibit regular spiking, local interneurons produce fast spiking patterns, and projection neurons display intrinsic bursting, consistent with electrophysiological recordings from the fly brain (Turner et al., 2008).

**Table 1.** Izhikevich parameters for *Drosophila* cell types.

| Cell Type | Label | *a* | *b* | *c* (mV) | *d* | Firing Pattern |
|-----------|-------|-----|-----|-----------|-----|----------------|
| Kenyon Cell | KC | 0.02 | 0.2 | -65 | 8 | Regular spiking |
| Projection Neuron (exc.) | PN | 0.02 | 0.2 | -50 | 2 | Bursting |
| Projection Neuron (inh.) | PN_inh | 0.1 | 0.2 | -65 | 2 | Fast spiking |
| Local Interneuron | LN | 0.1 | 0.2 | -65 | 2 | Fast spiking |
| MBON (cholinergic) | MBON_ACh | 0.02 | 0.2 | -65 | 8 | Regular spiking |
| MBON (GABAergic) | MBON_GABA | 0.02 | 0.2 | -65 | 8 | Regular spiking |
| MBON (glutamatergic) | MBON_Glut | 0.02 | 0.2 | -65 | 8 | Regular spiking |
| DAN (PPL1) | DAN_PPL1 | 0.02 | 0.25 | -65 | 6 | Regular spiking (tonic) |
| DAN (PAM) | DAN_PAM | 0.02 | 0.25 | -65 | 6 | Regular spiking (tonic) |
| ORN | ORN | 0.02 | 0.2 | -65 | 8 | Regular spiking |
| Fast Spiking | FS | 0.1 | 0.2 | -65 | 2 | Fast spiking |
| Bursting | Burst | 0.02 | 0.2 | -50 | 2 | Bursting |

### 1.2 Numerical Integration

We employ a two half-step integration scheme for numerical stability, following the recommendation of Izhikevich (2003):

```
v += 0.5 * dt * (0.04 * v^2 + 5 * v + 140 - u + I)
v += 0.5 * dt * (0.04 * v^2 + 5 * v + 140 - u + I)
u += dt * a * (b * v - u)
```

The default timestep is $\Delta t = 0.1$ ms. This split-step method reduces the numerical error inherent in the quadratic voltage equation, permitting stable integration at timesteps up to an order of magnitude larger than naive Euler integration.

### 1.3 Divergence Recovery

A NaN/Inf guard mechanism resets divergent neurons to physiologically valid states. If $v$ becomes non-finite, it is reset to $c$; if $u$ becomes non-finite, it is reset to $b \cdot c$. This prevents individual neuron instabilities from propagating through the network via synaptic currents.

### 1.4 Leaky Integrate-and-Fire Alternative

For large-scale connectivity studies where individual neuron dynamics are less critical, an alternative leaky integrate-and-fire (LIF) model is available:

$$\frac{dv}{dt} = \frac{-(v - v_{\text{rest}}) + R_m \cdot I}{\tau_m}$$

with parameters $\tau_m = 20$ ms, $v_{\text{rest}} = -70$ mV, $v_{\text{thresh}} = -55$ mV, $v_{\text{reset}} = -70$ mV, and $R_m = 10$ M$\Omega$. The LIF model requires a single ODE per neuron, offering approximately twice the throughput of the Izhikevich model on the same hardware.

### 1.5 Parallelization

Neuron updates are embarrassingly parallel (each neuron is independent within a single timestep). OpenMP parallelization is applied with static scheduling when the neuron count exceeds 10,000. CUDA GPU kernels port the same integration code to massively parallel execution, with one CUDA thread per neuron and 256 threads per block.


## 2. Synapse Model

### 2.1 Sparse Storage

Synaptic connectivity is stored in Compressed Sparse Row (CSR) format, sorted by pre-synaptic neuron index. For the full FlyWire connectome (~50M synapses), this representation requires approximately 450 MB of memory (4 bytes post-synaptic index + 4 bytes weight + 1 byte neurotransmitter type per synapse). The CSR layout provides cache-friendly sequential memory access during spike propagation: when a pre-synaptic neuron fires, its outgoing synapses occupy a contiguous range `[row_ptr[i], row_ptr[i+1])` in memory.

CSR construction proceeds from an unsorted coordinate (COO) representation. Synapses are first sorted by pre-synaptic index, then the CSR row pointer array is computed by cumulative counting.

### 2.2 Neurotransmitter-Specific Signaling

Each synapse carries a neurotransmitter (NT) type annotation derived from electron microscopy predictions (Eckstein et al., 2024). The NT type determines the sign of synaptic transmission:

| NT Type | Code | Sign | Role |
|---------|------|------|------|
| Acetylcholine (ACh) | 0 | +1 | Excitatory |
| GABA | 1 | -1 | Inhibitory |
| Glutamate (Glut) | 2 | -1 | Inhibitory in *Drosophila* (GluCl receptors) |
| Dopamine (DA) | 3 | +1 | Modulatory |
| Serotonin (5-HT) | 4 | +1 | Modulatory |
| Octopamine (OA) | 5 | +1 | Modulatory |

Synaptic current delivered to post-synaptic neuron $j$ upon a spike from pre-synaptic neuron $i$ is computed as:

$$I_{\text{syn},j} \mathrel{+}= \text{Sign}(\text{NT}_s) \cdot w_s \cdot w_{\text{scale}}$$

where $w_s$ is the synapse weight, $w_{\text{scale}}$ is a global gain parameter (default 1.0), and $\text{Sign}(\text{NT}_s)$ maps the neurotransmitter type to +1 or -1 as specified above.

### 2.3 Spike Propagation

Spike propagation is the computational hot loop. For each pre-synaptic neuron that spiked in the current timestep, all outgoing synapses are traversed and weighted current is delivered to post-synaptic targets. The CSR layout ensures sequential memory access patterns. On the GPU, spike propagation is parallelized with one thread per pre-synaptic neuron, using atomic additions for the post-synaptic current accumulation.


## 3. Plasticity

### 3.1 Spike-Timing-Dependent Plasticity

Synaptic plasticity follows the exponential STDP rule (Bi & Poo, 1998). For a synapse from neuron $i$ to neuron $j$, the weight change depends on the relative timing $\Delta t$ of pre- and post-synaptic spikes:

$$\Delta w = \begin{cases}
A^+ \exp(-\Delta t / \tau^+) & \text{if pre fires before post (potentiation)} \\
-A^- \exp(\Delta t / \tau^-) & \text{if post fires before pre (depression)}
\end{cases}$$

with default parameters $A^+ = 0.01$, $A^- = 0.012$, $\tau^+ = \tau^- = 20$ ms. STDP windows are truncated at $5\tau$ for computational efficiency. Weights are bounded to $[w_{\min}, w_{\max}] = [0, 10]$.

The slight asymmetry ($A^- > A^+$) produces a net depression bias, which prevents runaway excitation in recurrent networks and promotes sparse representations in the mushroom body, consistent with experimental observations (Honegger et al., 2011).

### 3.2 Dopamine-Gated Reward-Modulated STDP

Following the framework of Izhikevich (2007), STDP weight changes are modulated by local dopamine concentration:

$$\Delta w_{\text{eff}} = \Delta w \cdot (1 + \alpha_{\text{DA}} \cdot [\text{DA}]_j)$$

where $[\text{DA}]_j \in [0, 1]$ is the dopamine concentration at the post-synaptic neuron and $\alpha_{\text{DA}} = 5.0$ is the modulation strength. This implements a three-factor learning rule where the eligibility trace (spike-timing relationship) is gated by a reward signal (dopamine), solving the distal reward problem in the mushroom body circuit. High dopamine at a post-synaptic target enhances potentiation, enabling associative olfactory learning as described by Aso & Rubin (2016).

### 3.3 Neuromodulator Dynamics

Neuromodulator concentrations are tracked per neuron and evolve according to release-and-decay dynamics:

- **Dopamine**: Released by DAN neurons (PPL1, PAM cell types) upon spiking, with a release amplitude of 0.2 per spike delivered to all post-synaptic targets. Autoreceptor feedback at 50% release rate provides self-regulatory dampening. Exponential decay at rate 0.005 per ms.
- **Serotonin**: Decay rate 0.002 per ms. Available for future state-dependent modulation.
- **Octopamine**: Released by fast-spiking interneurons (amplitude 0.1) as an arousal signal, analogous to norepinephrine in vertebrates. Decay rate 0.003 per ms.

All concentrations are clamped to $[0, 1]$.

### 3.4 Structural Plasticity

Structural plasticity operates at longer timescales (default interval: every 5000 steps) through two mechanisms:

**Pruning**: Synapses with absolute weight below a threshold (default 0.05) are zeroed. Weight-zeroed synapses remain in the CSR topology but contribute no current during spike propagation, effectively removing them from the functional circuit.

**Sprouting**: New excitatory (ACh) synapses are created between co-active neurons (both spiked in the current step) with probability determined by a sprout rate parameter (default 0.001). New synapses are initialized at the pruning threshold weight, requiring STDP-driven potentiation to become functionally significant. A per-neuron cap (default 100 outgoing synapses) prevents excessive connectivity. Sprouting requires a CSR rebuild from the modified COO representation.

### 3.5 Supervised Calibration

For per-fly calibration of the digital twin, a perturbation-based gradient-free weight optimization method accumulates prediction error between digital and biological spike patterns. For each synapse where the pre-synaptic neuron spiked, the prediction error at the post-synaptic neuron is attributed to that synapse. Weight updates follow momentum SGD with L2 regularization:

$$v_s \leftarrow \mu v_s - \eta \nabla_s$$
$$w_s \leftarrow w_s + v_s - \lambda w_s$$

with default parameters $\eta = 0.001$, $\mu = 0.9$, $\lambda = 10^{-5}$. Weights are clamped to $[0, 20]$. Updates are applied periodically (default every 10,000 steps) after error accumulation across many timesteps.


## 4. Connectome

### 4.1 Data Source

The connectome is derived from the FlyWire whole-brain reconstruction of an adult female *Drosophila melanogaster* (Dorkenwald et al., 2024), comprising approximately 140,000 neurons and 50 million synapses. Neurotransmitter type predictions for each neuron are obtained from the classifier of Eckstein et al. (2024), which achieves >90% accuracy across the six major NT classes in *Drosophila*.

### 4.2 Region Segmentation

Neurons are assigned to neuropil regions based on FlyWire annotations:

| Region | Abbreviation | Description |
|--------|-------------|-------------|
| Antennal Lobe | AL | Primary olfactory processing |
| Mushroom Body | MB | Associative learning center |
| Lateral Horn | LH | Innate olfactory responses |
| Central Complex | CX | Navigation and motor planning |
| Optic Lobe | OL | Visual processing |
| Subesophageal Zone | SEZ | Gustatory and mechanosensory |

### 4.3 Binary Format

Connectome data is stored in a compact binary format for fast loading:

**neurons.bin**: `[count:u32] [root_id:u64, x:f32, y:f32, z:f32, type:u8] x count`

Each neuron record is 21 bytes: an 8-byte FlyWire root ID, three 4-byte spatial coordinates (in nanometers), and a 1-byte cell type index.

**synapses.bin**: `[count:u32] [pre:u32, post:u32, weight:f32, nt:u8] x count`

Each synapse record is 13 bytes: two 4-byte neuron indices, a 4-byte weight, and a 1-byte NT type. Synapse files are loaded into COO format and then converted to CSR for simulation. Integrity checks validate index bounds, detect self-loops, NaN weights, and isolated neurons.

### 4.4 Connectome Import

Raw connectome data is imported from the FlyWire CAVE API using a Python script (`import_connectome.py`) with optional region filtering and neuron count limits. A test mode generates synthetic circuits without requiring CAVE access.


## 7. Parametric Brain Generation

### 7.1 Brain Specification Format

Synthetic connectomes are generated from declarative `.brain` specification files that define brain regions, inter-region projections, and timed stimulus patterns. The format uses `key = value` syntax with hierarchical prefixes:

```
name = mushroom_body_model
seed = 42
weight_mean = 1.0
weight_std = 0.3

region.0.name = antennal_lobe
region.0.n_neurons = 500
region.0.density = 0.12
region.0.types = PN:0.4 LN:0.6
region.0.nt_dist = ACh:0.4 GABA:0.6

projection.0.from = antennal_lobe
projection.0.to = mushroom_body
projection.0.density = 0.01
projection.0.nt = ACh
projection.0.weight_mean = 1.5
projection.0.weight_std = 0.4

stimulus.0.label = odor_presentation
stimulus.0.region = antennal_lobe
stimulus.0.start = 500
stimulus.0.end = 1500
stimulus.0.intensity = 8.0
stimulus.0.fraction = 0.3
```

### 7.2 Region Definitions

Each region specifies a neuron count, internal connection density (probability of intra-region synapse), cell type distribution (as fractional proportions summing to 1.0), and neurotransmitter distribution for internal synapses. Neurons within a region occupy a contiguous index range in the global neuron array. Cell types are assigned proportionally, with the last type absorbing rounding remainder.

### 7.3 Inter-Region Projections

Projections define long-range connections between named regions. Each projection specifies connection probability (density), neurotransmitter type, and weight distribution parameters (mean and standard deviation of a normal distribution, clamped to a minimum of 0.01). Self-loops are skipped for internal connections but permitted in inter-region projections.

### 7.3.1 Geometric Skip Sampling

For large sparse regions (>100,000 potential pairs at <10% density), connection generation uses geometric skip sampling instead of O(n²) Bernoulli trials. The next edge is sampled by computing:

$$\text{skip} = \lfloor \log(U) / \log(1 - p) \rfloor$$

where $U \sim \text{Uniform}(0,1)$ and $p$ is the connection density. This produces edges with the same statistical properties as independent Bernoulli sampling but in O(expected edges) time, which is critical for the optic lobe (80K neurons) where naive sampling would require 6.4 billion trials.

### 7.3.2 Background Synaptic Bombardment

Brain specs support `background_mean` and `background_std` fields that inject Gaussian noise current to all neurons at each timestep, simulating tonic synaptic bombardment from unmodeled brain regions. This maintains neurons near threshold and produces biologically plausible spontaneous firing rates (0.3-3.5 Hz across regions with mean=12, std=4, weight_scale=0.3).

### 7.4 Timed Stimuli

Stimulus specifications target a named region with a fractional subset of neurons, delivering external current of specified intensity during a time window. For fractional targeting, a deterministic subset of the region's first $N$ neurons is selected, ensuring stimulus stability across timesteps.

### 7.5 Example: Mushroom Body Circuit

A representative mushroom body specification includes three regions: antennal lobe (500 neurons, 40% PN / 60% LN, mixed ACh/GABA), mushroom body (2000 neurons, 85% KC / 8% MBON / 4% DAN_PAM / 3% DAN_PPL1, ACh/GABA/DA), and lateral horn (300 neurons, 50% PN / 30% LN / 20% FS). Inter-region projections implement the canonical AL-to-MB (sparse, excitatory), AL-to-LH (dense, excitatory), MB-to-LH (sparse, excitatory), and LH-to-MB (sparse, inhibitory) pathways.

### 7.6 Parameter Sweep

The parameter sweep engine auto-tunes Izhikevich parameters $(a, b, c, d)$ for each cell type using a three-stage procedure:

1. **Grid sweep**: Exhaustive $N^4$ search over the parameter space (default $N = 5$, yielding 625 evaluations)
2. **Random sweep**: Uniform sampling of additional candidate points
3. **Hill-climbing refinement**: Stochastic perturbation of the best-scoring parameters with Gaussian noise, accepting improvements over multiple iterations

Built-in scoring functions include `TargetFiringRate` (inverse of firing rate error), `ActivityInRange` (fraction active within bounds), and `RealisticCV` (coefficient of variation of inter-spike intervals targeting biologically plausible values of ~0.7).

### 7.7 Parametric Sync

The sync engine tunes a parametric model to match a reference brain (real or simulated) through three concurrent adaptation mechanisms:

| Timescale | Interval | Mechanism |
|-----------|----------|-----------|
| Fast | Every step | Corrective current injection: $I_{\text{ext},i} \mathrel{+}= \gamma (v_{\text{ref},i} - v_{\text{model},i})$ with gain $\gamma = 0.5$ |
| Medium | 100 steps | Momentum SGD on per-synapse error attribution ($\eta = 0.0005$, $\mu = 0.9$) |
| Slow | 1000 steps | Izhikevich parameter nudges: adjusts $(a, c, d)$ based on firing rate mismatch |

Convergence is tracked per neuron via EMA correlation ($\alpha = 0.01$) and declared when the target fraction (default 95%) of neurons exceed the convergence correlation threshold (default 0.85). The sync engine supports early termination upon convergence.


## 8. Analysis Pipeline

### 8.1 Data Recording

Simulation output is recorded in binary and CSV formats via the `Recorder` module:

- **spikes.bin**: Binary spike rasters. Header: `[n_neurons:u32] [n_steps:u32]`. Per step: `[time_ms:f32] [spiked[0..n]:u8]`.
- **voltages.bin**: Full voltage traces (optional, memory-intensive). Same header format; per step: `[time_ms:f32] [v[0..n]:f32]`.
- **metrics.csv**: Time series of spike counts, correlation, RMSE, voltage error, false positives/negatives, resync count, and replacement percentage.
- **per_neuron_error.bin**: Per-neuron prediction error over time. Per step: `[time_ms:f32] [error[0..n]:f32]`.

### 8.2 Per-Region Metrics

When running in parametric mode, per-region activity is tracked at configurable intervals:

- **Spike count**: Number of spikes per region per snapshot
- **Firing rate** (Hz): Spikes per neuron per second within the measurement window
- **Fraction active**: Proportion of neurons that have spiked at least once
- **Mean voltage**: Average membrane potential across the region

Region summaries report total spikes, peak firing rate, and mean fraction active across all snapshots.

### 8.3 Population Sparseness

Sparse coding in the mushroom body is assessed via the fraction of active Kenyon cells per odor presentation. Biologically, mushroom body coding is highly sparse with approximately 5-10% of KCs responding to any given odor (Honegger et al., 2011). The per-region fraction active metric directly measures this property.

### 8.4 Oscillation Analysis

Post-hoc analysis of spike rasters supports FFT-based oscillation detection (`analyze_results.py`). Population firing rate time series are computed by binning spike counts, and power spectral density estimates reveal dominant oscillation frequencies for comparison with known *Drosophila* neural oscillations.

### 8.5 Connectome Validation

Upon loading, the connectome undergoes structural validation:

- Degree distribution statistics (in/out min, max, mean, median)
- NT type ratios (excitatory, inhibitory, modulatory)
- Weight statistics (min, max, mean)
- Integrity checks: self-loops, out-of-bounds indices, NaN weights, zero weights, isolated neurons (no incoming or outgoing synapses)

### 8.6 Biological Plausibility

Simulated dynamics are validated against published electrophysiological data:

- KC firing rates compared with Turner et al. (2008): 0-15 Hz baseline, 5-30 Hz during odor response
- Mushroom body sparseness compared with Honegger et al. (2011): 5-10% of KCs active per odor
- MBON odor responses compared with Aso & Rubin (2016): differential response to trained vs. novel odors after dopamine-gated plasticity


## References

Aso Y, Rubin GM (2016). Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5:e16135.

Bi GQ, Poo MM (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience* 18(24):10464-10472.

Dorkenwald S et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634:124-138.

Eckstein N et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Nature Methods* 21:74-82.

Honegger KS, Campbell RAA, Turner GC (2011). Cellular-resolution population imaging reveals robust sparse coding in the *Drosophila* mushroom body. *Journal of Neuroscience* 31(33):11772-11785.

Izhikevich EM (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks* 14(6):1569-1572.

Izhikevich EM (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex* 17(10):2443-2452.

Turner GC, Bazhenov M, Laurent G (2008). Olfactory representations by *Drosophila* mushroom body neurons. *Journal of Neurophysiology* 99(2):734-746.
