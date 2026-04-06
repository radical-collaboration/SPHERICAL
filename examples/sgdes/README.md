# SGDES — Structure-Guided Deep Evolution Solver

Runs the SGDES protein engineering workflow on HPC GPU nodes (Delta / Bridges-2).
Each mutation target is processed independently and in parallel; results land in
`{outdir}/{mutation}_wd/`.

---

## What the workflow does

SGDES optimises protein sequences for a set of mutation targets using an
iterative loop of three stages:

```
for each foldtune round:
    1. DES     — Deep Evolution Solver generates candidate sequences
    2. Embed   — ESM2-650M embeds all candidates into a vector space
    3. Foldseek — structural similarity screen (ProstT5 fast-folding path)
       ├── createdb: convert FASTA → foldseek DB (generates 3Di SS via ProstT5)
       └── easy-search: compare generated vs. reference structures
    4. Rank    — cosine + L2 distance in embedding space selects the top-k most
                 diverse sequences to seed the next round
```

**DES** (`amortized_bo` / `MutationPredictorSolver`) explores sequence space by
mutating an initial population and scoring each candidate with a
`FoldseekSimilarityProblem` oracle that calls foldseek internally.

**Fast-folding path** (default, `fast_folding: true`): ProstT5 predicts 3Di
secondary-structure tokens directly from sequence — no ESMFold needed, much faster.

**Slow-folding path** (`fast_folding: false`): ESMFold runs first to produce PDB
structures; foldseek then compares them with TM-score alignment.

---

## 1. Environment Setup

Choose the setup script for your cluster and run it **once** (or after changing
dependencies):

| Cluster   | Script                  | Python env   | CUDA          |
|-----------|-------------------------|--------------|---------------|
| Delta     | `delta_env_setup.sh`    | Cray PE venv | HPC SDK 12.8  |
| Bridges-2 | `bridges2_env_setup.sh` | Conda env    | module 12.6.1 |

### Delta (NCSA)

```bash
# Optional: override defaults
bash delta_env_setup.sh \
    --env-dir /u/$USER/conda_env/sgdes \
    --sgdes-dir /scratch/bblj/$USER/SGDES \
    --spherical-dir /scratch/bblj/$USER/SPHERICAL
```

The script clones SGDES and SPHERICAL if not present, creates a Python 3.11 venv
from the Cray PE Python, and installs all dependencies including the foldseek GPU
binary and seqkit.  Requires `cray-python/3.11.7` to be available.

### Bridges-2 (PSC)

```bash
# Set your project directory first
export PROJECT=/ocean/projects/<account>/$USER
bash bridges2_env_setup.sh
```

Load modules before running:
```bash
module load anaconda3
module load cuda/12.6.1
```

The script creates a conda env at `$PROJECT/conda_env/sgdes` with Python 3.11 and
installs PyTorch 2.4.0+cu121, TF, JAX, PyTorch Geometric, flash-attn, TRILL, and
SPHERICAL (editable).

Both scripts pin version-sensitive packages (`numpy<2`, `setuptools<71`,
`tensorboard<2.17`) and print a verification summary at the end.

---

## 2. Apply Runtime Patch (Temporary)

Dragon requires one patch to work correctly with SLURM multi-node jobs.  This
is a **temporary workaround** for an upstream bug that may be fixed in a future
DragonHPC release.  Apply it **once after environment setup**, and again after
any `pip install --upgrade dragonhpc`:

```bash
python apply_slurm_patch.py
```

**`apply_slurm_patch.py`**
Target: `dragon/launcher/wlm/slurm.py`

Replaces Dragon's default `srun --ntasks=N` launch command with
`srun --ntasks-per-node=1 --overlap` so that Dragon's backend processes start
correctly in a SLURM multi-node allocation.

---

## 3. Running

### 3a. Single node — interactive node

On an already-allocated interactive GPU node run:

```bash
bash run_interactive.sh --gpus 4
# or with a custom config:
bash run_interactive.sh --gpus 1 --config config_test.yaml
```

`run_interactive.sh` activates the venv, sets `TOTAL_GPUS`, and launches
`dragon -s` automatically.  Pass `--gpus N` to match the number of GPUs
allocated to the node.

### 3b. Single node — sbatch

Submit via SLURM:
```bash
sbatch delta_gpu_sbatch.sh      # Delta
# sbatch bridges2_gpu_sbatch.sh  # Bridges-2
```

**Example sbatch settings for 1 node × 4 GPUs:**
```sh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
```

### 3c. Multi-node — sbatch only

Dragon distributes one mutation worker per GPU; each mutation runs on its
own dedicated GPU.

**Example sbatch settings:**
```sh
#SBATCH --nodes=2             # or 4
#SBATCH --gpus-per-node=1    # 1 GPU per node (see note below)
```

> **Important:** Use `--gpus-per-node=1`, **not** `--gpus=N`.
> `--gpus=N` is a total count that SLURM may satisfy by allocating all GPUs
> to one node, leaving remote nodes with no GPU and causing foldseek to fail
> with *"No GPU devices found"*.

The sbatch script automatically sets `TOTAL_GPUS = nodes × gpus-per-node`
and selects `dragon -s` (single node) or `dragon -m` (multi-node) based on
`SLURM_NNODES`.  No manual edits needed when switching between node counts.

### 3d. Environment variables in the sbatch script

No paths are hardcoded in the Python files.  Set these exports in your sbatch
script before the `dragon` launch line:

| Variable        | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| `CUDA_HOME`     | CUDA toolkit root; `$CUDA_HOME/lib64` is prepended to `LD_LIBRARY_PATH` so foldseek's ggml-CUDA backend can find `libcudart` on every node, including remote Dragon workers |
| `SGDES_DIR`     | Root of the SGDES/TRILL fork; used to add `amortized_bo` to `sys.path` |
| `SPHERICAL_DIR` | Root of the SPHERICAL repo; added to `sys.path`                |
| `TOTAL_GPUS`    | Total GPUs across all nodes (`nodes × gpus-per-node`); set automatically by the sbatch script from SLURM variables; controls mutation concurrency |

**Delta example (`delta_gpu_sbatch.sh`):**
```sh
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SGDES_DIR=/scratch/bblj/$USER/SGDES
export SPHERICAL_DIR=/scratch/bblj/$USER/SPHERICAL
```

**Bridges-2 example (`bridges2_gpu_sbatch.sh`):**
```sh
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SGDES_DIR=$PROJECT/sgdes/SGDES
export SPHERICAL_DIR=$PROJECT/htp/SPHERICAL
```

---

## 4. Configuration Reference (`config.yaml`)

```yaml
# Paths
outdir:    "mayv_output"           # output root; {mutation}_wd/ appended per target
query_dir: ".../splited_mayv"      # directory with one FASTA per mutation target
wt_query:  ".../mayv_ori.fasta"    # wild-type reference sequence

# Execution
engine:          dragon   # "dragon" (HPC) or "concurrent" (local asyncio, no GPU affinity)
# total_gpus is not set here — derived from TOTAL_GPUS env var (nodes × gpus-per-node)
foldtune_rounds: 3        # outer optimisation rounds
fast_folding:    true     # true = ProstT5 3Di (fast); false = ESMFold (slow)
fold_batch_size: 1

# DES hyperparameters
des_rounds:        3      # DES steps per foldtune round
des_batch_size:    100    # candidate sequences evaluated per DES step
des_num_sequences: 50     # top sequences kept after DES
num_mutations:     1      # point mutations per candidate
topk:              10     # sequences forwarded to next round

# Telemetry (independent flags)
collect_nvml_telemetry:   true          # per-node GPU util via NVML; all worker nodes captured
collect_dragon_telemetry: true          # Dragon runtime metrics; dragon engine only
nvml_dir:                 "nvml-telemetry"
dragon_telemetry_dir:     "dragon-telemetry"
nvml_collection_rate:     1.0           # seconds between NVML samples
nvml_checkpoint_interval: 30.0          # seconds between checkpoint flushes

# Mutation targets — uncomment to enable
mutations:
  - T365F
  # - Y155T
  # ...
```

---

## 5. Output Structure

```
{outdir}/
└── T365F_wd/
    ├── T365F_run_experiment_config.json
    ├── T365F_run_foldtune_input_esm2_t33_650M_AVG.csv    # input embeddings
    ├── T365F_run_round1_des.fasta                         # DES candidates
    ├── T365F_run_foldtune_generated_sequences_round1.fasta
    ├── T365F_run_foldtune_foldseek_round1.tsv             # structural similarity
    ├── T365F_run_foldtune_most-distant_round1.fasta       # selected for round 2
    ├── T365F_run_round1_seq_records.csv                   # per-sequence metrics
    ├── T365F_run_round1_eval_*.json                       # round evaluation stats
    └── ...  (rounds 2, 3, …)
nvml-telemetry/                                            # per-node GPU util logs (one file per node)
dragon-telemetry/                                          # Dragon runtime metrics (dragon engine only)
```

---

## 6. Monitoring

SLURM output goes to `slurm-<jobid>.out`.  Key log lines to watch:

```
[T365F] ── Foldtuning round 1/3 ──
[T365F] DES start (ft_round=1  des_rounds=3  batch=100  n_seq=50 ...)
[T365F] foldseek createdb (generated seqs) → ...
[T365F] foldseek easy-search → ...
[T365F] Round 1 eval → struct(bits): min=3047  mean=3110  max=3146
[T365F] ── Round 1 complete  elapsed=...s ──
Done.
```

Two independent telemetry streams are collected when enabled in `config.yaml`:

**NVML telemetry** (`collect_nvml_telemetry: true`) — GPU utilisation and memory
sampled every 1 s, checkpointed every 30 s.  A `NvmlMonitor` is started on each
worker node via Dragon function tasks, so all nodes write to `nvml-telemetry/`
(filenames are hostname-namespaced; no collisions on shared Lustre storage).

**Dragon telemetry** (`collect_dragon_telemetry: true`, dragon engine only) —
Dragon runtime metrics collected via `DragonTelemetryCollector` and written to
`dragon-telemetry/`.

To plot GPU utilization and memory after a run:

```bash
# from the SPHERICAL root
python src/plot/plot_nvml.py \
    --telemetry-dir examples/sgdes/nvml-telemetry \
    --output gpu_util.png
```

This reads `nvml_checkpoint_*.json` files and produces two subplots (compute
utilization and memory usage per GPU) plus a per-GPU summary table.

To plot Dragon runtime metrics:

```bash
python src/plot/plot_dragon.py \
    --telemetry-dir examples/sgdes/dragon-telemetry \
    --output dragon_util.png
```

See the [root README](../../README.md#metrics--visualization) for full plotting
documentation.

---

## 7. Design Notes

**foldseek as `function_task` (not `executable_task`)**
Launching foldseek as a direct Dragon executable subprocess causes ggml-CUDA
initialisation to fail silently (empty `_ss` files) or hang inside Dragon's
CUDA IPC context.  Running via `subprocess.run()` inside a Python worker thread
avoids that context entirely.

**Three `task_description` policy tiers: `_TD_GPU`, `_TD_HOST`, `_TD_CPU`**

Every task in `_register_tasks` is pinned to the correct node and/or GPU via
one of three policy templates:

| Template   | Placement   | `gpu_affinity` | Used by                              |
|------------|-------------|----------------|--------------------------------------|
| `_TD_GPU`  | `HOST_NAME` | `[gpu_id]`     | `embed`, `fold`, `foldseek_createdb` |
| `_TD_HOST` | `HOST_NAME` | *(none)*       | `foldseek_search`                    |
| `_TD_CPU`  | default     | *(none)*       | `seqkit_grep`, `seqkit_stats`        |
| *(none)*   | —           | —              | `run_des`                            |

`_TD_HOST` was designed to pin `run_des` to the correct node via `HOST_NAME`
placement without `gpu_affinity` (so Dragon would not pre-initialise a CUDA IPC
context that deadlocks TensorFlow).  However, `_TD_HOST` on `run_des` causes
Dragon to spawn a **new managed process** on the remote node to satisfy the
policy.  That process imports the module, initialising JAX (which is
multithreaded), and then `_run_des` calls `os.fork()` internally via
multiprocessing inside `amortized_bo` / TF model evaluation.  `fork()` after a
multithreaded JAX init deadlocks reliably.

`run_des` is therefore left **unbound** (no `task_description`).  Dragon runs it
as a thread inside an existing worker — no new process, no fork-after-JAX
problem.  `CUDA_VISIBLE_DEVICES` is set manually to point TF at the right GPU.

`foldseek_search` is safe with `_TD_HOST` because it only calls `subprocess.run`
with no fork/multiprocessing involved.

**`JAX_PLATFORMS=cpu`**
Forced at import time.  The system cuDNN may be older than what jaxlib was
compiled against; forcing CPU avoids a version-mismatch error on import.

**`TF_FORCE_GPU_ALLOW_GROWTH=true`**
Prevents TensorFlow (used by DES) from pre-allocating ~90% of GPU memory on
first use, which would starve subsequent trill embed calls on the same GPU.

---

## 8. Dragon Scaling

Three concurrent runs on Delta (NCSA) compared on **2026-04-05** with identical
config (`des_rounds=3`, `des_batch_size=100`, `des_num_sequences=50`,
`foldtune_rounds=3`, `fast_folding=true`):

| Nodes | Mutations | Wall time | Per-mutation |
|------:|----------:|----------:|-------------:|
|     1 |         1 |     443 s |        443 s |
|     2 |         2 |     484 s |        242 s |
|     4 |         4 |     517 s |        129 s |

**Throughput scaling:** 2 nodes → 1.8× speedup per mutation; 4 nodes → 3.4× —
roughly linear efficiency (91% at 2 nodes, 85% at 4 nodes).

### Round-by-round breakdown

Each mutation runs three foldtune rounds (embed → DES → foldseek).  Times shown
are per-mutation wall times, averaged across mutations when multiple ran in
parallel (all mutations within a run execute concurrently and finish within ~1 s
of each other):

| Round        | Component       | 1 node  | 2 nodes | 4 nodes |
|:-------------|:----------------|--------:|--------:|--------:|
| R1           | Input embed     |    33 s |    28 s |    59 s |
| R1           | DES             |    47 s |    48 s |    58 s |
| R1           | Generated embed |    31 s |    32 s |    31 s |
| R1           | Foldseek search |    40 s |    44 s |    50 s |
| **R1 total** |                 | **189 s** | **195 s** | **249 s** |
| R2           | DES             |    29 s |    28 s |    28 s |
| R2           | Generated embed |    33 s |    33 s |    32 s |
| R2           | Foldseek search |    35 s |    33 s |    34 s |
| **R2 total** |                 | **136 s** | **132 s** | **133 s** |
| R3           | DES             |    29 s |    28 s |    28 s |
| R3           | Generated embed |    32 s |    37 s |    32 s |
| R3           | Foldseek search |    27 s |    40 s |    35 s |
| **R3 total** |                 | **119 s** | **155 s** | **131 s** |

### Observations

**Round 1 overhead at 4 nodes (+60 s vs 1 node)**
Input embed takes 59 s on 4 nodes vs 33 s on 1 node.  The extra ~26 s is Dragon
inter-node dispatch latency: the first embed call on a remote node incurs Dragon
GS (Global Services) round-trips to establish the managed process and load ESM2
weights into GPU memory.  Rounds 2–3 reuse the already-warm worker and run at
the same speed regardless of node count.

**DES is node-count-independent after R1**
R2–R3 DES times (28–29 s) are identical across all three runs.  DES runs
unbound inside an existing Dragon worker thread — no inter-node scheduling
overhead — and TF model weights are already loaded from R1.

**Foldseek varies more at scale**
R1 foldseek search grows from 40 s (1 node) to 50 s (4 nodes) because four
concurrent `foldseek easy-search` processes compete for Lustre I/O bandwidth
when reading the reference DB.  R2–R3 foldseek is near-identical (33–35 s)
since the Lustre metadata is cached.

**Practical guideline**
For ≤ 4 mutations the per-mutation overhead of multi-node Dragon is negligible
beyond R1 cold-start.  Scaling beyond 4 nodes (more mutations) is expected to
remain near-linear for R2+ while R1 overhead grows proportionally with the
number of remote nodes.

### GPU utilization across all 4 nodes (4-node run)

![GPU utilization — all 4 nodes](gpu_util.png)

Each subplot is one node (gpub004, gpub029, gpub051, gpub085).  The GPU
utilization patterns (black) are nearly identical across all four nodes —
each node carries exactly one mutation, performs the same embed → DES →
foldseek sequence, and saturates the GPU at the same workflow stages.  The
symmetric load distribution confirms that Dragon's `HOST_NAME` + `gpu_affinity`
policy correctly pins one mutation per node with no skew.
