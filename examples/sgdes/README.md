# SGDES — Structure-Guided Deep Evolution Solver

Runs the SGDES protein engineering workflow on Bridges-2 (PSC) GPU nodes.
Each mutation target is processed independently and in parallel; results land in
`mayv_output/{mutation}_wd/`.

---

## What the workflow does

SGDES optimises protein sequences for a set of mutation targets using an
iterative loop of three stages:

```
for each foldtune round:
    1. DES  — Deep Evolution Solver generates candidate sequences
    2. Embed — ESM2-650M embeds all candidates into a vector space
    3. Foldseek — structural similarity screen (ProstT5 fast-folding path)
       ├── createdb: convert FASTA → foldseek DB (generates 3Di SS via ProstT5)
       └── easy-search: compare generated vs. reference structures
    4. Rank — cosine + L2 distance in embedding space selects the top-k most
               diverse sequences to seed the next round
```

**DES** (`amortized_bo` / `MutationPredictorSolver`) explores sequence space by
mutating an initial population and scoring each candidate with a
`FoldseekSimilarityProblem` oracle that calls foldseek internally.

**Fast-folding path** (default, `fast_folding: true`): ProstT5 predicts 3Di
secondary-structure tokens directly from sequence — no ESMFold needed, much faster.

**Slow-folding path** (`fast_folding: false`): ESMFold runs first to produce PDB
structures; foldseek then compares them with TM-score alignment.

After each round the top-k most distant sequences (by embedding distance from
the input) are saved as `{mutation}_run_foldtune_most-distant_round{i}.fasta`
and used as the seed for the next round.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11 | managed by conda |
| PyTorch | 2.4.0+cu121 | GPU training |
| TRILL (SGDES fork) | editable | `$PROJECT/sgdes/SGDES` |
| SPHERICAL | editable | this repo |
| foldseek (GPU build) | 10-941cd33 | bundles ProstT5 / ggml |
| Dragon HPC | system | loaded via `gpu_sbatch.sh` |
| seqkit, ambertools | conda-forge | sequence utilities |

---

## Environment setup

Run once (or after changing dependencies):

```bash
bash env_setup.sh
```

The script:
1. Loads `anaconda3` and `cuda` modules.
2. Creates (or recreates) a conda env at `$PROJECT/conda_env/sgdes` with Python 3.11.
3. Installs PyTorch 2.4.0+cu121, JAX, TensorFlow, PyTorch Geometric, flash-attn.
4. Downloads the foldseek GPU binary from GitHub releases (includes ProstT5 / ggml-CUDA).
5. Installs TRILL (editable) from `$PROJECT/sgdes/SGDES` and SPHERICAL (editable).
6. Pins version-sensitive packages (`numpy<2`, `setuptools<71`, `tensorboard<2.17`).
7. Prints a verification summary at the end.

**Required environment variables** (set in your `.bashrc` or job script):

```bash
export PROJECT=/ocean/projects/dmr170002p/goliyad   # PSC Bridges-2 project dir
```

---

## Running on Bridges-2

```bash
cd $PROJECT/htp/SPHERICAL/examples/sgdes
sbatch gpu_sbatch.sh
```

`gpu_sbatch.sh` requests **1 node, 8 GPUs, 40 CPUs, 2.5 h** on the GPU partition
and does the following before launching:

1. Loads `anaconda3`, `cuda/12.6.1`, `cudnn/8.0.4`.
2. Activates the `$PROJECT/conda_env/sgdes` conda env.
3. Sets `CUDA_HOME` and `LD_LIBRARY_PATH` for the CUDA 12.6.1 toolkit.
4. Clears stale output dirs (`mayv_output/`, `tmp*/`, `nvml-telemetry/`).
5. Launches `run_workflow.py` under Dragon: `dragon -s run_workflow.py`.

Dragon provides distributed process management and GPU affinity control.
`run_workflow.py` initialises `DragonExecutionBackendV3` (16 workers), creates
an asyncflow `WorkflowEngine`, then calls `SGDESWorkflow.run()`.

---

## Configuration — `config.yaml`

```yaml
# Paths
outdir:    "mayv_output"          # output root; {mutation}_wd/ appended per target
query_dir: ".../splited_mayv"     # one FASTA per mutation target
wt_query:  ".../mayv_ori.fasta"   # wild-type reference sequence

# Execution
engine:          dragon    # "dragon" (HPC) or "concurrent" (local asyncio)
total_gpus:      8         # number of GPUs on the node
foldtune_rounds: 3         # outer optimisation rounds
fast_folding:    true      # use ProstT5 (true) or ESMFold (false)
fold_batch_size: 1

# DES hyperparameters
des_rounds:        3       # DES steps per foldtune round
des_batch_size:    100     # candidate sequences evaluated per DES step
des_num_sequences: 50      # top sequences kept after DES
num_mutations:     1       # point mutations per candidate
topk:              10      # sequences forwarded to next round

# Mutation targets (uncomment to enable)
mutations:
  - T365F
  # - Y155T
  # ...
```

To run multiple mutations concurrently, uncomment additional entries in the
`mutations` list. Each mutation gets its own GPU (round-robin) via Dragon
`Policy(gpu_affinity=[...])`.

---

## Output structure

```
mayv_output/
└── T365F_wd/
    ├── T365F_run_experiment_config.json
    ├── T365F_run_foldtune_input_esm2_t33_650M_AVG.csv   # input embeddings
    ├── T365F_run_round1_des.fasta                        # DES output
    ├── T365F_run_foldtune_generated_sequences_round1.fasta
    ├── T365F_run_foldtune_foldseek_round1.tsv            # structural similarity
    ├── T365F_run_foldtune_most-distant_round1.fasta      # selected for next round
    ├── T365F_run_round1_seq_records.csv                  # per-sequence metrics
    ├── T365F_run_round1_eval_*.json                      # round evaluation stats
    └── ...  (rounds 2, 3, ...)
nvml-telemetry/                                           # GPU utilisation logs
```

---

## Monitoring

SLURM output goes to `slurm-<jobid>.out` in the working directory.
Key log lines to watch:

```
[T365F] ── Foldtuning round 1/3 ──
[DES ft_round=1] Starting 3 DES steps  batch_size=100
[DES ft_round=1 step=1/3] step=460.3s  best_reward=0.8421  mean_reward=0.7103
[T365F] foldseek createdb (generated seqs) → ...
[T365F] Foldseek done (12.4s)
[T365F] Round 1 eval done → ...json  struct(bits): min=... mean=...
[T365F] ── Round 1 complete  elapsed=...s ──
```

GPU utilisation is sampled every 1 s and checkpointed to `nvml-telemetry/`
every 30 s. Use `plot_nvml_gpu_util.py` to visualise after the run.

---

## Known issues / notes

- **foldseek tasks run as Dragon `function_task`** (not `executable_task`).
  Launching foldseek as a direct Dragon subprocess causes ggml-CUDA
  initialisation to fail silently (empty `_ss` files) or hang. Running via
  `subprocess.run()` inside a Python worker avoids that context entirely.

- **`JAX_PLATFORMS=cpu`** is forced at import time. The system cuDNN (8.0.4)
  is older than what jaxlib was compiled against (8.9.6); without this flag JAX
  raises a version mismatch error on import.

- **`TF_FORCE_GPU_ALLOW_GROWTH=true`** prevents TensorFlow (used by DES) from
  pre-allocating 90% of GPU memory on first use, which would starve subsequent
  trill embed calls on the same GPU.

- **DES is CPU-bound** (`CUDA_VISIBLE_DEVICES=-1` inside `_run_des`). Foldseek
  requires a clean CUDA context and conflicts with Dragon's CUDA IPC. Each DES
  step takes ~7–8 minutes for 100 sequences on CPU.
