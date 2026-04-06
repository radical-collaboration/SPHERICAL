"""
SGDESWorkflow — asyncflow-based SGDES workflow.

Dispatches all shell commands through asyncflow tasks instead of blocking
subprocess.run calls, enabling concurrent execution of multiple mutations.

Task types
----------
executable_task — returns a command string; Dragon/asyncflow launches it as a
                  subprocess.  Used for trill (embed, fold) so Dragon can apply
                  GPU affinity and HOST_NAME placement via task_description.
function_task   — runs a Python async function directly in a Dragon worker
                  thread.  Used for foldseek and DES (run_des) because those
                  need subprocess.run with a custom environment, or call
                  TensorFlow code that must not share a CUDA context with
                  Dragon's gpu_affinity infrastructure.

GPU / CUDA handling
-------------------
- embed / fold (executable_task): receive task_description with a Policy
  containing gpu_affinity + HOST_NAME; Dragon sets CUDA_VISIBLE_DEVICES for
  those subprocesses automatically.
- foldseek_createdb / foldseek_search (function_task): set LD_LIBRARY_PATH
  (from $CUDA_HOME/lib64) in the subprocess env so foldseek's ggml-CUDA
  backend can find libcudart on every node, including remote Dragon workers
  that may not inherit the head-node environment.
- foldseek_search (function_task, _TD_HOST): pinned to the correct node via a
  Policy with HOST_NAME placement but NO gpu_affinity, so Dragon does not
  pre-initialise a CUDA IPC context.
- run_des (function_task, no task_description): intentionally unbound.  _TD_HOST
  was tried but Dragon spawns a new managed process to satisfy the policy; that
  process initialises JAX (multithreaded) and then _run_des calls os.fork() via
  multiprocessing inside amortized_bo, which deadlocks after JAX init.  Running
  unbound lets Dragon execute run_des as a thread in an existing worker — no new
  process, no fork-after-JAX deadlock.  CUDA_VISIBLE_DEVICES is set manually.

All CUDA-related paths are read from environment variables set in the sbatch
script (CUDA_HOME, SGDES_DIR, SPHERICAL_DIR) — no hardcoded paths in this file.
"""

import asyncio
import json
import os
import shutil
import sys
import time
import types
import warnings
from datetime import datetime

# Must be set before TensorFlow initialises its GPU context.
# amortized_bo (DES) uses TF v1, which by default pre-allocates ~90 % of GPU
# memory on first use and never releases it, starving subsequent trill embed
# subprocess calls.  Memory-growth mode allocates only what is actually needed.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# jaxlib on this system was compiled with cuDNN 8.9.6 but only 8.0.4 is installed;
# force JAX to use the CPU backend to avoid a cuDNN version mismatch error.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# amortized_bo imports JAX at module level, making this process multithreaded.
# Dragon then uses os.fork() to spawn executable_task subprocesses (trill embed),
# which triggers Python's fork-after-threads warning.  The warning is harmless —
# trill runs in its own fresh subprocess and completes normally.
warnings.filterwarnings(
    "ignore",
    message="os.fork\\(\\) was called.*JAX is multithreaded",
    category=RuntimeWarning,
)

import numpy as np
import pandas as pd
from Bio import SeqIO

from src.utils.logger import Logger

logger = Logger(name="sgdes", use_colors=True)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# amortized_bo uses absolute imports (e.g. `from amortized_bo import data`),
# so its parent directory must be on sys.path.
_SGDES_DIR = os.environ.get("SGDES_DIR", "")
_ABO_PARENT = os.path.join(_SGDES_DIR, "trill/utils/abo") if _SGDES_DIR else ""
if _ABO_PARENT and _ABO_PARENT not in sys.path:
    sys.path.insert(0, _ABO_PARENT)

from trill.utils.abo.amortized_bo import controller, data
from trill.utils.abo.amortized_bo.deep_evolution_solver import MutationPredictorSolver
from trill.utils.abo.amortized_bo.foldseek_similarity_problem import FoldseekSimilarityProblem
from trill.utils.fasta_files import remove_invalid_seqs_aa, truncate_seqs
from trill.utils.foldseek_utils import run_foldseek_databases
from trill.utils.sgdes import (
    compute_average_rank_without_df,
    highest_avg_score_by_query,
    save_round_records,
)

AA = "ACDEFGHIKLMNPQRSTVWY"
_aa2id = {a: i for i, a in enumerate(AA)}
_id2aa = list(AA)

try:
    from dragon.infrastructure.policy import Policy
except Exception:
    Policy = None


def _fasta_to_int_array(fa_path, length):
    seqs, cur = [], []
    with open(fa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
                continue
            for c in line.upper():
                if c in _aa2id:
                    cur.append(c)
    if cur:
        seqs.append("".join(cur))
    out = []
    for s in seqs:
        s = (s + "A" * length)[:length]
        out.append([_aa2id[c] for c in s])
    return np.array(out, dtype=int)


def _int_array_to_fasta(arr, out_path, prefix):
    with open(out_path, "w") as f:
        for i, row in enumerate(arr):
            s = "".join(_id2aa[int(x)] for x in row)
            f.write(f">{prefix}_{i}\n{s}\n")


def _fasta_to_numeric(fasta_path, max_length=None, pad_char="A"):
    seq_list = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).strip().upper()
        seq_list.append([_aa2id[a] for a in seq if a in _aa2id])
    if max_length is None:
        max_length = max(len(s) for s in seq_list)
    pad = _aa2id[pad_char]
    arr = []
    for s in seq_list:
        if len(s) < max_length:
            s = s + [pad] * (max_length - len(s))
        else:
            s = s[:max_length]
        arr.append(s)
    return np.array(arr, dtype=np.int32)


def _run_des(
    de_ref_fasta: str,
    fixed_len: int,
    des_fasta: str,
    ori_ref: str,
    ori_ref_structs: str,
    round_i: int,
    fast_folding: bool,
    prostt5_weights_path,
    fold_batch_size: int,
    gpus: str,
    rng_seed: int,
    num_mutations: int,
    des_rounds: int,
    des_batch_size: int,
    des_num_sequences: int,
) -> None:

    ref_path = ori_ref if fast_folding else ori_ref_structs

    t0_total = time.time()

    print(f"[DES ft_round={round_i}] Building input DB from {ref_path}")
    t0 = time.time()
    problem = FoldseekSimilarityProblem(
        input_fasta=ref_path,
        length=fixed_len,
        fast_folding=fast_folding,
        use_gpu=fast_folding,
        # use_gpu=False,
        prostt5_model_path=prostt5_weights_path if fast_folding else None,
        fold_batch_size=fold_batch_size,
        gpus=gpus,
    )
    print(f"[DES ft_round={round_i}] Input DB ready ({time.time() - t0:.1f}s)")

    def my_initializer(domain, batch_size, random_state):
        fasta_array = _fasta_to_numeric(ori_ref, fixed_len)
        n = len(fasta_array)
        assert fasta_array.shape[1] == fixed_len
        if n < batch_size:
            extra_idx = random_state.randint(0, n, size=batch_size - n)
            fasta_array = np.concatenate([fasta_array, fasta_array[extra_idx]])
        return fasta_array[:batch_size]

    solver = MutationPredictorSolver(domain=problem.domain, random_state=int(rng_seed))
    solver.cfg.initialize_dataset_fn = my_initializer
    solver.cfg.num_mutations = num_mutations

    cand_init = _fasta_to_int_array(de_ref_fasta, fixed_len)
    print(f"[DES ft_round={round_i}] cand_init={len(cand_init)} seqs  fixed_len={fixed_len}")
    init_population = data.Population.from_arrays(
        structures=cand_init,
        rewards=np.zeros(len(cand_init)),
        batch_index=0,
    )

    def _init_with_ref(domain, batch_size, random_state):
        if len(cand_init) >= batch_size:
            return cand_init[:batch_size]
        return cand_init[np.resize(np.arange(len(cand_init)), batch_size)]

    solver._config().update(dict(initialize_dataset_fn=_init_with_ref))

    _des_step_times = [time.time()]

    def _des_step_callback(population, force_write=False):
        now = time.time()
        step = population.current_batch_index
        step_elapsed = now - _des_step_times[-1]
        total_elapsed = now - t0_total
        rewards = [s.reward for s in population.get_last_batch()]
        best_r = max(rewards) if rewards else float("nan")
        mean_r = sum(rewards) / len(rewards) if rewards else float("nan")
        print(
            f"[DES ft_round={round_i} step={step}/{des_rounds}] "
            f"step={step_elapsed:.1f}s  total={total_elapsed:.1f}s  "
            f"best_reward={best_r:.4f}  mean_reward={mean_r:.4f}  "
            f"pop={len(population)}"
        )
        _des_step_times.append(now)

    print(f"[DES ft_round={round_i}] Starting {des_rounds} DES steps  batch_size={des_batch_size}")
    population = controller.run(
        problem,
        solver,
        num_rounds=int(des_rounds),
        batch_size=int(des_batch_size),
        initial_population=init_population,
        callbacks=[_des_step_callback],
    )

    best = population.best_n(int(des_num_sequences), discard_duplicates=True)
    print(
        f"[DES ft_round={round_i}] Done — best_n={len(best)}  "
        f"total={time.time() - t0_total:.1f}s  → {des_fasta}"
    )
    des_arr = (
        np.array([s.structure for s in best])
        if len(best) > 0
        else cand_init[: min(int(des_num_sequences), len(cand_init))]
    )
    # Ensure the output directory exists.  run_des runs as an unbound Dragon
    # worker thread and may execute before the head-node makedirs call is
    # visible on Lustre, or on a node that hasn't yet seen the directory.
    os.makedirs(os.path.dirname(des_fasta), exist_ok=True)
    _int_array_to_fasta(des_arr, des_fasta, prefix=f"des_r{round_i}")


class SGDESWorkflow:
    def __init__(self, config: dict, asyncflow=None, policies=None):
        cfg = config

        # ── Paths ─────────────────────────────────────────────────────────────
        self.base_outdir = os.path.abspath(cfg["outdir"])
        self.query_dir = str(cfg["query_dir"])
        self.wt_query = str(cfg["wt_query"])
        self.mutations = list(cfg["mutations"])

        # ── Execution ─────────────────────────────────────────────────────────
        self.total_gpus = int(cfg.get("total_gpus", 1))
        # self.total_cpus      = int(cfg.get("total_cpus", os.cpu_count()))
        self.rng_seed = int(cfg.get("RNG_seed", 42))
        self.foldtune_rounds = int(cfg.get("foldtune_rounds", 3))
        self.fast_folding = bool(cfg.get("fast_folding", True))
        self.fold_batch_size = int(cfg.get("fold_batch_size", 1))

        # ── DES hyperparameters ───────────────────────────────────────────────
        self.des_rounds = int(cfg.get("des_rounds", 3))
        self.des_batch_size = int(cfg.get("des_batch_size", 500))
        self.des_num_sequences = int(cfg.get("des_num_sequences", 200))
        self.num_mutations_des = int(cfg.get("num_mutations", 1))
        self.topk = int(cfg.get("topk", 20))

        if asyncflow is None:
            raise ValueError(
                "SGDESWorkflow_asyncflow requires asyncflow= argument. "
                "Use sgdes_workflow.SGDESWorkflow for the subprocess-based variant."
            )
        self.asyncflow = asyncflow

        os.makedirs(self.base_outdir, exist_ok=True)

        self.policies = policies
        self.collect_nvml_telemetry = bool(cfg.get("collect_nvml_telemetry", True))
        # Absolute path so Dragon workers on remote nodes resolve it correctly.
        self.telemetry_dir = os.path.abspath(cfg.get("nvml_dir", "nvml-telemetry"))
        self.nvml_rate = float(cfg.get("nvml_collection_rate", 1.0))
        self.nvml_checkpoint = float(cfg.get("nvml_checkpoint_interval", 30.0))

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def _register_tasks(self, cuda_device: int, policy):
        """
        Register asyncflow executable tasks for one mutation slot.

        Each task is a closure over `cuda_device` and `policy` so that all
        commands dispatched for the same mutation target the same GPU.

        Returns a SimpleNamespace with callable attributes:
            .embed          — trill embed esm2_t33_650M
            .fold           — trill fold ESMFold  (slow-folding path only)
            .foldseek_createdb  — foldseek createdb
            .foldseek_search    — foldseek easy-search
            .seqkit_grep    — seqkit grep … > output  (shell redirect)
            .seqkit_stats   — seqkit stats -a -T … > tsv  (shell redirect)
        """
        flow = self.asyncflow

        _TD_GPU = (
            {
                "process_template": {
                    "policy": policy,
                },
            }
            if policy is not None
            else {}
        )

        _TD_CPU = (
            {
                "process_template": {
                    "policy": Policy() if Policy is not None else None,
                },
            }
            if Policy is not None
            else {}
        )

        # Host-only policy: pins task to the correct node without gpu_affinity so
        # Dragon does not pre-initialise a CUDA IPC context in the worker.
        # Used for run_des (TF deadlocks on Dragon's CUDA context) and
        # foldseek_search (no GPU needed, but must run on the same node as its DBs).
        if policy is not None and Policy is not None:
            _host_policy = Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=policy.host_name,
            )
            _TD_HOST = {"process_template": {"policy": _host_policy}}
        else:
            _TD_HOST = {}

        # ── trill embed ────────────────────────────────────────────────────────
        @flow.executable_task
        async def embed(task_description=_TD_GPU, **kwargs):
            """Run trill embed esm2_t33_650M as an executable_task.
            kwargs: name, GPUs, seed, outdir, query
            """
            name = kwargs["name"]
            GPUs = kwargs["GPUs"]
            seed = kwargs["seed"]
            outdir = kwargs["outdir"]
            query = kwargs["query"]
            cmd = (
                f"trill {name} {GPUs} --RNG_seed {seed} --outdir {outdir} "
                f"embed esm2_t33_650M {query} --avg"
            )
            print(f"[embed] cmd: {cmd}", flush=True)
            return cmd

        # ── trill fold (ESMFold, slow path only) ───────────────────────────────
        @flow.executable_task
        async def fold(task_description=_TD_GPU, **kwargs):
            """Run trill fold ESMFold as an executable_task.
            kwargs: name, GPUs, seed, outdir, query, batch_size
            """
            name = kwargs["name"]
            GPUs = kwargs["GPUs"]
            seed = kwargs["seed"]
            outdir = kwargs["outdir"]
            query = kwargs["query"]
            batch_size = kwargs["batch_size"]
            cmd = (
                f"trill {name} {GPUs} --RNG_seed {seed} --outdir {outdir} "
                f"fold ESMFold {query} --batch_size {batch_size}"
            )
            print(f"[fold] cmd: {cmd}", flush=True)
            return cmd

        # ── foldseek createdb ─────────────────────────────────────────────────
        # NOTE: implemented as function_task (not executable_task) so that foldseek
        # runs as a grandchild subprocess of the Dragon worker process.  When foldseek
        # is launched as a *direct* Dragon executable_task subprocess the ggml-CUDA
        # backend initialises inside Dragon's CUDA IPC context, which causes a silent
        # failure: the _ss (3Di secondary structure) file is written as empty even
        # without --gpu 1.  Running via subprocess.run() avoids that context entirely.
        @flow.function_task
        async def foldseek_createdb(task_description=_TD_GPU, **kwargs):
            """Run foldseek createdb via subprocess.
            kwargs: fasta, db_path, prostt5_model (optional)
            Note: --gpu 1 omitted — foldseek ggml-CUDA crashes in Dragon subprocess context.
            """
            import subprocess as _sp

            fasta = kwargs["fasta"]
            db_path = kwargs["db_path"]
            prostt5_model = kwargs.get("prostt5_model", "")
            model_flag = f"--prostt5-model {prostt5_model}" if prostt5_model else ""
            gpu_flag = " --gpu 1" if prostt5_model else ""
            cmd = f"foldseek createdb {fasta} {db_path} {model_flag}{gpu_flag}".strip()
            print(f"[foldseek_createdb] cmd: {cmd}", flush=True)
            import os as _os

            env = _os.environ.copy()
            # Ensure foldseek ggml-CUDA can find libcudart on Delta
            cuda_lib = os.path.join(os.environ.get("CUDA_HOME", ""), "lib64")
            if cuda_lib and cuda_lib != "/lib64":
                env["LD_LIBRARY_PATH"] = cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            # env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
            result = _sp.run(cmd, shell=True, capture_output=True, text=True, env=env)
            if result.stdout:
                print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"foldseek createdb failed (rc={result.returncode}): {result.stderr[-500:]}"
                )

        # ── foldseek easy-search ──────────────────────────────────────────────
        # NOTE: function_task for the same reason as foldseek_createdb — direct
        # Dragon executable_task subprocess hangs due to ggml-CUDA context conflict.
        @flow.function_task
        async def foldseek_search(task_description=_TD_HOST, **kwargs):
            """Run foldseek easy-search via subprocess.
            kwargs: query_db, target_db, tsv_out, tmp_dir, extra_flags (optional)
            """
            import subprocess as _sp

            query_db = kwargs["query_db"]
            target_db = kwargs["target_db"]
            tsv_out = kwargs["tsv_out"]
            tmp_dir = kwargs["tmp_dir"]
            extra_flags = kwargs.get("extra_flags", "")
            cmd = f"foldseek easy-search {query_db} {target_db} {tsv_out} {tmp_dir} {extra_flags}".rstrip()
            print(f"[foldseek_search] cmd: {cmd}", flush=True)
            result = _sp.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"foldseek easy-search failed (rc={result.returncode}): {result.stderr[-500:]}"
                )

        # ── seqkit grep → file ────────────────────────────────────────────────
        @flow.function_task
        async def seqkit_grep(task_description=_TD_CPU, **kwargs):
            """Run seqkit grep via subprocess, redirecting stdout to output_fasta.
            kwargs: pattern_file, input_fasta, output_fasta
            """
            import subprocess as _sp

            pattern_file = kwargs["pattern_file"]
            input_fasta = kwargs["input_fasta"]
            output_fasta = kwargs["output_fasta"]
            cmd = f"seqkit grep --pattern-file {pattern_file} {input_fasta} > {output_fasta}"
            print(f"[seqkit_grep] cmd: {cmd}", flush=True)
            result = _sp.run(cmd, shell=True, capture_output=True, text=True)
            if result.stderr:
                print(result.stderr, flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"seqkit grep failed (rc={result.returncode}): {result.stderr[-500:]}"
                )

        # ── seqkit stats → TSV file ───────────────────────────────────────────
        @flow.function_task
        async def seqkit_stats(task_description=_TD_CPU, **kwargs):
            """Run seqkit stats -a -T via subprocess, redirecting stdout to out_tsv.
            kwargs: input_fasta, out_tsv
            """
            import subprocess as _sp

            input_fasta = kwargs["input_fasta"]
            out_tsv = kwargs["out_tsv"]
            cmd = f"seqkit stats -a -T {input_fasta} > {out_tsv}"
            print(f"[seqkit_stats] cmd: {cmd}", flush=True)
            result = _sp.run(cmd, shell=True, capture_output=True, text=True)
            if result.stderr:
                print(result.stderr, flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"seqkit stats failed (rc={result.returncode}): {result.stderr[-500:]}"
                )

        @flow.function_task
        async def run_des(**kwargs):
            # No task_description — intentionally unbound.
            #
            # _TD_HOST (HOST_NAME, no gpu_affinity) was tried to pin run_des to
            # the correct node, but Dragon spawns a new managed process on the
            # remote node to satisfy the policy.  That new process imports the
            # module, initialising JAX (multithreaded), then _run_des calls
            # os.fork() via multiprocessing inside amortized_bo / TF model
            # evaluation.  fork() after a multithreaded JAX init deadlocks.
            #
            # Without a policy Dragon runs run_des as a thread inside an existing
            # worker — no new process, no fork-after-JAX problem.
            # CUDA_VISIBLE_DEVICES below points TF at the right GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
            # Dragon workers on remote nodes may not inherit LD_LIBRARY_PATH from
            # the head node's sbatch environment.  foldseek's ggml-CUDA backend
            # needs $CUDA_HOME/lib64 to dlopen libcudart at runtime.
            _cuda_lib = os.path.join(os.environ.get("CUDA_HOME", ""), "lib64")
            if _cuda_lib and _cuda_lib != "/lib64":
                os.environ["LD_LIBRARY_PATH"] = (
                    _cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                )
            _run_des(
                de_ref_fasta=kwargs["de_ref_fasta"],
                fixed_len=kwargs["fixed_len"],
                des_fasta=kwargs["des_fasta"],
                ori_ref=kwargs["ori_ref"],
                ori_ref_structs=kwargs["ori_ref_structs"],
                round_i=kwargs["round_i"],
                fast_folding=kwargs["fast_folding"],
                prostt5_weights_path=kwargs["prostt5_weights_path"],
                fold_batch_size=kwargs["fold_batch_size"],
                gpus=kwargs["gpus"],
                rng_seed=kwargs["rng_seed"],
                num_mutations=kwargs["num_mutations"],
                des_rounds=kwargs["des_rounds"],
                des_batch_size=kwargs["des_batch_size"],
                des_num_sequences=kwargs["des_num_sequences"],
            )

        # ── per-node NVML telemetry ───────────────────────────────────────────
        # start_node_telemetry / stop_node_telemetry use _TD_HOST (HOST_NAME,
        # no gpu_affinity) to run on the correct remote node.
        #
        # The monitor runs as an INDEPENDENT SUBPROCESS (nvml_monitor_worker.py),
        # NOT as a daemon thread inside the Dragon worker.  Dragon kills its
        # managed processes after each function_task returns; a daemon thread
        # would die with the process before writing any data.  A subprocess with
        # its own event loop survives Dragon's worker lifecycle.
        #
        # Communication: start writes a PID file to the shared Lustre outdir;
        # stop reads that PID file and sends SIGTERM to the subprocess, which
        # flushes remaining samples before exiting.  This works regardless of
        # which Dragon worker process runs start vs. stop.
        @flow.function_task
        async def start_node_telemetry(task_description=_TD_HOST, **kwargs):
            import os as _os
            import socket as _socket
            import subprocess as _sp
            import sys as _sys

            outdir = kwargs["outdir"]
            rate = kwargs.get("rate", 1.0)
            interval = kwargs.get("checkpoint_interval", 30.0)
            spherical_dir = _os.environ.get("SPHERICAL_DIR", "")

            _os.makedirs(outdir, exist_ok=True)

            hostname = _socket.gethostname()
            pid_file = _os.path.join(outdir, f"nvml_pid_{hostname}.txt")
            if _os.path.exists(pid_file):
                return  # already running on this node

            worker = _os.path.join(spherical_dir, "src", "utils", "nvml_monitor_worker.py")
            cmd = [
                _sys.executable, worker,
                "--outdir", outdir,
                "--rate", str(rate),
                "--checkpoint-interval", str(interval),
                "--spherical-dir", spherical_dir,
            ]
            log_path = _os.path.join(outdir, f"nvml_worker_{hostname}.log")
            log_fh = open(log_path, "w")
            _sp.Popen(cmd, start_new_session=True, stdout=log_fh, stderr=log_fh)
            print(f"[start_node_telemetry] launched nvml_monitor_worker on {hostname} → {log_path}")

        @flow.function_task
        async def stop_node_telemetry(task_description=_TD_HOST, **kwargs):
            import os as _os
            import signal as _signal
            import socket as _socket
            import time as _time

            outdir = kwargs["outdir"]
            hostname = _socket.gethostname()
            pid_file = _os.path.join(outdir, f"nvml_pid_{hostname}.txt")

            if not _os.path.exists(pid_file):
                return

            try:
                with open(pid_file) as _f:
                    pid = int(_f.read().strip())
                _os.kill(pid, _signal.SIGTERM)
                _time.sleep(3)  # allow final checkpoint flush
            except (OSError, ValueError, ProcessLookupError):
                pass

        return types.SimpleNamespace(
            embed=embed,
            fold=fold,
            foldseek_createdb=foldseek_createdb,
            foldseek_search=foldseek_search,
            seqkit_grep=seqkit_grep,
            seqkit_stats=seqkit_stats,
            run_des=run_des,
            start_node_telemetry=start_node_telemetry,
            stop_node_telemetry=stop_node_telemetry,
        )

    # ------------------------------------------------------------------
    # Per-mutation async implementation
    # ------------------------------------------------------------------

    async def _sgdes_async(self, mutation: str, cuda_device: int, policy) -> None:
        """Async port of _sgdes_blocking; all commands run via executable_task."""
        tasks = self._register_tasks(cuda_device, policy)

        abspath = os.path.join(self.base_outdir, f"{mutation}_wd")
        os.makedirs(abspath, exist_ok=True)

        if self.collect_nvml_telemetry:
            await tasks.start_node_telemetry(
                outdir=self.telemetry_dir,
                rate=self.nvml_rate,
                checkpoint_interval=self.nvml_checkpoint,
            )

        name = f"{mutation}_run"
        query = os.path.join(self.query_dir, f"mayv_{mutation}.fasta")
        GPUs = "1"

        args = types.SimpleNamespace(
            name=name,
            outdir=abspath,
            query=query,
            wt_query=self.wt_query,
            GPUs=GPUs,
            RNG_seed=self.rng_seed,
            fast_folding=self.fast_folding,
            fold_batch_size=self.fold_batch_size,
            foldtune_rounds=self.foldtune_rounds,
            des_rounds=self.des_rounds,
            des_batch_size=self.des_batch_size,
            des_num_sequences=self.des_num_sequences,
            num_mutations=self.num_mutations_des,
            topk=self.topk,
            finetune_batch_size=None,
            finetune_strategy=None,
            lang_gen_batch_size=None,
        )

        # ── Save experiment config ─────────────────────────────────────────────
        exp_config = {
            "experiment_name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "foldtune_rounds": args.foldtune_rounds,
            "query_file": args.query,
            "wt_query_file": args.wt_query,
            "output_dir": abspath,
            "GPUs": args.GPUs,
            "RNG_seed": args.RNG_seed,
            "fast_folding": args.fast_folding,
            "fold_batch_size": args.fold_batch_size,
            "finetune_batch_size": args.finetune_batch_size,
            "finetune_strategy": args.finetune_strategy,
            "lang_gen_batch_size": args.lang_gen_batch_size,
            "num_des_round": args.des_rounds,
            "num_seq_each_des_round": args.des_batch_size,
            "num_selected_seq": args.des_num_sequences,
            "topk": args.topk,
            "num_mutations": args.num_mutations,
            "des_batch_size": args.des_batch_size,
        }
        with open(os.path.join(abspath, f"{name}_experiment_config.json"), "w") as f:
            json.dump(exp_config, f, indent=2)
        logger.info(f"[{mutation}] Config saved")

        prostt5_weights_path = None
        output_fasta = None
        median = None

        t_workflow_start = time.time()
        for i in range(1, int(args.foldtune_rounds) + 1):
            t_round_start = time.time()
            logger.info(f"[{mutation}] ── Foldtuning round {i}/{args.foldtune_rounds} ──")

            # ── Round-1 setup ──────────────────────────────────────────────────
            if args.fast_folding and i == 1:
                logger.info(f"[{mutation}] Finding ProstT5 weights")
                cache_dir = os.path.join(os.path.expanduser("~"), ".trill_cache")
                os.makedirs(cache_dir, exist_ok=True)
                prostt5_weights_path = run_foldseek_databases(
                    types.SimpleNamespace(outdir=abspath, cache_dir=cache_dir)
                )

            if i == 1:
                logger.info(f"[{mutation}] Embedding input sequences (ESM2-650M)")
                t0 = time.time()
                await tasks.embed(
                    name=f"{name}_foldtune_input",
                    GPUs=GPUs,
                    seed=args.RNG_seed,
                    outdir=abspath,
                    query=query,
                )
                logger.info(f"[{mutation}] Embed done ({time.time() - t0:.1f}s)")

                if not args.fast_folding:
                    logger.info(f"[{mutation}] Folding input sequences (ESMFold)")
                    t0 = time.time()
                    await tasks.fold(
                        name=f"{name}_foldtune_input",
                        GPUs=GPUs,
                        seed=args.RNG_seed,
                        outdir=f"{abspath}/{name}_foldtune_input_structs",
                        query=query,
                        batch_size=args.fold_batch_size,
                    )
                    logger.info(f"[{mutation}] Fold done ({time.time() - t0:.1f}s)")

                stats_tsv = os.path.join(abspath, f"{name}_seqkit_stats.tsv")
                await tasks.seqkit_stats(input_fasta=query, out_tsv=stats_tsv)
                df_stats = pd.read_csv(stats_tsv, sep="\t")
                median = df_stats.Q2.values
                logger.info(f"[{mutation}] Sequence length median={median[0]}  (from {query})")

            # ── DES ────────────────────────────────────────────────────────────
            de_ref_fasta = query if i == 1 else output_fasta
            fixed_len = int(median[0])
            des_fasta = os.path.join(abspath, f"{name}_round{i}_des.fasta")
            ori_ref = query
            ori_ref_structs = os.path.join(abspath, f"{name}_foldtune_input_structs")

            logger.info(
                f"[{mutation}] DES start (ft_round={i}  des_rounds={args.des_rounds}"
                f"  batch={args.des_batch_size}  n_seq={args.des_num_sequences}"
                f"  ref={de_ref_fasta})"
            )
            t0 = time.time()
            await tasks.run_des(
                de_ref_fasta=de_ref_fasta,
                fixed_len=fixed_len,
                des_fasta=des_fasta,
                ori_ref=ori_ref,
                ori_ref_structs=ori_ref_structs,
                round_i=i,
                fast_folding=args.fast_folding,
                prostt5_weights_path=prostt5_weights_path if args.fast_folding else None,
                fold_batch_size=args.fold_batch_size,
                gpus=GPUs,
                rng_seed=args.RNG_seed,
                num_mutations=args.num_mutations,
                des_rounds=args.des_rounds,
                des_batch_size=args.des_batch_size,
                des_num_sequences=args.des_num_sequences,
            )

            logger.info(f"[{mutation}] DES done ({time.time() - t0:.1f}s) → {des_fasta}")

            # ── Merge / truncate / clean generated FASTAs ──────────────────────
            gen_files = [
                f
                for f in os.listdir(abspath)
                if f.startswith(f"{name}_round{i}") and f.endswith(".fasta")
            ]
            merged = os.path.join(abspath, f"{name}_foldtune_generated_sequences_round{i}.fasta")
            with open(merged, "w") as outfile:
                for fname in gen_files:
                    with open(os.path.join(abspath, fname)) as infile:
                        for line in infile:
                            if line.strip():
                                outfile.write(line)
                    outfile.write("\n")

            truncate_seqs(merged, int(median[0]))
            remove_invalid_seqs_aa(
                os.path.join(
                    abspath, f"truncated_{name}_foldtune_generated_sequences_round{i}.fasta"
                )
            )
            cleaned = os.path.join(
                abspath, f"cleaned_truncated_{name}_foldtune_generated_sequences_round{i}.fasta"
            )
            n_cleaned = sum(1 for l in open(cleaned) if l.startswith(">"))
            logger.info(f"[{mutation}] Merged/cleaned → {n_cleaned} seqs  ({cleaned})")

            # ── Embed generated sequences ──────────────────────────────────────
            logger.info(f"[{mutation}] Embedding {n_cleaned} generated sequences (ESM2-650M)")
            t0 = time.time()
            await tasks.embed(
                name=f"{name}_round{i}",
                GPUs=GPUs,
                seed=args.RNG_seed,
                outdir=abspath,
                query=cleaned,
            )

            logger.info(f"[{mutation}] Embed done ({time.time() - t0:.1f}s)")
            input_embs = (
                os.path.join(abspath, f"{name}_foldtune_input_esm2_t33_650M_AVG.csv")
                if i == 1
                else os.path.join(
                    abspath, f"{name}_foldtune_most-distant_round{i - 1}_esm2_t33_650M_AVG.csv"
                )
            )
            generated_embs = os.path.join(abspath, f"{name}_round{i}_esm2_t33_650M_AVG.csv")
            input_df = pd.read_csv(input_embs)
            test_df = pd.read_csv(generated_embs)

            # ── Fold generated sequences (slow mode only) ──────────────────────
            if not args.fast_folding:
                logger.info(f"[{mutation}] Folding generated sequences (ESMFold)")
                t0 = time.time()
                await tasks.fold(
                    name=f"{name}_round{i}",
                    GPUs=GPUs,
                    seed=args.RNG_seed,
                    outdir=f"{abspath}/{name}_foldtune_generated_structs_round{i}",
                    query=cleaned,
                    batch_size=args.fold_batch_size,
                )
                logger.info(f"[{mutation}] Fold done ({time.time() - t0:.1f}s)")

            # ── Foldseek structural similarity ─────────────────────────────────
            logger.info(f"[{mutation}] Foldseek structural similarity (round {i})")
            foldseek_tsv = os.path.join(abspath, f"{name}_foldtune_foldseek_round{i}.tsv")
            t0 = time.time()

            if args.fast_folding:
                gen_db = f"{abspath}/{name}_foldtune_generated_sequences_round{i}_db"
                logger.info(f"[{mutation}] foldseek createdb (generated seqs) → {gen_db}")
                await tasks.foldseek_createdb(
                    fasta=cleaned,
                    db_path=gen_db,
                    prostt5_model=prostt5_weights_path,
                )
                if i == 1:
                    input_db = f"{abspath}/{name}_foldtune_input_db"
                    logger.info(f"[{mutation}] foldseek createdb (input seqs) → {input_db}")
                    await tasks.foldseek_createdb(
                        fasta=query,
                        db_path=input_db,
                        prostt5_model=prostt5_weights_path,
                    )
                logger.info(f"[{mutation}] foldseek easy-search → {foldseek_tsv}")
                await tasks.foldseek_search(
                    query_db=gen_db,
                    target_db=input_db,
                    tsv_out=foldseek_tsv,
                    tmp_dir=f"{abspath}/tmp_round{i}",
                )
            else:
                logger.info(f"[{mutation}] foldseek easy-search (struct) → {foldseek_tsv}")
                await tasks.foldseek_search(
                    query_db=f"{abspath}/{name}_foldtune_generated_structs_round{i}/",
                    target_db=f"{abspath}/{name}_foldtune_input_structs/",
                    tsv_out=foldseek_tsv,
                    tmp_dir=f"{abspath}/tmp_round{i}",
                    extra_flags='--alignment-type 1 --format-output "query,target,fident,bits,alntmscore"',
                )
            logger.info(f"[{mutation}] Foldseek done ({time.time() - t0:.1f}s)")

            highest_avg_score_by_query(foldseek_tsv, cleaned, args)

            # ── Embedding distance ranking ─────────────────────────────────────
            test_embeddings = test_df.iloc[:, :-1].to_numpy()
            input_embeddings = input_df.iloc[:, :-1].to_numpy()
            labels = test_df.iloc[:, -1].to_numpy()

            most_distant_indices = compute_average_rank_without_df(
                input_embeddings, test_embeddings, top_k=args.topk
            )
            most_distant_embeddings = test_df.iloc[most_distant_indices]
            most_distant_labels = [labels[idx] for idx in most_distant_indices]
            logger.info(
                f"[{mutation}] Selected top-{len(most_distant_labels)} most-distant sequences"
            )

            most_distant_embeddings.to_csv(
                os.path.join(
                    abspath, f"{name}_foldtune_most-distant_round{i}_esm2_t33_650M_AVG.csv"
                ),
                index=False,
            )
            labels_file = os.path.join(abspath, f"{name}_foldtune_most-distant_round{i}_labels.txt")
            with open(labels_file, "w") as f:
                f.write("\n".join(most_distant_labels))

            # ── seqkit grep → most-distant FASTA ──────────────────────────────
            output_fasta = os.path.join(abspath, f"{name}_foldtune_most-distant_round{i}.fasta")
            await tasks.seqkit_grep(
                pattern_file=labels_file,
                input_fasta=cleaned,
                output_fasta=output_fasta,
            )

            # ── Save per-sequence metrics ──────────────────────────────────────
            tmp_input_df = pd.read_csv(
                os.path.join(abspath, f"{name}_foldtune_input_esm2_t33_650M_AVG.csv")
            )
            wt_seq = str(next(SeqIO.parse(args.wt_query, "fasta")).seq)
            save_round_records(
                round_i=i,
                fasta_file=cleaned,
                embedding_csv=generated_embs,
                foldseek_tsv=foldseek_tsv,
                input_embeddings=tmp_input_df.iloc[:, :-1].to_numpy(),
                wt_seq=wt_seq,
                out_path=os.path.join(abspath, f"{name}_round{i}_seq_records.csv"),
            )

            # ── Evaluation ────────────────────────────────────────────────────
            logger.info(
                f"[{mutation}] Evaluating round {i} selection (foldseek on {len(most_distant_labels)} selected seqs)"
            )
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            eval_prefix = f"{name}_round{i}_eval_{timestamp}"
            eval_tsv = os.path.join(abspath, f"{eval_prefix}.tsv")

            if not args.fast_folding:
                logger.info(f"[{mutation}] Folding selected sequences for eval (ESMFold)")
                await tasks.fold(
                    name=f"{name}_round{i}",
                    GPUs=GPUs,
                    seed=args.RNG_seed,
                    outdir=os.path.join(abspath, f"{eval_prefix}_structs"),
                    query=output_fasta,
                    batch_size=args.fold_batch_size,
                )

            eval_ref_db = os.path.join(abspath, f"{eval_prefix}_ref_db")
            eval_gen_db = os.path.join(abspath, f"{eval_prefix}_gen_db")

            if args.fast_folding:
                await tasks.foldseek_createdb(
                    fasta=query,
                    db_path=eval_ref_db,
                    prostt5_model=prostt5_weights_path,
                )
                await tasks.foldseek_createdb(
                    fasta=output_fasta,
                    db_path=eval_gen_db,
                    prostt5_model=prostt5_weights_path,
                )
                await tasks.foldseek_search(
                    query_db=eval_gen_db,
                    target_db=eval_ref_db,
                    tsv_out=eval_tsv,
                    tmp_dir=f"{abspath}/tmp_eval{i}",
                    extra_flags="--format-output 'query,target,bits'",
                )
                df_eval = pd.read_csv(eval_tsv, sep="\t", names=["query", "target", "bits"])
                struct_key = "bits"
            else:
                eval_ref_structs = os.path.join(abspath, f"{name}_foldtune_input_structs")
                eval_gen_structs = os.path.join(abspath, f"{eval_prefix}_structs")
                await tasks.foldseek_createdb(fasta=eval_ref_structs, db_path=eval_ref_db)
                await tasks.foldseek_createdb(fasta=eval_gen_structs, db_path=eval_gen_db)
                await tasks.foldseek_search(
                    query_db=eval_gen_db,
                    target_db=eval_ref_db,
                    tsv_out=eval_tsv,
                    tmp_dir=f"{abspath}/tmp_eval{i}",
                    extra_flags="--alignment-type 1 --format-output 'query,target,alntmscore'",
                )
                df_eval = pd.read_csv(eval_tsv, sep="\t", names=["query", "target", "alntmscore"])
                struct_key = "alntmscore"

            struct_stats = {
                "min": float(df_eval[struct_key].min()),
                "max": float(df_eval[struct_key].max()),
                "mean": float(df_eval[struct_key].mean()),
            }

            selected_embs = most_distant_embeddings.iloc[:, :-1].to_numpy()
            cos_dists = cosine_distances(selected_embs, input_embeddings).min(axis=1)
            l2_dists = euclidean_distances(selected_embs, input_embeddings).min(axis=1)
            embed_stats = {
                "cosine_min": float(cos_dists.min()),
                "cosine_max": float(cos_dists.max()),
                "cosine_mean": float(cos_dists.mean()),
                "l2_min": float(l2_dists.min()),
                "l2_max": float(l2_dists.max()),
                "l2_mean": float(l2_dists.mean()),
            }

            eval_result = {"round": i, "embedding": embed_stats, "structure": struct_stats}
            with open(os.path.join(abspath, f"{eval_prefix}.json"), "w") as f:
                json.dump(eval_result, f, indent=2)
            logger.info(
                f"[{mutation}] Round {i} eval done → {eval_prefix}.json  "
                f"struct({struct_key}): min={struct_stats['min']:.4f} mean={struct_stats['mean']:.4f} max={struct_stats['max']:.4f}  "
                f"cosine_dist: min={embed_stats['cosine_min']:.4f} mean={embed_stats['cosine_mean']:.4f}"
            )
            logger.info(
                f"[{mutation}] ── Round {i} complete  "
                f"elapsed={time.time() - t_round_start:.1f}s  "
                f"total={time.time() - t_workflow_start:.1f}s ──"
            )

            for db_path in [eval_gen_db, eval_ref_db]:
                shutil.rmtree(db_path, ignore_errors=True)
                for ext in [".dbtype", ".lookup", ".source", "_h", "_ss", "_ca"]:
                    p = db_path + ext
                    if os.path.exists(p):
                        os.remove(p)

        logger.info(f"[{mutation}] SGDES complete → {abspath}")

        if self.collect_nvml_telemetry:
            await tasks.stop_node_telemetry(outdir=self.telemetry_dir)

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _run_one(self, mutation: str, idx: int) -> None:
        if self.policies is not None:
            policy = self.policies[idx % len(self.policies)]
            # In multi-node Dragon, each worker is routed to a specific node+GPU via
            # policy. The GPU index in CUDA_VISIBLE_DEVICES must match the per-node
            # local GPU id (gpu_affinity), not the global idx % total_gpus calculation
            # (which would exceed the number of GPUs on any single node).
            cuda_device = policy.gpu_affinity[0] if policy.gpu_affinity else 0
            logger.info(f"[{mutation}] Starting (policy={policy})")
        else:
            policy = None
            cuda_device = idx % self.total_gpus
            logger.info(f"[{mutation}] Starting")

        try:
            await self._sgdes_async(mutation, cuda_device, policy)
        except Exception as e:
            import traceback

            logger.error(f"[{mutation}] FAILED: {e}\n{traceback.format_exc()}")
            raise

    async def run(self) -> None:
        """Run all mutations in parallel; asyncflow schedules task concurrency."""
        tasks = [self._run_one(mutation, idx) for idx, mutation in enumerate(self.mutations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [self.mutations[i] for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed:
            logger.error(f"Failed mutations: {failed}")
