"""
SGDESWorkflow_asyncflow — port of SGDESWorkflow that dispatches shell commands
through asyncflow.executable_task instead of blocking subprocess.run calls.

Key differences from sgdes_workflow.py:
- Requires asyncflow (radical.asyncflow.WorkflowEngine) passed to __init__
- _sgdes_blocking → _sgdes_async (truly async, no asyncio.to_thread)
- subprocess.run → @flow.executable_task functions that return command strings
- seqkit stats redirects stdout to a TSV file via shell > and is read back
  by Python after the task completes
- DES logic in module-level _run_des(), called via asyncio.to_thread
task_description fields used:
    shell: True   — commands are run via a shell so operators (&&, >, |) work
    gpus_per_rank — informational for Dragon backend; ignored by concurrent backend
    cores_per_rank — informational; not enforced by concurrent backend

GPU / env isolation is left to the caller: set CUDA_VISIBLE_DEVICES in os.environ
before spawning the workflow (e.g. in the wrapper's run() method).
"""

import asyncio
import json
import os
import shutil
import sys
import time
import types
from datetime import datetime
from pathlib import Path

# Must be set before TensorFlow initialises its GPU context.
# amortized_bo (DES) uses TF v1, which by default pre-allocates ~90 % of GPU
# memory on first use and never releases it, starving subsequent trill embed
# subprocess calls.  Memory-growth mode allocates only what is actually needed.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# jaxlib on this system was compiled with cuDNN 8.9.6 but only 8.0.4 is installed;
# force JAX to use the CPU backend to avoid a cuDNN version mismatch error.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
from Bio import SeqIO
from src.utils.logger import Logger
logger = Logger(name='sgdes', use_colors=True)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

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

from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node

def make_gpu_policies(nprocs: int):
    """Round-robin GPU assignment across all Dragon nodes."""
    all_gpus = [
        (node.hostname, gpu_id)
        for huid in System().nodes
        for node in [Node(huid)]
        for gpu_id in node.gpus
    ]
    return [
        Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=all_gpus[i % len(all_gpus)][0],
            gpu_affinity=[all_gpus[i % len(all_gpus)][1]],
        )
        for i in range(nprocs)
    ]

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
        use_gpu=False,
        prostt5_model_path=prostt5_weights_path if fast_folding else None,
        fold_batch_size=fold_batch_size,
        gpus=gpus,
    )
    print(f"[DES ft_round={round_i}] Input DB ready ({time.time()-t0:.1f}s)")

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
        problem, solver,
        num_rounds=int(des_rounds),
        batch_size=int(des_batch_size),
        initial_population=init_population,
        callbacks=[_des_step_callback],
    )

    best = population.best_n(int(des_num_sequences), discard_duplicates=True)
    print(
        f"[DES ft_round={round_i}] Done — best_n={len(best)}  "
        f"total={time.time()-t0_total:.1f}s  → {des_fasta}"
    )
    des_arr = (
        np.array([s.structure for s in best]) if len(best) > 0
        else cand_init[:min(int(des_num_sequences), len(cand_init))]
    )
    _int_array_to_fasta(des_arr, des_fasta, prefix=f"des_r{round_i}")


class SGDESWorkflow:
    def __init__(self, config: dict, asyncflow=None):
        cfg = config

        # ── Paths ─────────────────────────────────────────────────────────────
        self.base_outdir = os.path.abspath(cfg["outdir"])
        self.query_dir   = str(cfg["query_dir"])
        self.wt_query    = str(cfg["wt_query"])
        self.mutations   = list(cfg["mutations"])

        # ── Execution ─────────────────────────────────────────────────────────
        self.total_gpus      = int(cfg.get("total_gpus", 1))
        #self.total_cpus      = int(cfg.get("total_cpus", os.cpu_count()))
        self.rng_seed        = int(cfg.get("RNG_seed", 42))
        self.foldtune_rounds = int(cfg.get("foldtune_rounds", 3))
        self.fast_folding    = bool(cfg.get("fast_folding", True))
        self.fold_batch_size = int(cfg.get("fold_batch_size", 1))

        # ── DES hyperparameters ───────────────────────────────────────────────
        self.des_rounds        = int(cfg.get("des_rounds", 3))
        self.des_batch_size    = int(cfg.get("des_batch_size", 500))
        self.des_num_sequences = int(cfg.get("des_num_sequences", 200))
        self.num_mutations_des = int(cfg.get("num_mutations", 1))
        self.topk              = int(cfg.get("topk", 20))

        if asyncflow is None:
            raise ValueError(
                "SGDESWorkflow_asyncflow requires asyncflow= argument. "
                "Use sgdes_workflow.SGDESWorkflow for the subprocess-based variant."
            )
        self.asyncflow = asyncflow

        os.makedirs(self.base_outdir, exist_ok=True)

        self.policies = make_gpu_policies(nprocs=len(self.mutations))

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def _register_tasks(self, cuda_device: int, policy: Policy):
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

        _TD_GPU = {
            "process_template": {
                "policy": policy,
            },
        }

        _TD_CPU = {
            "process_template": {
                "policy": Policy(),
            },
        }

        # ── trill embed ────────────────────────────────────────────────────────
        @flow.executable_task
        async def embed(task_description=_TD_GPU, **kwargs):
            """Run trill embed esm2_t33_650M.
            kwargs: name, GPUs, seed, outdir, query
            """
            name   = kwargs["name"]
            GPUs   = kwargs["GPUs"]
            seed   = kwargs["seed"]
            outdir = kwargs["outdir"]
            query  = kwargs["query"]
            return (
                f"trill {name} {GPUs} --RNG_seed {seed} --outdir {outdir} "
                f"embed esm2_t33_650M {query} --avg"
            )

        # ── trill fold (ESMFold, slow path only) ───────────────────────────────
        @flow.executable_task
        async def fold(task_description=_TD_GPU, **kwargs):
            """Run trill fold ESMFold.
            kwargs: name, GPUs, seed, outdir, query, batch_size
            """
            name       = kwargs["name"]
            GPUs       = kwargs["GPUs"]
            seed       = kwargs["seed"]
            outdir     = kwargs["outdir"]
            query      = kwargs["query"]
            batch_size = kwargs["batch_size"]
            return (
                f"trill {name} {GPUs} --RNG_seed {seed} --outdir {outdir} "
                f"fold ESMFold {query} --batch_size {batch_size}"
            )

        # ── foldseek createdb ─────────────────────────────────────────────────
        # NOTE: implemented as function_task (not executable_task) so that foldseek
        # runs as a grandchild subprocess of the Dragon worker process.  When foldseek
        # is launched as a *direct* Dragon executable_task subprocess the ggml-CUDA
        # backend initialises inside Dragon's CUDA IPC context, which causes a silent
        # failure: the _ss (3Di secondary structure) file is written as empty even
        # without --gpu 1.  Running via subprocess.run() avoids that context entirely.
        @flow.function_task
        async def foldseek_createdb(**kwargs):
            """Run foldseek createdb via subprocess.
            kwargs: fasta, db_path, prostt5_model (optional)
            Note: --gpu 1 omitted — foldseek ggml-CUDA crashes in Dragon subprocess context.
            """
            import subprocess as _sp
            fasta         = kwargs["fasta"]
            db_path       = kwargs["db_path"]
            prostt5_model = kwargs.get("prostt5_model", "")
            model_flag    = f"--prostt5-model {prostt5_model}" if prostt5_model else ""
            cmd = f"foldseek createdb {fasta} {db_path} {model_flag}".strip()
            print(f"[foldseek_createdb] cmd: {cmd}", flush=True)
            result = _sp.run(cmd, shell=True, capture_output=True, text=True)
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
        async def foldseek_search(**kwargs):
            """Run foldseek easy-search via subprocess.
            kwargs: query_db, target_db, tsv_out, tmp_dir, extra_flags (optional)
            """
            import subprocess as _sp
            query_db    = kwargs["query_db"]
            target_db   = kwargs["target_db"]
            tsv_out     = kwargs["tsv_out"]
            tmp_dir     = kwargs["tmp_dir"]
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
        @flow.executable_task
        async def seqkit_grep(task_description=_TD_CPU, **kwargs):
            """Run seqkit grep, redirecting stdout to output_fasta.
            kwargs: pattern_file, input_fasta, output_fasta
            """
            pattern_file  = kwargs["pattern_file"]
            input_fasta   = kwargs["input_fasta"]
            output_fasta  = kwargs["output_fasta"]
            return (
                f"bash -c 'seqkit grep "
                f"--pattern-file {pattern_file} {input_fasta} > {output_fasta}'"
            )

        # ── seqkit stats → TSV file ───────────────────────────────────────────
        @flow.executable_task
        async def seqkit_stats(task_description=_TD_CPU, **kwargs):
            """Run seqkit stats -a -T, redirecting stdout to out_tsv.
            kwargs: input_fasta, out_tsv
            """
            input_fasta = kwargs["input_fasta"]
            out_tsv     = kwargs["out_tsv"]
            return f"bash -c 'seqkit stats -a -T {input_fasta} > {out_tsv}'"

        @flow.function_task
        async def run_des(**kwargs):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
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

        return types.SimpleNamespace(
            embed=embed,
            fold=fold,
            foldseek_createdb=foldseek_createdb,
            foldseek_search=foldseek_search,
            seqkit_grep=seqkit_grep,
            seqkit_stats=seqkit_stats,
            run_des=run_des,
        )

    # ------------------------------------------------------------------
    # Per-mutation async implementation
    # ------------------------------------------------------------------

    async def _sgdes_async(self, mutation: str, cuda_device: int, policy: Policy) -> None:
        """Async port of _sgdes_blocking; all commands run via executable_task."""
        tasks = self._register_tasks(cuda_device, policy)

        abspath = os.path.join(self.base_outdir, f"{mutation}_wd")
        os.makedirs(abspath, exist_ok=True)

        name    = f"{mutation}_run"
        query   = os.path.join(self.query_dir, f"mayv_{mutation}.fasta")
        GPUs    = "1"

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
            "experiment_name":        name,
            "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "foldtune_rounds":        args.foldtune_rounds,
            "query_file":             args.query,
            "wt_query_file":          args.wt_query,
            "output_dir":             abspath,
            "GPUs":                   args.GPUs,
            "RNG_seed":               args.RNG_seed,
            "fast_folding":           args.fast_folding,
            "fold_batch_size":        args.fold_batch_size,
            "finetune_batch_size":    args.finetune_batch_size,
            "finetune_strategy":      args.finetune_strategy,
            "lang_gen_batch_size":    args.lang_gen_batch_size,
            "num_des_round":          args.des_rounds,
            "num_seq_each_des_round": args.des_batch_size,
            "num_selected_seq":       args.des_num_sequences,
            "topk":                   args.topk,
            "num_mutations":          args.num_mutations,
            "des_batch_size":         args.des_batch_size,
        }
        with open(os.path.join(abspath, f"{name}_experiment_config.json"), "w") as f:
            json.dump(exp_config, f, indent=2)
        logger.info(f"[{mutation}] Config saved")

        prostt5_weights_path = None
        output_fasta         = None
        median               = None

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
                logger.info(f"[{mutation}] Embed done ({time.time()-t0:.1f}s)")

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
                    logger.info(f"[{mutation}] Fold done ({time.time()-t0:.1f}s)")

                stats_tsv = os.path.join(abspath, f"{name}_seqkit_stats.tsv")
                await tasks.seqkit_stats(input_fasta=query, out_tsv=stats_tsv)
                df_stats = pd.read_csv(stats_tsv, sep="\t")
                median   = df_stats.Q2.values
                logger.info(f"[{mutation}] Sequence length median={median[0]}  (from {query})")

            # ── DES ────────────────────────────────────────────────────────────
            de_ref_fasta    = query if i == 1 else output_fasta
            fixed_len       = int(median[0])
            des_fasta       = os.path.join(abspath, f"{name}_round{i}_des.fasta")
            ori_ref         = query
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

            logger.info(f"[{mutation}] DES done ({time.time()-t0:.1f}s) → {des_fasta}")

            # ── Merge / truncate / clean generated FASTAs ──────────────────────
            gen_files = [
                f for f in os.listdir(abspath)
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
                os.path.join(abspath, f"truncated_{name}_foldtune_generated_sequences_round{i}.fasta")
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

            logger.info(f"[{mutation}] Embed done ({time.time()-t0:.1f}s)")
            input_embs = (
                os.path.join(abspath, f"{name}_foldtune_input_esm2_t33_650M_AVG.csv") if i == 1
                else os.path.join(abspath, f"{name}_foldtune_most-distant_round{i-1}_esm2_t33_650M_AVG.csv")
            )
            generated_embs = os.path.join(abspath, f"{name}_round{i}_esm2_t33_650M_AVG.csv")
            input_df       = pd.read_csv(input_embs)
            test_df        = pd.read_csv(generated_embs)

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
                logger.info(f"[{mutation}] Fold done ({time.time()-t0:.1f}s)")

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
                    tmp_dir=f"tmp_round{i}",
                )
            else:
                logger.info(f"[{mutation}] foldseek easy-search (struct) → {foldseek_tsv}")
                await tasks.foldseek_search(
                    query_db=f"{abspath}/{name}_foldtune_generated_structs_round{i}/",
                    target_db=f"{abspath}/{name}_foldtune_input_structs/",
                    tsv_out=foldseek_tsv,
                    tmp_dir=f"tmp_round{i}",
                    extra_flags='--alignment-type 1 --format-output "query,target,fident,bits,alntmscore"',
                )
            logger.info(f"[{mutation}] Foldseek done ({time.time()-t0:.1f}s)")

            highest_avg_score_by_query(foldseek_tsv, cleaned, args)

            # ── Embedding distance ranking ─────────────────────────────────────
            test_embeddings  = test_df.iloc[:, :-1].to_numpy()
            input_embeddings = input_df.iloc[:, :-1].to_numpy()
            labels           = test_df.iloc[:, -1].to_numpy()

            most_distant_indices    = compute_average_rank_without_df(
                input_embeddings, test_embeddings, top_k=args.topk
            )
            most_distant_embeddings = test_df.iloc[most_distant_indices]
            most_distant_labels     = [labels[idx] for idx in most_distant_indices]
            logger.info(f"[{mutation}] Selected top-{len(most_distant_labels)} most-distant sequences")

            most_distant_embeddings.to_csv(
                os.path.join(abspath, f"{name}_foldtune_most-distant_round{i}_esm2_t33_650M_AVG.csv"),
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
            logger.info(f"[{mutation}] Evaluating round {i} selection (foldseek on {len(most_distant_labels)} selected seqs)")
            timestamp   = time.strftime("%Y%m%d-%H%M%S")
            eval_prefix = f"{name}_round{i}_eval_{timestamp}"
            eval_tsv    = os.path.join(abspath, f"{eval_prefix}.tsv")

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
                    tmp_dir=f"tmp_eval{i}",
                    extra_flags="--format-output 'query,target,bits'",
                )
                df_eval    = pd.read_csv(eval_tsv, sep="\t", names=["query", "target", "bits"])
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
                    tmp_dir=f"tmp_eval{i}",
                    extra_flags="--alignment-type 1 --format-output 'query,target,alntmscore'",
                )
                df_eval    = pd.read_csv(eval_tsv, sep="\t", names=["query", "target", "alntmscore"])
                struct_key = "alntmscore"

            struct_stats = {
                "min":  float(df_eval[struct_key].min()),
                "max":  float(df_eval[struct_key].max()),
                "mean": float(df_eval[struct_key].mean()),
            }

            selected_embs = most_distant_embeddings.iloc[:, :-1].to_numpy()
            cos_dists     = cosine_distances(selected_embs, input_embeddings).min(axis=1)
            l2_dists      = euclidean_distances(selected_embs, input_embeddings).min(axis=1)
            embed_stats   = {
                "cosine_min":  float(cos_dists.min()),
                "cosine_max":  float(cos_dists.max()),
                "cosine_mean": float(cos_dists.mean()),
                "l2_min":      float(l2_dists.min()),
                "l2_max":      float(l2_dists.max()),
                "l2_mean":     float(l2_dists.mean()),
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
                f"elapsed={time.time()-t_round_start:.1f}s  "
                f"total={time.time()-t_workflow_start:.1f}s ──"
            )

            for db_path in [eval_gen_db, eval_ref_db]:
                shutil.rmtree(db_path, ignore_errors=True)
                for ext in [".dbtype", ".lookup", ".source", "_h", "_ss", "_ca"]:
                    p = db_path + ext
                    if os.path.exists(p):
                        os.remove(p)

        logger.info(f"[{mutation}] SGDES complete → {abspath}")

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _run_one(self, mutation: str, idx: int) -> None:
        cuda_device = idx % self.total_gpus
        policy = self.policies[idx % len(self.policies)]
        logger.info(f"[{mutation}] Starting (policy={policy})")

        try:
            await self._sgdes_async(mutation, cuda_device, policy)
        except Exception as e:
            logger.error(f"[{mutation}] FAILED: {e}")
            raise

    async def run(self) -> None:
        """Run all mutations in parallel; asyncflow schedules task concurrency."""
        tasks = [
            self._run_one(mutation, idx)
            for idx, mutation in enumerate(self.mutations)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed  = [
            self.mutations[i] for i, r in enumerate(results)
            if isinstance(r, Exception)
        ]
        if failed:
            logger.error(f"Failed mutations: {failed}")
