"""
SGDESWorkflow — asyncflow-based SGDES workflow.

Dispatches all shell commands through asyncflow tasks instead of blocking
subprocess.run calls, enabling concurrent execution of multiple mutations.

Task types
----------
executable_task — trill embed/fold, foldseek createdb, seqkit grep/stats;
                  Dragon launches each as a subprocess with GPU affinity and
                  HOST_NAME placement via task_description.
function_task   — foldseek_search only; runs via subprocess.run to avoid the
                  ggml-CUDA context conflict that occurs when foldseek easy-search
                  runs as a direct Dragon executable_task subprocess.
_run_des        — plain async method on SGDESWorkflow; orchestrates the DES
                  loop (solver.propose → foldseek scoring → population.add_samples)
                  using the registered tasks.

All environment variables (CUDA_HOME, SGDES_DIR, SPHERICAL_DIR, JAX_PLATFORMS,
TF_FORCE_GPU_ALLOW_GROWTH) are set in the sbatch script.
"""

import asyncio
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types
import uuid
import warnings
from datetime import datetime

# amortized_bo imports JAX at module level, making this process multithreaded.
# Dragon then uses os.fork() to spawn executable_task subprocesses (trill embed),
# which triggers Python's fork-after-threads warning.  The warning is harmless —
# trill runs in its own fresh subprocess and completes normally.
warnings.filterwarnings(
    "ignore",
    message="os.fork\\(\\) was called.*JAX is multithreaded",
    category=RuntimeWarning,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio import SeqIO  # noqa: E402

from src.utils.logger import Logger  # noqa: E402

logger = Logger(name="sgdes", use_colors=True)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances  # noqa: E402

# amortized_bo uses absolute imports (e.g. `from amortized_bo import data`),
# so its parent directory must be on sys.path.
_SGDES_DIR = os.environ.get("SGDES_DIR", "")
_ABO_PARENT = os.path.join(_SGDES_DIR, "trill/utils/abo") if _SGDES_DIR else ""
if _ABO_PARENT and _ABO_PARENT not in sys.path:
    sys.path.insert(0, _ABO_PARENT)

from trill.utils.abo.amortized_bo import data, domains  # noqa: E402
from trill.utils.abo.amortized_bo.deep_evolution_solver import MutationPredictorSolver  # noqa: E402
from trill.utils.fasta_files import remove_invalid_seqs_aa, truncate_seqs  # noqa: E402
from trill.utils.foldseek_utils import run_foldseek_databases  # noqa: E402
from trill.utils.sgdes import (  # noqa: E402
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


def _parse_foldseek_avg(tsv_path, fast_mode=True):
    if fast_mode:
        cols = [
            "query",
            "target",
            "fident",
            "alnlen",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bits",
        ]
        df = pd.read_csv(tsv_path, sep="\t", names=cols)
        score_col = "bits"
    else:
        cols = ["query", "target", "fident", "bits", "alntmscore"]
        df = pd.read_csv(tsv_path, sep="\t", names=cols)
        score_col = "alntmscore"
    avg = df.groupby("query")[score_col].mean().rename("avg_score").reset_index()
    return avg


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
        # None → _run_des falls back to Lustre (os.path.dirname(des_fasta)).
        # Set to a /tmp path by run_workflow.py on single-node runs.
        self.des_workdir_base = cfg.get("des_workdir_base", None)

        if asyncflow is None:
            raise ValueError(
                "SGDESWorkflow_asyncflow requires asyncflow= argument. "
                "Use sgdes_workflow.SGDESWorkflow for the subprocess-based variant."
            )
        self.asyncflow = asyncflow

        os.makedirs(self.base_outdir, exist_ok=True)

        self.policies = policies

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def _register_tasks(self, policy):
        """Register asyncflow tasks for one mutation slot.

        Each task closes over `policy` so all commands for the same mutation
        target the same node/GPU.
        """
        flow = self.asyncflow

        _TD_GPU = (  # noqa: N806
            {
                "process_template": {
                    "policy": policy,
                },
            }
            if policy is not None
            else {}
        )

        # Host-only policy: same placement + host_name as _TD_GPU, no gpu_affinity
        if policy is not None and Policy is not None:
            _host_policy = Policy(
                placement=policy.placement,
                host_name=policy.host_name,
            )
            _TD_HOST = {"process_template": {"policy": _host_policy}}  # noqa: N806
        else:
            _host_policy = None
            _TD_HOST = {}  # noqa: N806

        # ── trill embed ────────────────────────────────────────────────────────
        @flow.executable_task
        async def embed(task_description=_TD_GPU, **kwargs):
            """Run trill embed esm2_t33_650M as an executable_task.
            kwargs: name, GPUs, seed, outdir, query
            """
            name = kwargs["name"]
            gpus = kwargs["GPUs"]
            seed = kwargs["seed"]
            outdir = kwargs["outdir"]
            query = kwargs["query"]
            cmd = (
                f"trill {name} {gpus} --RNG_seed {seed} --outdir {outdir} "
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
            gpus = kwargs["GPUs"]
            seed = kwargs["seed"]
            outdir = kwargs["outdir"]
            query = kwargs["query"]
            batch_size = kwargs["batch_size"]
            cmd = (
                f"trill {name} {gpus} --RNG_seed {seed} --outdir {outdir} "
                f"fold ESMFold {query} --batch_size {batch_size}"
            )
            print(f"[fold] cmd: {cmd}", flush=True)
            return cmd

        # ── foldseek createdb ─────────────────────────────────────────────────
        @flow.executable_task
        async def foldseek_createdb(task_description=_TD_GPU, **kwargs):
            """Run foldseek createdb as an executable_task.
            kwargs: fasta, db_path, prostt5_model (optional)
            """
            fasta = kwargs["fasta"]
            db_path = kwargs["db_path"]
            prostt5_model = kwargs.get("prostt5_model", "")
            model_flag = f"--prostt5-model {prostt5_model}" if prostt5_model else ""
            gpu_flag = " --gpu 1" if prostt5_model else ""
            cmd = f"foldseek createdb {fasta} {db_path} {model_flag}{gpu_flag}".strip()
            print(f"[foldseek_createdb] cmd: {cmd}", flush=True)
            return cmd

        # ── foldseek easy-search ──────────────────────────────────────────────
        # NOTE: function_task (not executable_task) to avoid the ggml-CUDA context
        # conflict that causes foldseek easy-search to hang when launched as a direct
        # Dragon subprocess.  subprocess.run() sidesteps that context entirely.
        @flow.function_task
        async def foldseek_search(task_description=_TD_GPU, **kwargs):
            """Run foldseek easy-search via subprocess.run (function_task).
            kwargs: query_db, target_db, tsv_out, tmp_dir, extra_flags (optional)

            tmp_dir may be on node-local /tmp (when DES_TMPDIR is set); this task
            creates it on the worker so the head process doesn't need to.
            """
            import os as _os
            import subprocess as _sp

            query_db = kwargs["query_db"]
            target_db = kwargs["target_db"]
            tsv_out = kwargs["tsv_out"]
            tmp_dir = kwargs["tmp_dir"]
            extra_flags = kwargs.get("extra_flags", "")

            # Create tmp_dir here on the worker — it may be on node-local /tmp
            # which the head process cannot reach.
            _os.makedirs(tmp_dir, exist_ok=True)

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

        # ── seqkit grep → file (stdout redirected via ProcessTemplate) ───────
        @flow.executable_task
        async def seqkit_grep(task_description=_TD_HOST, **kwargs):
            """Run seqkit grep; Dragon writes stdout to output_fasta via task_description.
            kwargs: pattern_file, input_fasta
            """
            pattern_file = kwargs["pattern_file"]
            input_fasta = kwargs["input_fasta"]
            cmd = f"seqkit grep --pattern-file {pattern_file} {input_fasta}"
            print(f"[seqkit_grep] cmd: {cmd}", flush=True)
            return cmd

        # ── seqkit stats → returns TSV via stdout ─────────────────────────────
        @flow.executable_task
        async def seqkit_stats(task_description=_TD_HOST, **kwargs):
            """Run seqkit stats -a -T; stdout is returned by asyncflow as a string.
            kwargs: input_fasta
            """
            input_fasta = kwargs["input_fasta"]
            cmd = f"seqkit stats -a -T {input_fasta}"
            print(f"[seqkit_stats] cmd: {cmd}", flush=True)
            return cmd

        return types.SimpleNamespace(
            embed=embed,
            fold=fold,
            foldseek_createdb=foldseek_createdb,
            foldseek_search=foldseek_search,
            seqkit_grep=seqkit_grep,
            seqkit_stats=seqkit_stats,
            host_policy=_host_policy,
        )

    # ------------------------------------------------------------------
    # DES orchestration
    # ------------------------------------------------------------------

    async def _run_des(self, tasks, **kwargs) -> None:
        """Orchestrate one DES round using asyncflow tasks for each subprocess call.

        Manually steps through the DES loop (solver.propose → score via
        tasks.foldseek_createdb/foldseek_search → population.add_samples) instead of delegating to
        controller.run() + FoldseekSimilarityProblem.
        """
        de_ref_fasta = kwargs["de_ref_fasta"]
        fixed_len = kwargs["fixed_len"]
        des_fasta = kwargs["des_fasta"]
        ori_ref = kwargs["ori_ref"]
        ori_ref_structs = kwargs["ori_ref_structs"]
        round_i = kwargs["round_i"]
        fast_folding = kwargs["fast_folding"]
        prostt5_weights_path = kwargs["prostt5_weights_path"]
        fold_batch_size = kwargs["fold_batch_size"]
        gpus = kwargs["gpus"]
        rng_seed = kwargs["rng_seed"]
        num_mutations = kwargs["num_mutations"]
        des_rounds = kwargs["des_rounds"]
        des_batch_size = kwargs["des_batch_size"]
        des_num_sequences = kwargs["des_num_sequences"]

        ref_path = ori_ref if fast_folding else ori_ref_structs

        # des_workdir_base is set by run_workflow.py:
        #   single-node → DES_TMPDIR (/tmp): fast local I/O, ~10 s/createdb step
        #   multi-node  → None (Lustre):      shared across nodes, ~27-130 s/step
        # See run_workflow.py for the full rationale.
        _workdir_base = self.des_workdir_base if self.des_workdir_base is not None \
            else os.path.dirname(des_fasta)
        workdir = tempfile.mkdtemp(prefix="fsim_", dir=_workdir_base)

        # foldseek_search's internal scratch stays on node-local /tmp regardless.
        _des_tmp = os.environ.get("DES_TMPDIR", "/tmp")

        print(
            f"[DES ft_round={round_i}] workdir={workdir}  foldseek_tmp={_des_tmp}",
            flush=True,
        )
        t0_total = time.time()

        # ── Build input DB once ────────────────────────────────────────────────
        input_db = os.path.join(workdir, "input_db")
        print(f"[DES ft_round={round_i}] Building input DB from {ref_path}")
        t0 = time.time()
        await tasks.foldseek_createdb(
            fasta=ref_path,
            db_path=input_db,
            prostt5_model=prostt5_weights_path if fast_folding else "",
        )
        print(f"[DES ft_round={round_i}] Input DB ready ({time.time() - t0:.1f}s)")

        # ── Initialise solver ──────────────────────────────────────────────────
        domain = domains.FixedLengthDiscreteDomain(vocab_size=len(AA), length=fixed_len)

        def my_initializer(dom, batch_size, random_state):
            fasta_array = _fasta_to_numeric(ori_ref, fixed_len)
            n = len(fasta_array)
            assert fasta_array.shape[1] == fixed_len
            if n < batch_size:
                extra_idx = random_state.randint(0, n, size=batch_size - n)
                fasta_array = np.concatenate([fasta_array, fasta_array[extra_idx]])
            return fasta_array[:batch_size]

        solver = MutationPredictorSolver(domain=domain, random_state=int(rng_seed))
        solver.cfg.initialize_dataset_fn = my_initializer
        solver.cfg.num_mutations = num_mutations

        cand_init = _fasta_to_int_array(de_ref_fasta, fixed_len)
        print(f"[DES ft_round={round_i}] cand_init={len(cand_init)} seqs  fixed_len={fixed_len}")
        population = data.Population.from_arrays(
            structures=cand_init,
            rewards=np.zeros(len(cand_init)),
            batch_index=0,
        )

        _step_times = [time.time()]

        # ── Manual DES loop ────────────────────────────────────────────────────
        print(
            f"[DES ft_round={round_i}] Starting {des_rounds} DES steps  batch_size={des_batch_size}"
        )
        for step in range(int(des_rounds)):
            samples = solver.propose(int(des_batch_size), population.copy())
            if not isinstance(samples[0], data.Sample):
                samples = [data.Sample(structure=s) for s in samples]

            structures = [s.structure for s in samples]
            arr = np.array(structures, dtype=int)
            tag = str(uuid.uuid4())[:8]
            cand_fa = os.path.join(workdir, f"cands_{tag}.fa")
            headers = [f"cand_{tag}_{i}" for i in range(len(structures))]
            seqs = ["".join(_id2aa[int(x)] for x in row) for row in arr]
            with open(cand_fa, "w") as _f:
                for h, s in zip(headers, seqs, strict=False):
                    _f.write(f">{h}\n{s}\n")

            # Multi-node: workdir is on Lustre → cand_fa, out_tsv, cand_db all
            # visible to any node.  Single-node: workdir is on /tmp (fast I/O).
            out_tsv = os.path.join(workdir, f"res_{tag}.tsv")
            cand_db = os.path.join(workdir, f"cand_db_{tag}")
            # tmp_dir: foldseek_search creates it on the worker's local /tmp
            tmp_dir = os.path.join(_des_tmp, f"fsim_{tag}")

            if fast_folding:
                await tasks.foldseek_createdb(
                    fasta=cand_fa,
                    db_path=cand_db,
                    prostt5_model=prostt5_weights_path or "",
                )
                await tasks.foldseek_search(
                    query_db=cand_db,
                    target_db=input_db,
                    tsv_out=out_tsv,
                    tmp_dir=tmp_dir,
                    extra_flags="--gpu 1",
                )
                avg = _parse_foldseek_avg(out_tsv, fast_mode=True)
            else:
                # fold writes PDB files; head creates dir on Lustre so fold task
                # (executable_task on worker) can write and head can clean up later
                cand_struct_dir = os.path.join(workdir, f"cand_structs_{tag}")
                os.makedirs(cand_struct_dir, exist_ok=True)
                await tasks.fold(
                    name=tag,
                    GPUs=gpus,
                    seed=rng_seed,
                    outdir=cand_struct_dir,
                    query=cand_fa,
                    batch_size=fold_batch_size,
                )
                await tasks.foldseek_createdb(fasta=cand_struct_dir, db_path=cand_db)
                await tasks.foldseek_search(
                    query_db=cand_db,
                    target_db=input_db,
                    tsv_out=out_tsv,
                    tmp_dir=tmp_dir,
                    extra_flags='--alignment-type 1 --format-output "query,target,alntmscore"',
                )
                avg = _parse_foldseek_avg(out_tsv, fast_mode=False)

            score_map = dict(zip(avg["query"].values, avg["avg_score"].values, strict=False))
            rewards_np = np.array([score_map.get(h, 0.0) for h in headers], dtype=np.float32)

            try:
                # Clean up per-step files to avoid accumulating them.
                os.remove(cand_fa)
                os.remove(out_tsv)
                import glob as _glob
                for _f in _glob.glob(f"{cand_db}*"):
                    try:
                        os.remove(_f)
                    except Exception:
                        pass
                if not fast_folding:
                    shutil.rmtree(cand_struct_dir, ignore_errors=True)
            except Exception:
                pass

            batch_index = population.current_batch_index + 1
            scored = [
                sample.copy(reward=float(r), batch_index=batch_index, new_key=False)
                for sample, r in zip(samples, rewards_np, strict=False)
            ]
            population.add_samples(scored)

            now = time.time()
            step_elapsed = now - _step_times[-1]
            total_elapsed = now - t0_total
            last_rewards = [s.reward for s in population.get_last_batch()]
            best_r = max(last_rewards) if last_rewards else float("nan")
            mean_r = sum(last_rewards) / len(last_rewards) if last_rewards else float("nan")
            print(
                f"[DES ft_round={round_i} step={step + 1}/{des_rounds}] "
                f"step={step_elapsed:.1f}s  total={total_elapsed:.1f}s  "
                f"best_reward={best_r:.4f}  mean_reward={mean_r:.4f}  "
                f"pop={len(population)}"
            )
            _step_times.append(now)

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
        os.makedirs(os.path.dirname(des_fasta), exist_ok=True)
        _int_array_to_fasta(des_arr, des_fasta, prefix=f"des_r{round_i}")
        shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Per-mutation async implementation
    # ------------------------------------------------------------------

    async def _sgdes_async(self, mutation: str, policy) -> None:
        """Run one full SGDES mutation: embed → DES → foldseek → rank → eval."""
        tasks = self._register_tasks(policy)

        abspath = os.path.join(self.base_outdir, f"{mutation}_wd")
        os.makedirs(abspath, exist_ok=True)

        name = f"{mutation}_run"
        query = os.path.join(self.query_dir, f"mayv_{mutation}.fasta")
        gpus_count = "1"

        args = types.SimpleNamespace(
            name=name,
            outdir=abspath,
            query=query,
            wt_query=self.wt_query,
            GPUs=gpus_count,
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
                    GPUs=gpus_count,
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
                        GPUs=gpus_count,
                        seed=args.RNG_seed,
                        outdir=f"{abspath}/{name}_foldtune_input_structs",
                        query=query,
                        batch_size=args.fold_batch_size,
                    )
                    logger.info(f"[{mutation}] Fold done ({time.time() - t0:.1f}s)")

                stats_out = await tasks.seqkit_stats(input_fasta=query)
                df_stats = pd.read_csv(io.StringIO(stats_out), sep="\t")
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
            await self._run_des(
                tasks,
                de_ref_fasta=de_ref_fasta,
                fixed_len=fixed_len,
                des_fasta=des_fasta,
                ori_ref=ori_ref,
                ori_ref_structs=ori_ref_structs,
                round_i=i,
                fast_folding=args.fast_folding,
                prostt5_weights_path=prostt5_weights_path if args.fast_folding else None,
                fold_batch_size=args.fold_batch_size,
                gpus=gpus_count,
                rng_seed=args.RNG_seed,
                num_mutations=args.num_mutations,
                des_rounds=args.des_rounds,
                des_batch_size=args.des_batch_size,
                des_num_sequences=args.des_num_sequences,
                workdir_base=abspath,
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
            n_cleaned = sum(1 for line in open(cleaned) if line.startswith(">"))
            logger.info(f"[{mutation}] Merged/cleaned → {n_cleaned} seqs  ({cleaned})")

            # ── Embed generated sequences ──────────────────────────────────────
            logger.info(f"[{mutation}] Embedding {n_cleaned} generated sequences (ESM2-650M)")
            t0 = time.time()
            await tasks.embed(
                name=f"{name}_round{i}",
                GPUs=gpus_count,
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
                    GPUs=gpus_count,
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
            # _grep_td pins stdout to output_fasta via Dragon's ProcessTemplate;
            # re-enable if seqkit_grep is switched back to writing stdout directly
            # rather than returning it as a string.
            # _grep_td = (
            #     {
            #         "capture_stdio": True,
            #         "process_template": {"policy": tasks.host_policy, "stdout": output_fasta},
            #     }
            #     if tasks.host_policy is not None
            #     else {
            #         "capture_stdio": True,
            #         "process_template": {"stdout": output_fasta},
            #     }
            # )

            res = await tasks.seqkit_grep(
                # task_description=_grep_td,
                pattern_file=labels_file,
                input_fasta=cleaned,
            )
            with open(output_fasta, "w+") as output_file:
                output_file.write(res)

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
                    GPUs=gpus_count,
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

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _run_one(self, mutation: str, idx: int) -> None:
        if self.policies is not None:
            policy = self.policies[idx % len(self.policies)]
            logger.info(f"[{mutation}] Starting (policy={policy})")
        else:
            policy = None
            logger.info(f"[{mutation}] Starting")

        try:
            await self._sgdes_async(mutation, policy)
        except Exception as e:
            import traceback

            logger.error(f"[{mutation}] FAILED: {e}\n{traceback.format_exc()}")
            raise

    # async def run(self) -> None:
    #     """Run all mutations in parallel; asyncflow schedules task concurrency."""
    #     tasks = [self._run_one(mutation, idx) for idx, mutation in enumerate(self.mutations)]
    #     results = await asyncio.gather(*tasks, return_exceptions=True)
    #     failed = [self.mutations[i] for i, r in enumerate(results) if isinstance(r, Exception)]
    #     if failed:
    #         logger.error(f"Failed mutations: {failed}")

    async def run(self) -> None:
        """Run mutations with concurrency capped at the number of GPU slots.

        A semaphore limits active mutations to len(self.policies) so that
        each running mutation maps to a unique policy slot (node + GPU).
        When a mutation finishes it releases the semaphore, and the next
        queued mutation picks up that slot via idx % len(policies).
        """
        num_slots = len(self.policies) if self.policies else len(self.mutations)
        sem = asyncio.Semaphore(num_slots)

        async def _bounded(mutation, idx):
            async with sem:
                await self._run_one(mutation, idx)

        tasks = [_bounded(m, i) for i, m in enumerate(self.mutations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [self.mutations[i] for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed:
            logger.error(f"Failed mutations: {failed}")
