"""
DDSimWorkflow — DeepDriveMD-style simulation campaign workflow.

Pipeline
--------
    simulation → train_model → training → selection → inference

Each stage is a method on this class.  The first stage (simulation) uses
an ``after_simulation`` override to demonstrate conditional branching before
the chain continues.  All other stages chain automatically via TaskSpec
on_completion strings.
"""

import time

from src.campaign import BaseWorkflow, TaskSpec


class DDSimWorkflow(BaseWorkflow):
    """
    Five-stage iterative ML/simulation workflow.

    Stage order  (priority 9 → 5):
      simulation → train_model → training → selection → inference
    """

    workflow_id = "ddsim"
    init_tasks  = ["simulation"]

    def task_specs(self):
        return {
            "simulation": TaskSpec(
                priority       = 9,
                ranks          = 1,
                cores_per_rank = 1,
                gpus_per_rank  = 1,
                on_completion  = "train_model",
            ),
            "train_model": TaskSpec(
                priority       = 8,
                ranks          = 1,
                cores_per_rank = 1,
                gpus_per_rank  = 1,
                on_completion  = "training",
            ),
            "training": TaskSpec(
                priority       = 7,
                ranks          = 1,
                cores_per_rank = 4,
                gpus_per_rank  = 1,
                on_completion  = "selection",
            ),
            "selection": TaskSpec(
                priority       = 6,
                ranks          = 1,
                cores_per_rank = 2,
                gpus_per_rank  = 0,
                on_completion  = "inference",
            ),
            "inference": TaskSpec(
                priority       = 5,
                ranks          = 1,
                cores_per_rank = 1,
                gpus_per_rank  = 0,
            ),
        }

    # ------------------------------------------------------------------
    # Executors  (run_<task_type>)
    # ------------------------------------------------------------------

    def run_simulation(self, task_desc: dict) -> dict:
        """Run molecular dynamics / physics simulation."""
        print(f"  [simulation] starting  — {task_desc['ranks']} rank(s), "
              f"{task_desc['gpus_per_rank']} GPU/rank")
        time.sleep(0.05)
        result = {"trajectories": 100, "converged": True}
        print(f"  [simulation] done → {result}")
        return result

    def run_train_model(self, task_desc: dict) -> dict:
        """Train a surrogate model on simulation outputs."""
        print(f"  [train_model] starting  — {task_desc['gpus_per_rank']} GPU/rank")
        time.sleep(0.05)
        result = {"model_path": "/tmp/surrogate_v1.pt", "val_loss": 0.012}
        print(f"  [train_model] done → {result}")
        return result

    def run_training(self, task_desc: dict) -> dict:
        """Full training pass (more cores, longer)."""
        print(f"  [training] starting  — {task_desc['cores_per_rank']} core(s), "
              f"{task_desc['gpus_per_rank']} GPU/rank")
        time.sleep(0.05)
        result = {"model_path": "/tmp/model_final.pt", "val_loss": 0.008}
        print(f"  [training] done → {result}")
        return result

    def run_selection(self, task_desc: dict) -> dict:
        """Select next batch of candidates from surrogate predictions."""
        print("  [selection] starting")
        time.sleep(0.02)
        result = {"selected": 50, "batch_file": "/tmp/candidates.csv"}
        print(f"  [selection] done → {result}")
        return result

    def run_inference(self, task_desc: dict) -> dict:
        """Run model inference on selected candidates."""
        print("  [inference] starting")
        time.sleep(0.02)
        result = {"scored": 50, "output": "/tmp/scores.csv"}
        print(f"  [inference] done → {result}")
        return result

    # ------------------------------------------------------------------
    # on_completion override  (after_<task_type>)
    # Overrides the TaskSpec string chain for 'simulation' only.
    # Demonstrates conditional branching: abort on failure.
    # ------------------------------------------------------------------

    def after_simulation(
        self, final_state: str, cm, workflow_id: str
    ) -> None:
        """
        Called after simulation completes.

        Demonstrates conditional branching: only proceed if the simulation
        converged.  When final_state == 'done' we manually submit the next
        task (same behaviour as the string chain, but with full control).
        """
        print(f"  [after_simulation] state={final_state!r}")
        if final_state == "done":
            cm.submit("train_model", workflow_id)
        else:
            print("  [after_simulation] simulation failed — aborting ddsim workflow")
