#!/usr/bin/env python3
"""
BaseWorkflow — abstract base class for campaign workflows.

Each workflow encapsulates:
  - its identity (workflow_id, init_tasks)
  - resource/priority specs per task type (task_specs())
  - executor logic  (run_<task_type> methods  OR  executor_for() override)
  - chaining logic  (after_<task_type> methods OR  on_completion_for() override)

Quick-start
-----------
    class SimWorkflow(BaseWorkflow):
        workflow_id = "sim"
        init_tasks  = ["simulation"]

        def task_specs(self):
            return {
                "simulation": TaskSpec(priority=9, gpus_per_rank=1,
                                       on_completion="training"),
                "training":   TaskSpec(priority=7, cores_per_rank=4,
                                       gpus_per_rank=1),
            }

        def run_simulation(self, task_desc: dict) -> dict:
            ...
            return result

        def run_training(self, task_desc: dict) -> dict:
            ...
            return result

        # Optional: override default string-chain with conditional logic
        def after_simulation(self, final_state: str, cm, workflow_id: str) -> None:
            if final_state == "done":
                cm.submit("training", workflow_id)
            else:
                print("simulation failed — aborting")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# TaskSpec — resource & chaining descriptor for one task type
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """
    Resource requirements and chaining rule for a single task type.

    Parameters
    ----------
    priority
        Higher value → dispatched first when resources are contested.
    ranks
        Number of parallel MPI ranks per task instance.
    cores_per_rank
        CPU cores allocated per rank.
    gpus_per_rank
        GPUs allocated per rank (fractional values allowed).
    pre_exec
        Shell commands to run before the executor (e.g. module loads).
    shell
        Whether the task should be launched via a shell.
    on_completion
        Task-type name to submit automatically when this task group
        finishes successfully.  Set to ``None`` to terminate the chain.
        Overridden by a matching ``after_<task_type>`` method.
    """
    priority:       int            = 0
    ranks:          int            = 1
    cores_per_rank: int            = 1
    gpus_per_rank:  float          = 0.0
    pre_exec:       List[str]      = field(default_factory=list)
    shell:          bool           = True
    on_completion:  Optional[str]  = None


# ---------------------------------------------------------------------------
# BaseWorkflow — abstract base class
# ---------------------------------------------------------------------------


class BaseWorkflow(ABC):
    """
    Abstract base class for campaign workflows.

    Subclasses must define:
      - ``workflow_id``  (class attribute or property) — unique string
      - ``init_tasks``   (class attribute or property) — list of task types
                         submitted immediately when the workflow starts
      - ``task_specs()`` — dict mapping each task_type → TaskSpec

    Executors are resolved by calling ``executor_for(task_type)``.
    The default implementation looks for a method named ``run_<task_type>``.

    on_completion handlers are resolved by calling ``on_completion_for(task_type)``.
    The default implementation looks for a method named ``after_<task_type>``.
    When ``None`` is returned the TaskSpec.on_completion string chain is used.
    """

    # Subclasses set these as class attributes or override as properties.
    workflow_id: str       = ""
    init_tasks:  List[str] = []

    @abstractmethod
    def task_specs(self) -> Dict[str, "TaskSpec"]:
        """Return ``{task_type: TaskSpec}`` for every task in this workflow."""

    def executor_for(self, task_type: str) -> Optional[Callable]:
        """
        Return the executor callable for *task_type*, or ``None`` (→ no-op).

        Default behaviour: look for a method named ``run_<task_type>`` on
        ``self``.  Override this method for custom dispatch logic.

        The returned callable receives a single ``task_desc`` dict:

            {
                "ranks":          int,
                "cores_per_rank": int,
                "gpus_per_rank":  float,
                "pre_exec":       list[str],
                "shell":          bool,
                "workflow_id":    str,
                "task_type":      str,
            }
        """
        return getattr(self, f"run_{task_type}", None)

    def on_completion_for(self, task_type: str) -> Optional[Callable]:
        """
        Return an ``on_completion`` callable for *task_type*, or ``None``.

        When ``None`` is returned the ``TaskSpec.on_completion`` string chain
        is used instead.

        Default behaviour: look for a method named ``after_<task_type>`` on
        ``self``.

        The returned callable must have the signature::

            fn(final_state: str, cm: CampaignManager, workflow_id: str) -> None
        """
        return getattr(self, f"after_{task_type}", None)
