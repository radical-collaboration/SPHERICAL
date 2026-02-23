"""
InferenceWorkflow — single-task ESM2 client request workflow.

Submits one high-priority task that consumes all available GPUs on the node,
then terminates (no on_completion chain).
"""

import time

from src.campaign import BaseWorkflow, TaskSpec


class InferenceWorkflow(BaseWorkflow):
    """
    Single-task inference workflow.

    Sends a batch of requests to a remote ESM2 inference service.
    Runs at the highest priority and holds all GPUs until complete.
    """

    workflow_id = "inference"
    init_tasks  = ["client_req"]

    def task_specs(self):
        return {
            "client_req": TaskSpec(
                priority       = 10,
                ranks          = 1,
                cores_per_rank = 0,
                gpus_per_rank  = 8,   # claims all GPUs on the node
            ),
        }

    # ------------------------------------------------------------------
    # Executor  (run_<task_type>)
    # ------------------------------------------------------------------

    def run_client_req(self, task_desc: dict) -> dict:
        """Submit inference requests to the remote ESM2 service."""
        print(f"  [client_req] starting  — {task_desc['gpus_per_rank']} GPU/rank")
        time.sleep(0.1)
        result = {"requests_sent": 500, "successful": 500}
        print(f"  [client_req] done → {result}")
        return result
