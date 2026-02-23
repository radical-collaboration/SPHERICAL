#!/usr/bin/env python3
"""
Example: running a campaign of concurrent workflows with CampaignManager.

Campaign layout
---------------
  InferenceWorkflow  (priority 10 — claims all 8 GPUs)
      client_req  ──────────────────────────────────────── [done]

  DDSimWorkflow      (priority 9 → 5, five-stage chain)
      simulation → train_model → training → selection → inference

The two workflows run concurrently and share the node's resources.
Higher-priority tasks are dispatched first when resources are contested
(client_req holds all GPUs so DDSim simulation waits until it finishes).

Usage
-----
    # dry-run (no real work, just traces the chain)
    python run_campaign.py --dry-run

    # real run
    python run_campaign.py --config config.json
"""

import argparse
import sys
from pathlib import Path

# Make the src package importable when running directly from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.campaign import CampaignManager

from ddsim_workflow import DDSimWorkflow
from inference_workflow import InferenceWorkflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "config.json"),
        help="Path to campaign config JSON"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip real executors (useful for testing the chain)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config and create manager
    # ------------------------------------------------------------------
    cm = CampaignManager(args.config)

    if args.dry_run:
        cm._dry_run = True          # override config setting at runtime

    print("Campaign resources:")
    print(f"  total CPUs : {cm.total_cpus}")
    print(f"  total GPUs : {cm.total_gpus}")
    print()

    # ------------------------------------------------------------------
    # 2. Start the inference workflow
    #    Single task (client_req) that runs once then exits.
    # ------------------------------------------------------------------
    print("Starting 'inference' workflow ...")
    cm.run_workflow(InferenceWorkflow())

    # ------------------------------------------------------------------
    # 3. Start the ddsim workflow
    #    Five-stage chain: simulation → train_model → training →
    #                      selection → inference
    #
    #    after_simulation() on the workflow class overrides the default
    #    string chain for the first step, enabling conditional branching.
    #    The remaining steps use TaskSpec.on_completion strings.
    # ------------------------------------------------------------------
    print("Starting 'ddsim' workflow ...")
    cm.run_workflow(DDSimWorkflow())

    # ------------------------------------------------------------------
    # 4. Block until every workflow is done
    # ------------------------------------------------------------------
    print("\nWaiting for all workflows to complete ...")
    cm.wait_all()
    print("\nAll workflows done.")

    # ------------------------------------------------------------------
    # 5. Shutdown
    # ------------------------------------------------------------------
    cm.close()


if __name__ == "__main__":
    main()
