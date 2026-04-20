#!/usr/bin/env python3
"""Entry point for running the DeepDriveMD workflow.

Initializes the execution backend (Dragon or concurrent), creates the
asyncflow workflow engine, and runs the DDMdWorkflow pipeline.
"""

import asyncio

from radical.asyncflow import WorkflowEngine

from pipelines.ddmd_pipeline.ddmd_pipeline import DDMdWorkflow
from pipelines.ddmd_pipeline.utils import parse_args


async def run_ddmd(config, use_dragon=False):
    """Set up the execution backend and run the DeepDriveMD workflow."""

    if use_dragon:
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except ImportError:
            use_dragon = False

    if use_dragon:
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend
        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)

    # Initialize and run the workflow
    workflow = DDMdWorkflow(asyncflow=asyncflow, config=config)
    await workflow.start()
    await workflow.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_ddmd(args.config))
