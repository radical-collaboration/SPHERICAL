#!/usr/bin/env python3
"""
ESM2 Inference Runner

Example script demonstrating how to run ESM2 inference using the spherical framework.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.campaign import ResourceManager
from src.inference.esm2_service import ESM2Client, ESM2InferenceService
from src.utils.logger import Logger
from src.inference.orchestrator import init_clients, start_services, start_services_local
from src.inference.utils import load_config
from radical.asyncflow import WorkflowEngine
from pipelines.ddmd_pipeline.ddmd_pipeline import DDMdWorkflow
from pipelines.dummy_pipeline.dummy_pipeline import DummyWorkflow

logger = Logger(use_colors=True)


async def main(config_file: str, mode: str):
    config = load_config(config_file)
    ddsim_config = config.get("ddsim_config", {})

    num_services = config.get("num_services", 1)
    num_gpus_per_service = config.get("num_gpus_per_service", 1)
    num_cpus_per_service = config.get("num_cpus_per_service", 1)
    total_gpus = num_gpus_per_service * num_services
    total_cpus = num_cpus_per_service * num_services
    rm = ResourceManager(total_cpus=total_cpus, total_gpus=total_gpus)


    use_dragon = False
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
    #workflow = DDMdWorkflow(asyncflow=asyncflow, config=ddsim_config, resource_manager=rm)
    workflow = DummyWorkflow(asyncflow=asyncflow, resource_manager=rm)
    ddsim = workflow.start()

    if mode == "server":
        services = await start_services(config, ESM2InferenceService)
    else:
        services = await start_services_local(config, ESM2InferenceService)

    clients, collector = await init_clients(config, services, ESM2Client, resource_manager=rm)
    if collector:
        collector.start()

    try:
        if clients is None:
            logger.error("Unable to initiate clients")
        else:
            async def run_ddsim():
                try:
                    await ddsim
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    import traceback
                    logger.error(f"DDSim workflow failed: {e}\n{traceback.format_exc()}")
                finally:
                    if rm is not None:
                        released = rm.release_by_workflow(workflow.workflow_id)
                        logger.info(f"DDSim finished — released {released} RM slot(s)")

            await asyncio.gather(
                asyncio.gather(*(client.run() for client in clients)),
                run_ddsim(),
            )

    except Exception as e:
        logger.error(f"An error occurred while running inference: {e}")
    finally:
        # Close clients first (shuts down asyncflow)
        for client in clients:
            if hasattr(client, "close"):
                await client.close()

        # Then close services
        for service in services:
            if hasattr(service, "close"):
                await service.close()

        if collector:
            collector.stop()
        await workflow.close()

        rm.close()

    print("All work has been completed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM2 inference")
    parser.add_argument(
        "--config_file", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "local"],
        default="server",
        help="Run mode: server (hosting model on server, default) or use model locally",
    )
    args = parser.parse_args()

    asyncio.run(main(args.config_file, args.mode))

# Hint: use this command to run with dragon dragon -w ssh --network-config slurm.yaml  run_esm2_infern.py
