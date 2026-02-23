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

from src.inference.esm2_service import ESM2Client, ESM2InferenceService
from src.inference.logger import Logger
from src.inference.orchestrator import init_clients, start_services, start_services_local
from src.inference.utils import load_config

logger = Logger(use_colors=True)


async def main(config_file: str, mode: str):
    config = load_config(config_file)

    if mode == "server":
        services = await start_services(config, ESM2InferenceService)
    else:
        services = await start_services_local(config, ESM2InferenceService)

    clients, collector = await init_clients(config, services, ESM2Client)
    if collector:
        collector.start()

    if clients is None:
        logger.error("Unable to initiate clients")
        return

    try:
        await asyncio.gather(*(client.run() for client in clients))
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
