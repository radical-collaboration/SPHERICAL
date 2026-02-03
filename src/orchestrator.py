#!/usr/bin/env python3
"""
Inference Orchestrator

Manages multi-node deployment of inference services.

Features:
- Launch servers on multiple nodes
- Wait for servers to become healthy
- Create clients for remote inference
- Graceful shutdown coordination
"""

import asyncio
import itertools
import socket
from typing import Any, Optional

from aiohttp import web
from radical.asyncflow import WorkflowEngine

from src.logger import Logger
from src.server import get_app, init_server
from src.utils import get_devices_for_node, get_slurm_nodes
from src.utils import ensure_dir, init_collector

logger = Logger(use_colors=True)

__all__ = [
    "launch_server",
    "launch_servers",
    "wait_for_healthy",
    "start_services",
    "start_services_local",
    "init_clients",
    "ServiceHandle",
]


async def launch_server(
    config: dict[str, Any],
    node_rank: int,
    hostname: str,
    port: int,
    service_class: type[Any],
    use_https: bool = False,
) -> tuple[Optional[str], Optional[Any], Optional[web.AppRunner]]:
    """
    Launch an inference server on a single node.

    Args:
        config: Configuration dictionary
        node_rank: Rank/index of this node
        hostname: Hostname of the node
        port: Port to use for this server
        service_class: subclass to instantiate
        use_https: Whether to use HTTPS in endpoint URLs

    Returns:
        Tuple of (endpoint_url, inference_service, app_runner) or (None, None, None) if failed
    """
    engine = config.get("engine", False).lower()
    if "dragon" in engine and hostname:
        fqdn = hostname
    else:
        fqdn = socket.getfqdn(hostname) if hostname else socket.getfqdn()
    logger.info(f"Launching server on node {node_rank} ({fqdn}:{port})...")

    try:
        devices = get_devices_for_node(config)
        if not devices:
            logger.error(f"[Server {node_rank}] No GPUs allocated!")
            return None, None, None

        node_name = f"node{node_rank}_{hostname.split('.')[0]}"
        logger.info(f"[Server {node_rank}] Using devices: {devices}")

        inference_service = init_server(
            config, devices, node_rank, node_name, fqdn, port, service_class=service_class
        )

        app = get_app()

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, host="0.0.0.0", port=port)
        await site.start()

        protocol = "https" if use_https else "http"
        endpoint = f"{protocol}://{fqdn}:{port}"

        logger.info(f"[Server {node_rank}] Server started on {endpoint}")

        return endpoint, inference_service, runner

    except Exception as e:
        logger.error(f"Error launching server on {fqdn}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None, None, None


async def launch_servers(
    config: dict[str, Any],
    service_class: type[Any],
    nodes: Optional[list[str]] = None,
    base_port: int = 8000,
    use_https: bool = False,
) -> list[tuple[str, Any, web.AppRunner]]:
    """
    Launch servers on all available nodes concurrently.

    Args:
        config: Configuration dictionary
        service_class: subclass to instantiate
        nodes: List of node hostnames (auto-detected if None)
        base_port: Base port number (each node gets base_port + node_rank)
        use_https: Whether to use HTTPS in endpoint URLs

    Returns:
        List of (endpoint, inference_service, runner) tuples for successful launches
    """
    if nodes is None:
        nodes = get_slurm_nodes(config)
        logger.info(f"Auto-detected {len(nodes)} nodes: {nodes}")

    if not nodes:
        logger.error("No nodes available!")
        return []

    logger.separator(title=f"LAUNCHING SERVERS ON {len(nodes)} NODES")

    num_services = config.get("num_services", len(nodes))
    nodes = nodes[:num_services]

    launch_tasks = []
    for rank, hostname in enumerate(nodes):
        port = base_port + rank
        task = launch_server(config, rank, hostname, port, service_class, use_https)
        launch_tasks.append(task)

    results = await asyncio.gather(*launch_tasks, return_exceptions=True)

    servers = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Launch failed with exception: {result}")
        elif result[0] is not None:
            servers.append(result)

    logger.info(f"Successfully launched {len(servers)}/{len(nodes)} servers")
    return servers


async def wait_for_healthy(
    endpoints: list[str],
    timeout: int = 60,
    check_interval: int = 2,
) -> list[str]:
    """
    Wait for servers to be ready and perform health checks.

    Args:
        endpoints: List of server endpoints
        timeout: Maximum time to wait in seconds
        check_interval: Time between health checks in seconds

    Returns:
        List of healthy endpoints
    """
    import time

    import aiohttp

    logger.separator(title="WAITING FOR SERVERS TO BE READY")

    start_time = time.time()

    async def check_endpoint(endpoint: str) -> bool:
        """Check if endpoint is healthy."""
        health_url = f"{endpoint}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(
                            f"{endpoint} - healthy "
                            f"({data.get('total_workers', 0)} workers, "
                            f"{len(data.get('devices', []))} GPUs)"
                        )
                        return True
                    else:
                        logger.debug(f"{endpoint} - unhealthy (HTTP {resp.status})")
                        return False
        except Exception as e:
            logger.debug(f"{endpoint} - unreachable ({e})")
            return False

    healthy = []
    while (time.time() - start_time) < timeout:
        health_checks = [check_endpoint(ep) for ep in endpoints]
        health_results = await asyncio.gather(*health_checks)

        healthy = [ep for ep, is_healthy in zip(endpoints, health_results) if is_healthy]

        if len(healthy) == len(endpoints):
            logger.info(f"All {len(healthy)} servers are healthy!")
            return healthy

        if healthy:
            logger.info(f"{len(healthy)}/{len(endpoints)} servers ready, waiting...")
        else:
            logger.info(f"No servers ready yet, waiting {check_interval}s...")

        await asyncio.sleep(check_interval)

    if healthy:
        logger.warning(f"Timeout: {len(healthy)}/{len(endpoints)} servers healthy")
    else:
        logger.error("Timeout: No healthy servers found!")

    return healthy


class ServiceHandle:
    """
    Wrapper for a running inference service with its endpoint.

    Provides a clean interface for managing the service lifecycle.
    """

    def __init__(
        self,
        endpoint: Optional[str],
        service: Optional[Any],
        runner: Optional[web.AppRunner] = None,
    ):
        self.endpoint = endpoint
        self.service = service
        self.runner = runner

    async def close(self):
        """Shutdown the service and cleanup resources."""
        logger.separator(title="SHUTTING DOWN SERVERS")
        try:
            logger.info(f"Shutting down {self.endpoint}...")
            if self.service:
                await self.service.shutdown()
            if self.runner:
                await self.runner.cleanup()
        except Exception as e:
            logger.error(f"Error shutting down {self.endpoint}: {e}")

    def __iter__(self):
        """Allow unpacking as (endpoint, service) for backward compatibility."""
        return iter((self.endpoint, self.service))


async def start_services_local(
    config: dict[str, Any],
    service_class: type[Any],
) -> list[ServiceHandle]:
    """
    Initialize the inference service for single node.

    Args:
        config: Configuration dictionary
        service_class: service_class subclass to instantiate
    Returns:
        List of ServiceHandle objects for healthy servers
    """

    num_services = config.get("num_services", 1)

    num_gpus_per_service = config.get("num_gpus_per_service", 1)
    devices = get_devices_for_node(config)
    devices_cycle = itertools.cycle(devices)
    handles = []

    for rank in range(num_services):
        devices = [next(devices_cycle) for _ in range(num_gpus_per_service)]

        service = service_class(config=config, devices=devices, rank=rank)
        handles.append(ServiceHandle(endpoint=None, service=service))
        logger.info(f"[Server {rank}] Local Inference service initialized on devices: {devices}")

    return handles


async def start_services(
    config: dict[str, Any],
    service_class: type[Any],
) -> list[ServiceHandle]:
    """
    Launch servers on all nodes and wait for them to be healthy.

    Args:
        config: Configuration dictionary
        service_class: service_class subclass to instantiate

    Returns:
        List of ServiceHandle objects for healthy servers
    """
    nodes = config.get("nodes", None)
    base_port = config.get("server_port", 8000)
    use_https = config.get("use_https", False)

    if nodes is None:
        nodes = get_slurm_nodes(config)
        logger.info(f"Auto-detected {len(nodes)} nodes: {nodes}")

    if not nodes:
        logger.error("No nodes available!")
        return []

    servers = await launch_servers(config, service_class, nodes, base_port, use_https)
    if not servers:
        logger.error("Failed to launch any servers!")
        return []

    endpoints = [ep for ep, _, _ in servers]

    healthy_endpoints = await wait_for_healthy(endpoints, timeout=60)
    if not healthy_endpoints:
        logger.error("No healthy servers available!")
        return []

    handles = [
        ServiceHandle(ep, svc, runner) for ep, svc, runner in servers if ep in healthy_endpoints
    ]

    logger.info(f"{len(handles)} services ready")
    return handles


async def init_clients(
    config: dict[str, Any],
    services: list[ServiceHandle],
    client_class: type,
) -> tuple[Optional[list[Any]], Optional[Any]]:
    """
    Initialize clients for the given services.

    Args:
        config: Configuration dictionary
        services: List of ServiceHandle objects from start_services()
        client_class: Client class to instantiate

    Returns:
        Tuple of (list of client instances, telemetry collector or None)
    """
    engine = config.get("engine", "concurrent").lower()
    collector = None
    asyncflow = None

    if "concurrent" in engine:
        from concurrent.futures import ProcessPoolExecutor

        from radical.asyncflow import ConcurrentExecutionBackend

        engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
        engine_name = "ConcurrentExecutionBackend"
    elif "dragon" in engine:
        from radical.asyncflow import DragonExecutionBackendV3

        dragon_workers = config.get("dragon_workers", 100)
        engine = await DragonExecutionBackendV3(
            num_workers=dragon_workers, disable_background_batching=False
        )
        engine_name = "DragonExecutionBackendV3"
        # Initialize telemetry collector if running with dragon
        collect_telemetry = config.get("collect_telemetry", False)
        if collect_telemetry:
            collector_dir = ensure_dir(config.get("telemetry_dir", "telemetry-results"))
            collector = init_collector(collector_dir)
    else:
        from radical.asyncflow import DaskExecutionBackend

        engine = await DaskExecutionBackend()
        engine_name = "DaskExecutionBackend"

    asyncflow = await WorkflowEngine.create(engine)
    logger.info(f"Asyncflow enabled with {engine_name} backend")

    clients = []
    for rank, handle in enumerate(services):
        try:
            if isinstance(handle, ServiceHandle):
                endpoint = handle.endpoint
                service = handle.service
            else:
                endpoint, service = handle
            if endpoint is not None:
                endpoints = [endpoint] if isinstance(endpoint, str) else endpoint
                client = client_class(
                    endpoints=endpoints,
                    rank=rank,
                    service=service,
                    config=config,
                    asyncflow=asyncflow,
                )
                clients.append(client)
            else:
                if service is not None:
                    client = client_class(
                        endpoints=[],
                        rank=rank,
                        service=service,
                        config=config,
                        asyncflow=asyncflow,
                    )
                    clients.append(client)
                    await service.start_workers()
        except Exception as e:
            logger.error(f"Unable to initiate client for rank {rank}: {e}")

    if not clients:
        logger.error("No clients were created!")
        return None, collector

    logger.info(f"Created {len(clients)} clients")
    return clients, collector
