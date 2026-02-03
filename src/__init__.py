"""
Spherical - Multi-GPU Inference Service Framework

A framework for building multi-GPU inference services with:
- Worker pool management
- HTTP server/client architecture
- Metrics collection and logging
- Multi-node orchestration
"""

from .inference_service import GPUWorker, InferenceService
from .logger import Logger
from .orchestrator import (
    ServiceHandle,
    init_clients,
    launch_server,
    launch_servers,
    start_services,
    wait_for_healthy,
)
from .server import create_app, get_app, init_server
from .utils import ensure_dir, export_metrics, get_devices_for_node, get_slurm_nodes, load_config

__version__ = "0.1.0"

__all__ = [
    "InferenceService",
    "GPUWorker",
    "Logger",
    "init_server",
    "create_app",
    "get_app",
    "launch_server",
    "launch_servers",
    "wait_for_healthy",
    "start_services",
    "init_clients",
    "ServiceHandle",
    "load_config",
    "ensure_dir",
    "get_devices_for_node",
    "get_slurm_nodes",
    "export_metrics",
]
