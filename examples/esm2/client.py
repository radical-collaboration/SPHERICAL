#!/usr/bin/env python3
"""
ESM2 Inference Client

HTTP client for remote ESM2 inference with load balancing and retry logic.
"""

import asyncio
import itertools
from pathlib import Path
from typing import Any, Optional

import aiohttp

from src.logger import Logger
from src.utils import export_metrics


class ESM2Client:
    """
    HTTP client for remote ESM2 inference.

    Features:
    - Round-robin load balancing across endpoints
    - Retry logic with exponential backoff
    - Health checks
    - Concurrent request management
    - Queue processing
    """

    def __init__(
        self,
        endpoints: list[str],
        rank: int = 0,
        service: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
        asyncflow: Optional[Any] = None,
    ):
        """
        Initialize ESM2 client.

        Args:
            endpoints: List of server HTTP URLs
            rank: Client rank identifier
            service: Optional InferenceService for batch generation
            config: Optional configuration dictionary
            asyncflow: Optional AsyncFlow workflow engine
        """
        self.config = config or {}
        self.endpoints = endpoints
        self.endpoint_cycle = itertools.cycle(endpoints) if endpoints else None
        self.rank = rank
        self.max_concurrent = self.config.get("max_concurrent", 16)
        self.timeout = self.config.get("timeout", 600)
        self.max_retries = self.config.get("max_retries", 3)
        self.service = service
        self.metrics_dir = self.config.get("metrics_dir", "outputs")

        self.flow = asyncflow
        self.logger = Logger(use_colors=True)

        self.metrics = {
            "submitted": 0,
            "successful": 0,
            "failed": 0,
            "error_msgs": [],
            "retries": 0,
        }

        self.logger.info(
            f"[Client {self.rank}] Initialized with {len(endpoints)} endpoints, "
            f"max_concurrent={self.max_concurrent}, timeout={self.timeout}s"
        )

        self.debug = config.get("debug", False)
        # Use asyncflow if provided, otherwise use pure asyncio
        if self.flow:
            self._register_client()
        else:
            self.logger.critical("Unable to start client without asyncflow engine")

    def _register_client(self):
        """Register client function with asyncflow (Dragon-compatible)."""
        # Capture only simple serializable values - NO logger, NO iterators
        timeout = self.timeout
        max_retries = self.max_retries

        @self.flow.function_task
        async def client_req(
            batch_id: int,
            endpoint: str,
            request_timeout: int,
            retries: int,
        ) -> dict[str, Any]:
            """
            Submit a single batch for inference via HTTP POST.

            Dragon-compatible: no closures over non-serializable objects.
            Returns result dict for aggregation by caller.
            """
            url = f"{endpoint}/generate"
            payload = {
                "batch_ids": [batch_id],
                "timeout": request_timeout,
            }

            last_error = None
            retry_count = 0

            for attempt in range(retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=request_timeout),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()

                                if data.get("status") == "success":
                                    return {
                                        "status": "success",
                                        "batch_id": batch_id,
                                        "successful": data.get("successful", 0),
                                        "failed": data.get("failed", 0),
                                        "retries": retry_count,
                                    }
                                else:
                                    error = data.get("message", "Unknown error")
                                    return {
                                        "status": "error",
                                        "batch_id": batch_id,
                                        "error": error,
                                        "failed": data.get("failed", 1),
                                        "retries": retry_count,
                                    }
                            else:
                                error = f"HTTP {resp.status}: {await resp.text()}"
                                last_error = error

                                if attempt < retries - 1:
                                    retry_count += 1
                                    backoff = 0.5 * (2**attempt)
                                    await asyncio.sleep(backoff)
                                    continue
                                else:
                                    return {
                                        "status": "error",
                                        "batch_id": batch_id,
                                        "error": error,
                                        "failed": 1,
                                        "retries": retry_count,
                                    }

                except asyncio.TimeoutError:
                    last_error = "Timeout"
                    if attempt < retries - 1:
                        retry_count += 1
                        backoff = 0.5 * (2**attempt)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        return {
                            "status": "error",
                            "batch_id": batch_id,
                            "error": "Timeout",
                            "failed": 1,
                            "retries": retry_count,
                        }

                except Exception as e:
                    last_error = str(e)
                    if attempt < retries - 1:
                        retry_count += 1
                        backoff = 0.5 * (2**attempt)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        return {
                            "status": "error",
                            "batch_id": batch_id,
                            "error": str(e),
                            "failed": 1,
                            "retries": retry_count,
                        }

            return {
                "status": "error",
                "batch_id": batch_id,
                "error": f"Max retries exceeded: {last_error}",
                "failed": 1,
                "retries": retry_count,
            }

        # Store the task function and create a wrapper
        self._client_req_task = client_req

        def submit_request(batch_id: int, endpoint: str):
            """Wrapper that calls the asyncflow task with proper parameters."""
            return client_req(batch_id, endpoint, timeout, max_retries)

        self.client_req = submit_request

    async def health_check(self, endpoint: Optional[str] = None) -> bool:
        """Check if an endpoint is healthy."""
        if endpoint is None:
            endpoint = self.endpoints[0]

        health_url = f"{endpoint}/health"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.logger.info(
                            f"[Client {self.rank}] {endpoint} - healthy "
                            f"({data.get('total_workers', 0)} workers)"
                        )
                        return True
                    else:
                        self.logger.warning(
                            f"[Client {self.rank}] {endpoint} - unhealthy (HTTP {resp.status})"
                        )
                        return False
        except Exception as e:
            self.logger.warning(f"[Client {self.rank}] {endpoint} - unreachable ({e})")
            return False

    async def close(self):
        """Cleanup client resources."""
        self.logger.info(f"[Client {self.rank}] Closing")
        metrics_output = Path(self.metrics_dir, f"client_{self.rank}.json")
        await export_metrics(metrics_output, self.metrics)
        if self.flow:
            await self.flow.shutdown()

    async def run(self):
        """Run inference workflow with batch generation and remote submission."""
        self.logger.info(f"[Client {self.rank}] Starting remote inference")

        if self.service is None:
            self.logger.error(f"[Client {self.rank}] No service provided for batch generation")
            return

        try:
            self.logger.task_started(f"[rank {self.rank}] Batch generation")
            await self.service.init_queue()
            self.logger.task_completed(f"[rank {self.rank}] Batch generation")

            self.logger.task_started(f"[rank {self.rank}] Remote inference")
            await self._process_queue()
            self.logger.task_completed(f"[rank {self.rank}] Remote inference")

        except Exception as e:
            self.logger.error(f"[Client {self.rank}] Error in run(): {e}")
            raise
        finally:
            self.logger.info(f"[Client {self.rank}] Cleanup complete")

    async def _process_queue(self):
        """Process batches from seq_queue and dispatch to remote servers."""
        batch_count = 0
        tasks = []

        while True:
            batch_id = await self.service.seq_queue.get()

            try:
                if batch_id is None:
                    self.logger.info(f"[Client {self.rank}] Received shutdown sentinel")
                    if tasks:
                        await self._flush_tasks(tasks)
                    break

                batch_count += 1
                if self.endpoints:
                    endpoint = next(self.endpoint_cycle)
                    task = self.client_req(batch_id, endpoint)
                    if batch_count == 1 and self.debug:
                        # Debug: log the type of task object on first iteration
                        self.logger.debug(
                            f"[Client {self.rank}] Task type: {type(task)}, awaitable: {hasattr(task, '__await__')}"
                        )
                    tasks.append(task)

                    if self.debug and batch_count % 100 == 0:
                        self.logger.debug(f"[Client {self.rank}] Dispatched {batch_count} batches")

                    if len(tasks) >= self.max_concurrent:
                        await self._flush_tasks(tasks)
                        tasks = []
                else:
                    # Local mode - submit directly to work queue
                    self.service.work_queue.put_nowait((batch_id, None))

            finally:
                self.service.seq_queue.task_done()

        self.logger.info(f"[Client {self.rank}] Dispatched {batch_count} batches total")

    async def _flush_tasks(self, tasks: list):
        """Wait for a batch of tasks to complete and aggregate metrics."""
        if not tasks:
            return

        self.logger.debug(f"[Client {self.rank}] Flushing {len(tasks)} tasks")

        results = []

        # Asyncflow mode - await tasks sequentially for debugging
        for i, task in enumerate(tasks):
            try:
                if self.debug:
                    self.logger.debug(
                        f"[Client {self.rank}] Awaiting task {i}/{len(tasks)}, type={type(task).__name__}"
                    )
                # Try to await the task - different backends may return different types
                if hasattr(task, "__await__"):
                    result = await asyncio.wait_for(task, timeout=self.timeout)
                elif hasattr(task, "result"):
                    # Some futures have a .result() method
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, task.result),
                        timeout=self.timeout,
                    )
                else:
                    self.logger.error(
                        f"[Client {self.rank}] Task {i} is not awaitable: {type(task)}"
                    )
                    result = {
                        "status": "error",
                        "error": f"Not awaitable: {type(task)}",
                        "failed": 1,
                    }
                if self.debug:
                    self.logger.debug(f"[Client {self.rank}] Task {i} completed: {result}")
                results.append(result)
            except asyncio.TimeoutError:
                self.logger.error(f"[Client {self.rank}] Task {i} timed out after {self.timeout}s")
                results.append({"status": "error", "error": "Timeout", "failed": 1})
            except Exception as e:
                self.logger.error(
                    f"[Client {self.rank}] Task {i} failed with {type(e).__name__}: {e}"
                )
                import traceback

                self.logger.debug(f"[Client {self.rank}] Traceback: {traceback.format_exc()}")
                results.append({"status": "error", "error": str(e), "failed": 1})

        # Aggregate metrics from results
        for r in results:
            if isinstance(r, Exception):
                self.metrics["failed"] += 1
                self.metrics["error_msgs"].append(str(r))
            elif isinstance(r, dict):
                self.metrics["successful"] += r.get("successful", 0)
                self.metrics["failed"] += r.get("failed", 0)
                self.metrics["retries"] += r.get("retries", 0)
                if r.get("status") == "error":
                    batch_id = r.get("batch_id", "?")
                    error = r.get("error", "Unknown")
                    self.metrics["error_msgs"].append(f"batch {batch_id}: {error}")
                    self.logger.error(f"[Client {self.rank}] Batch {batch_id}: {error}")
