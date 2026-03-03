#!/usr/bin/env python3
"""
InferenceClient — generic HTTP client for remote inference services.

Handles:
- Round-robin load balancing across endpoints
- Asyncflow-compatible request submission
- Resource-aware scheduling via ResourceManager
- Retry logic with exponential backoff
- Preemption handling and batch re-queuing
- Metrics collection

Subclasses override :meth:`_get_batch_data` to prepare model-specific
payload data for each batch.  All transport, queueing, and scheduling
logic lives here.
"""

import asyncio
import itertools
from pathlib import Path
from typing import Any, Optional

import aiohttp

from ..utils.logger import Logger
from .utils import export_metrics


class InferenceClient:
    """
    Generic HTTP client for remote inference.

    Subclasses must implement:
    - ``_get_batch_data(batch_id)`` — return the payload dict to send with
      each request, or ``None`` to let the server look up the batch itself.

    All other behaviour (resource management, queue draining, flush logic,
    preemption handling, metrics) is provided here.
    """

    def __init__(
        self,
        endpoints: list[str],
        rank: int = 0,
        service: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
        asyncflow: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
    ):
        self.config = config or {}
        self.endpoints = endpoints
        self.endpoint_cycle = itertools.cycle(endpoints) if endpoints else None
        self.rank = rank
        self.max_concurrent = self.config.get("max_concurrent", 16)
        self.timeout = self.config.get("timeout", 600)
        self.max_retries = self.config.get("max_retries", 3)
        self.service = service
        self.metrics_dir = self.config.get("metrics_dir", "outputs")
        self.tasks_config = self.config.get("tasks_config", {})
        self.workflow_id = self.config.get("workflow_id", "infern_workflow")
        self.debug = self.config.get("debug", False)

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

        # ---- resource manager ----------------------------------------
        self._rm = resource_manager
        # Set to a live asyncio.Event during run_inference(); signals that
        # a running resource slot was preempted (post-grant).
        self._preempted_event: Optional[asyncio.Event] = None
        # Batches that were re-queued due to preemption.  Stored here rather
        # than appended directly to seq_queue so they are processed BEFORE
        # the shutdown sentinel (which was already placed at the tail of
        # seq_queue by init_queue and cannot be moved).
        self._pending_requeue: list = []

        if self.flow:
            self._register_client()
        else:
            self.logger.critical("Unable to start client without asyncflow engine")

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def _get_batch_data(self, batch_id: int) -> Optional[dict]:
        """
        Return the batch payload to include in the POST request body, or
        ``None`` to send only the batch ID and let the server look it up.

        Override in subclasses to attach model-specific data (e.g. serialised
        token tensors) when the server does not have pre-loaded batches.
        """
        return None

    # ------------------------------------------------------------------
    # Client registration (asyncflow)
    # ------------------------------------------------------------------

    def _register_client(self):
        """Register the HTTP request function with asyncflow."""
        timeout = self.timeout
        max_retries = self.max_retries

        @self.flow.function_task
        async def client_req(
            batch_id: int,
            endpoint: str,
            request_timeout: int,
            retries: int,
            batch_data: Optional[dict] = None,
        ) -> dict[str, Any]:
            """Submit a single batch for inference via HTTP POST."""
            url = f"{endpoint}/generate"
            if batch_data is not None:
                payload = {
                    "batch_id": batch_id,
                    "batch": batch_data,
                    "timeout": request_timeout,
                }
            else:
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
                                    await asyncio.sleep(0.5 * (2 ** attempt))
                                    continue
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
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
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
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
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

        self._client_req_task = client_req

        def submit_request(batch_id: int, endpoint: str, batch_data: dict = None):
            return client_req(batch_id, endpoint, timeout, max_retries, batch_data)

        self.client_req = submit_request

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health_check(self, endpoint: Optional[str] = None) -> bool:
        """Check if an endpoint is healthy."""
        if endpoint is None:
            endpoint = self.endpoints[0]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.logger.info(
                            f"[Client {self.rank}] {endpoint} - healthy "
                            f"({data.get('total_workers', 0)} workers)"
                        )
                        return True
                    self.logger.warning(
                        f"[Client {self.rank}] {endpoint} - unhealthy (HTTP {resp.status})"
                    )
                    return False
        except Exception as e:
            self.logger.warning(f"[Client {self.rank}] {endpoint} - unreachable ({e})")
            return False

    async def close(self):
        """Export metrics and shut down asyncflow."""
        self.logger.info(f"[Client {self.rank}] Closing")
        await export_metrics(
            Path(self.metrics_dir, f"client_{self.rank}.json"), self.metrics
        )
        if self.flow:
            await self.flow.shutdown()

    async def init_queue(self) -> None:
        """Populate the sequence queue via the service."""
        if self.service is None:
            self.logger.error(f"[Client {self.rank}] No service provided for batch generation")
            return
        self.logger.task_started(f"[rank {self.rank}] Batch generation")
        await self.service.init_queue()
        self.logger.task_completed(f"[rank {self.rank}] Batch generation")

    async def run_inference(self) -> None:
        """Drain the sequence queue and dispatch inference requests."""
        self._preempted_event = asyncio.Event()
        try:
            self.logger.task_started(f"[rank {self.rank}] Remote inference")
            await self._process_queue()
            self.logger.task_completed(f"[rank {self.rank}] Remote inference")
        except asyncio.CancelledError:
            self.logger.warning(f"[Client {self.rank}] Inference was preempted — stopping")
            raise
        finally:
            self._preempted_event = None

    async def run(self) -> None:
        """Run init_queue then run_inference (convenience wrapper)."""
        self.logger.info(f"[Client {self.rank}] Starting remote inference")
        if self.service is None:
            self.logger.error(f"[Client {self.rank}] No service provided for batch generation")
            return
        try:
            await self.init_queue()
            await self.run_inference()
        except Exception as e:
            self.logger.error(f"[Client {self.rank}] Error in run(): {e}")
            raise
        finally:
            self.logger.info(f"[Client {self.rank}] Cleanup complete")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    async def _wait_for_resource(self, task_id: str, task_type: str) -> None:
        """
        Request a resource slot and suspend until granted.

        Bridges thread-safe ResourceManager callbacks to the asyncio event
        loop via ``loop.call_soon_threadsafe``.

        Raises ``asyncio.CancelledError`` if preempted before grant.
        Post-grant preemption is signalled via ``_preempted_event``.
        """
        assert self._rm is not None, "_wait_for_resource called without a ResourceManager"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()

        def on_granted() -> None:
            loop.call_soon_threadsafe(fut.set_result, None)

        def on_preempted() -> None:
            # IMPORTANT: the fut.done() check MUST run in the asyncio thread.
            # on_granted schedules fut.set_result via call_soon_threadsafe; if
            # on_preempted checked fut.done() here (in the RM thread) it would
            # see False, schedule set_exception, and then both callbacks fire in
            # the event loop — set_result first, set_exception second → raises
            # InvalidStateError.  Running the check inside the callback makes it
            # atomic with the action.
            def _handle() -> None:
                if not fut.done():
                    fut.set_exception(
                        asyncio.CancelledError(
                            f"[Client {self.rank}] resource slot {task_id} preempted before grant"
                        )
                    )
                else:
                    # Grant beat preemption in the event loop.  The slot is
                    # already freed by RM; signal the post-grant preemption path
                    # so _process_queue can flush the task with task_ids=None
                    # (avoiding a spurious release attempt).
                    if self.debug:
                        self.logger.warning(
                            f"[Client {self.rank}] resource slot {task_id} preempted while running"
                        )
                    if self._preempted_event is not None:
                        self._preempted_event.set()
            loop.call_soon_threadsafe(_handle)

        cfg = self.tasks_config[task_type]
        self._rm.request(
            task_id     = task_id,
            workflow_id = self.workflow_id,
            task_type   = task_type,
            priority    = int(cfg.get("priority", 10)),
            cpus        = int(cfg.get("cpus", 1)),
            gpus        = float(cfg.get("gpus", 1.0 / self.max_concurrent)),
            on_granted  = on_granted,
            on_preempted= on_preempted,
        )
        await fut

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------

    async def _process_queue(self):
        """Drain seq_queue, acquire resource slots, and dispatch requests."""
        batch_count = 0
        tasks:     list      = []
        task_ids:  list[str] = []
        batch_ids: list      = []

        while True:
            batch_id = await self.service.seq_queue.get()

            try:
                if batch_id is None:
                    self.logger.info(f"[Client {self.rank}] Received shutdown sentinel")
                    if tasks:
                        await self._flush_tasks(tasks, task_ids or None, batch_ids or None)
                    if self._pending_requeue:
                        if self.debug:
                            self.logger.info(
                                f"[Client {self.rank}] {len(self._pending_requeue)} batch(es) "
                                f"were re-queued during preemption — re-enqueuing and continuing"
                            )
                        for bid in self._pending_requeue:
                            await self.service.seq_queue.put(bid)
                        await self.service.seq_queue.put(None)  # new sentinel
                        self._pending_requeue.clear()
                        tasks = []; task_ids = []; batch_ids = []
                        continue
                    break

                batch_count += 1

                if self.endpoints:
                    endpoint = next(self.endpoint_cycle)
                    bd = self._get_batch_data(batch_id)

                    if self._rm is not None:
                        tid = f"batch_{batch_id}"
                        # Early flush: CPU slots exhausted by accumulated tasks.
                        if tasks and self._rm.available_cpus == 0:
                            if self.debug:
                                self.logger.debug(
                                    f"[Client {self.rank}] CPUs exhausted with "
                                    f"{len(tasks)} accumulated — flushing before next request"
                                )
                            await self._flush_tasks(tasks, task_ids or None, batch_ids or None)
                            tasks = []; task_ids = []; batch_ids = []
                        # Early flush: stale preemption event would cancel all
                        # buffered tasks at the next flush, including valid ones.
                        elif tasks and self._preempted_event is not None and self._preempted_event.is_set():
                            if self.debug:
                                self.logger.debug(
                                    f"[Client {self.rank}] Stale preemption event with "
                                    f"{len(tasks)} accumulated — flushing before next request"
                                )
                            # RM already revoked these slots during preemption —
                            # pass task_ids=None to skip redundant release attempts.
                            await self._flush_tasks(tasks, None, batch_ids or None)
                            tasks = []; task_ids = []; batch_ids = []

                        while True:
                            try:
                                await self._wait_for_resource(tid, "client_req")
                                break
                            except asyncio.CancelledError:
                                self.logger.warning(
                                    f"[Client {self.rank}] Resource slot for batch {batch_id} "
                                    f"cancelled before grant — retrying"
                                )
                                self._rm.cancel(tid)
                        # Post-grant check: preemption may have fired while we were
                        # suspended in _wait_for_resource.  Flush stale tasks NOW,
                        # before appending the freshly-granted slot, so the new task
                        # starts in a clean batch and is not caught by the next
                        # pre-check flush.
                        if tasks and self._preempted_event is not None and self._preempted_event.is_set():
                            if self.debug:
                                self.logger.debug(
                                f"[Client {self.rank}] Stale preemption detected after "
                                f"resource wait — flushing {len(tasks)} stale task(s) "
                                f"before batch_{batch_id}"
                                )
                            # RM already revoked these slots during preemption —
                            # pass task_ids=None to skip redundant release attempts.
                            await self._flush_tasks(tasks, None, batch_ids or None)
                            tasks = []; task_ids = []; batch_ids = []
                        task_ids.append(tid)

                    task = self.client_req(batch_id, endpoint, bd)
                    if batch_count == 1 and self.debug:
                        self.logger.debug(
                            f"[Client {self.rank}] Task type: {type(task)}, "
                            f"awaitable: {hasattr(task, '__await__')}"
                        )
                    tasks.append(task)
                    batch_ids.append(batch_id)

                    if self.debug and batch_count % 100 == 0:
                        self.logger.debug(
                            f"[Client {self.rank}] Dispatched {batch_count} batches"
                        )

                    if len(tasks) >= self.max_concurrent:
                        await self._flush_tasks(tasks, task_ids or None, batch_ids or None)
                        tasks = []; task_ids = []; batch_ids = []
                else:
                    # Local mode — submit directly to the service work queue.
                    self.service.work_queue.put_nowait((batch_id, None, None))

            finally:
                self.service.seq_queue.task_done()

        self.logger.info(f"[Client {self.rank}] Dispatched {batch_count} batches total")

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------

    async def _await_or_preempt(
        self,
        task,
        task_index: int,
        preempt_waiter: "Optional[asyncio.Task]",
    ) -> dict:
        """
        Await *task*, racing it against *preempt_waiter*.

        Raises ``asyncio.CancelledError`` if preemption fires first.
        Raises ``asyncio.TimeoutError`` if neither completes in time.
        """
        if asyncio.isfuture(task) or asyncio.iscoroutine(task):
            task_fut: asyncio.Future = asyncio.ensure_future(task)
        elif hasattr(task, "__await__"):
            async def _wrap():
                return await task
            task_fut = asyncio.ensure_future(_wrap())
        elif hasattr(task, "result"):
            task_fut = asyncio.ensure_future(
                asyncio.get_event_loop().run_in_executor(None, task.result)
            )
        else:
            self.logger.error(
                f"[Client {self.rank}] Task {task_index} is not awaitable: {type(task)}"
            )
            return {"status": "error", "error": f"Not awaitable: {type(task)}", "failed": 1}

        if preempt_waiter is None:
            return await asyncio.wait_for(task_fut, timeout=self.timeout)

        try:
            done, _ = await asyncio.wait(
                {task_fut, preempt_waiter},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.timeout,
            )
        except Exception:
            task_fut.cancel()
            raise

        if not done:
            task_fut.cancel()
            raise asyncio.TimeoutError()

        if preempt_waiter in done and task_fut not in done:
            task_fut.cancel()
            raise asyncio.CancelledError(
                f"[Client {self.rank}] preempted while awaiting task {task_index}"
            )

        # Task finished (preemption simultaneous with completion: task wins).
        return task_fut.result()

    async def _flush_tasks(
        self,
        tasks: list,
        task_ids: "Optional[list[str]]" = None,
        batch_ids: "Optional[list]" = None,
    ):
        """
        Await a batch of tasks and aggregate metrics.

        Each task is raced against ``_preempted_event`` so that post-grant
        preemption interrupts the current await immediately.

        On preemption:
        - Remaining RM slots (``task_ids[i+1:]``) are released.
        - Affected batch IDs (``batch_ids[i:]``) are re-queued for retry.
        - ``_preempted_event`` is cleared for future detections.
        """
        if not tasks:
            return

        self.logger.debug(f"[Client {self.rank}] Flushing {len(tasks)} tasks")

        preempt_waiter: Optional[asyncio.Task] = None
        if self._preempted_event is not None:
            preempt_waiter = asyncio.ensure_future(self._preempted_event.wait())

        results = []

        try:
            for i, task in enumerate(tasks):
                if self.debug:
                    self.logger.debug(
                        f"[Client {self.rank}] Awaiting task {i}/{len(tasks)}, "
                        f"type={type(task).__name__}"
                    )
                _task_was_preempted = False
                try:
                    result = await self._await_or_preempt(task, i, preempt_waiter)
                    if self.debug:
                        self.logger.debug(f"[Client {self.rank}] Task {i} completed: {result}")
                    results.append(result)
                except asyncio.CancelledError:
                    _task_was_preempted = True
                    requeue_count = len(tasks) - i
                    self.logger.warning(
                        f"[Client {self.rank}] Preempted at task {i} — "
                        f"re-queuing {requeue_count} batch(es) for retry"
                    )
                    if self._rm is not None and task_ids is not None:
                        for j in range(i + 1, len(task_ids)):
                            self._rm.release(task_ids[j])
                    if batch_ids is not None and self.service is not None:
                        for bid in batch_ids[i:]:
                            self._pending_requeue.append(bid)
                    if self._preempted_event is not None:
                        self._preempted_event.clear()
                    break
                except asyncio.TimeoutError:
                    self.logger.error(
                        f"[Client {self.rank}] Task {i} timed out after {self.timeout}s"
                    )
                    results.append({"status": "error", "error": "Timeout", "failed": 1})
                except Exception as e:
                    self.logger.error(
                        f"[Client {self.rank}] Task {i} failed with {type(e).__name__}: {e}"
                    )
                    import traceback
                    self.logger.debug(
                        f"[Client {self.rank}] Traceback: {traceback.format_exc()}"
                    )
                    results.append({"status": "error", "error": str(e), "failed": 1})
                finally:
                    if self._rm is not None and task_ids is not None and not _task_was_preempted:
                        self._rm.release(task_ids[i])
        finally:
            if preempt_waiter is not None and not preempt_waiter.done():
                preempt_waiter.cancel()

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
