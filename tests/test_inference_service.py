"""Tests for inference_service module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.inference.inference_service import GPUWorker, InferenceService


class ConcreteInferenceService(InferenceService):
    """Concrete implementation for testing abstract base class."""

    def _load_models(self):
        """Mock model loading."""
        self.models = {device: MagicMock() for device in self.devices}

    def process_batch_sync(self, batch_id: int, device: str):
        """Mock batch processing."""
        self.logger.metrics["total_tokens"] += 100
        self.reply_store[batch_id] = [f"result_{batch_id}"]
        self.processed_queue.put_nowait(batch_id)

    async def generate_batch(self) -> tuple:
        """Mock batch generation."""
        seq = await self.input_queue.get()
        if seq is None:
            await self.input_queue.put(None)
            raise StopAsyncIteration
        return 100, {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


class TestInferenceService:
    """Tests for InferenceService base class."""

    def test_init_requires_devices(self, sample_config):
        """Test that service requires devices."""
        with pytest.raises(ValueError, match="No devices provided"):
            ConcreteInferenceService(config=sample_config, devices=[])

    def test_init_with_devices(self, sample_config, temp_dir):
        """Test service initialization with devices."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(
            config=sample_config, devices=["cuda:0", "cuda:1"], rank=0
        )

        assert service.devices == ["cuda:0", "cuda:1"]
        assert service.rank == 0
        assert service.num_batches == sample_config["num_batches"]
        assert service.max_batch_tokens == sample_config["max_batch_tokens"]

    def test_init_metrics(self, sample_config, temp_dir):
        """Test metrics initialization."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        assert service.logger.metrics["requests"] == 0
        assert service.logger.metrics["total_tokens"] == 0
        assert service.logger.metrics["errors"] == 0
        assert "cuda:0" in service.logger.metrics["gpu_stats"]

    def test_init_queues(self, sample_config, temp_dir):
        """Test queue initialization."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        assert service.work_queue is not None
        assert service.seq_queue is not None
        assert service.input_queue is not None
        assert service.processed_queue is not None

    def test_init_pending_requests(self, sample_config, temp_dir):
        """Test pending requests initialization."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        assert service.pending_requests == {}
        assert service._request_counter == 0

    def test_configuration_from_config(self, sample_config, temp_dir):
        """Test that configuration is read from config dict."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")
        sample_config["num_workers_per_gpu"] = 4
        sample_config["debug"] = True

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        assert service.num_workers_per_gpu == 4
        assert service.debug is True


class TestInferenceServiceAsync:
    """Async tests for InferenceService."""

    @pytest.mark.asyncio
    async def test_start_workers(self, sample_config, temp_dir):
        """Test starting workers."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")
        sample_config["num_workers_per_gpu"] = 1

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        await service.start_workers()

        assert len(service.workers) == 1
        assert len(service.worker_tasks) >= 1  # Workers + metrics task

        # Cleanup
        service.shutting_down.set()
        await service.work_queue.put(None)
        await asyncio.gather(*service.worker_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_submit_batch_returns_future(self, sample_config, temp_dir):
        """Test submitting a batch returns a Future."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        future = service.submit_batch(batch_id=1)

        # Check it returns a Future
        assert isinstance(future, asyncio.Future)

        # Check batch was added to work queue
        item = await service.work_queue.get()
        batch_id, request_id = item
        assert batch_id == 1
        assert request_id == 1  # First request

        # Check pending request was stored
        # Note: we already popped from work queue, but counter was incremented
        assert service._request_counter == 1

    @pytest.mark.asyncio
    async def test_submit_batch_increments_counter(self, sample_config, temp_dir):
        """Test that submit_batch increments request counter."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        service.submit_batch(batch_id=1)
        service.submit_batch(batch_id=2)
        service.submit_batch(batch_id=3)

        assert service._request_counter == 3

    @pytest.mark.asyncio
    async def test_resolve_request(self, sample_config, temp_dir):
        """Test resolving a pending request."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        future = service.submit_batch(batch_id=1)

        # Resolve the request
        service._resolve_request(1, {"status": "success"})

        # Future should now be done
        assert future.done()
        result = future.result()
        assert result == {"status": "success"}

    @pytest.mark.asyncio
    async def test_sequence_source(self, sample_config, temp_dir):
        """Test sequence source generator."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        sequences = []
        count = 0
        async for seq in service.sequence_source():
            sequences.append(seq)
            count += 1
            if count >= 3:
                service.shutdown_init.set()
                break

        assert len(sequences) == 3

    @pytest.mark.asyncio
    async def test_close_alias(self, sample_config, temp_dir):
        """Test that close is an alias for shutdown."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        # Mock shutdown to verify close calls it
        service.shutdown = AsyncMock()

        await service.close()

        service.shutdown.assert_called_once()


class TestGPUWorker:
    """Tests for GPUWorker class."""

    def test_init(self, sample_config, temp_dir):
        """Test GPUWorker initialization."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        worker = GPUWorker(device="cuda:0", worker_id=0, service=service)

        assert worker.device == "cuda:0"
        assert worker.worker_id == 0
        assert worker.service is service
        assert worker.is_busy is False
        assert worker.processed_count == 0

    @pytest.mark.asyncio
    async def test_worker_processes_batch(self, sample_config, temp_dir):
        """Test worker processes batches from queue."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")
        sample_config["debug"] = False

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        worker = GPUWorker(device="cuda:0", worker_id=0, service=service)

        # Put a batch with request_id and shutdown signal
        await service.work_queue.put((1, 42))  # batch_id=1, request_id=42
        await service.work_queue.put(None)

        # Create a future for the request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        service.pending_requests[42] = future

        # Run worker
        await worker.run()

        # Check processing occurred
        assert worker.processed_count == 1
        assert service.logger.metrics["requests"] == 1
        assert service.logger.metrics["gpu_stats"]["cuda:0"]["processed"] == 1

        # Check future was resolved
        assert future.done()
        assert future.result() == {"status": "success"}

    @pytest.mark.asyncio
    async def test_worker_handles_local_mode(self, sample_config, temp_dir):
        """Test worker handles None request_id (local mode)."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")
        sample_config["debug"] = False

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        worker = GPUWorker(device="cuda:0", worker_id=0, service=service)

        # Put a batch with None request_id (local mode)
        await service.work_queue.put((1, None))
        await service.work_queue.put(None)

        # Run worker
        await worker.run()

        # Check processing occurred
        assert worker.processed_count == 1
        assert service.logger.metrics["requests"] == 1

    @pytest.mark.asyncio
    async def test_worker_handles_shutdown(self, sample_config, temp_dir):
        """Test worker handles shutdown signal."""
        sample_config["output_dir"] = str(temp_dir / "outputs")
        sample_config["metrics_dir"] = str(temp_dir / "metrics")
        sample_config["debug"] = True

        service = ConcreteInferenceService(config=sample_config, devices=["cuda:0"], rank=0)

        worker = GPUWorker(device="cuda:0", worker_id=0, service=service)

        # Put only shutdown signal
        await service.work_queue.put(None)

        # Run worker - should exit cleanly
        await worker.run()

        assert worker.processed_count == 0
