"""Tests for CampaignManager and AsyncCampaignManager."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.campaign import AsyncCampaignManager, BaseWorkflow, CampaignManager, ResourcePool, WorkflowStats

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Run all async tests with the asyncio backend only."""
    return "asyncio"


# ---------------------------------------------------------------------------
# Workflow stubs
# ---------------------------------------------------------------------------


class NullWorkflow(BaseWorkflow):
    """No-op async workflow."""

    workflow_id = "null"

    async def run(self, replica_id: str) -> None:
        pass


class SleepWorkflow(BaseWorkflow):
    """Sleeps briefly so concurrency effects are observable."""

    workflow_id = "sleep"

    async def run(self, replica_id: str) -> None:
        await asyncio.sleep(0.02)


class RecordingWorkflow(BaseWorkflow):
    """Appends each replica_id it runs to a class-level list."""

    workflow_id = "recording"
    ran: list = []

    async def run(self, replica_id: str) -> None:
        RecordingWorkflow.ran.append(replica_id)


class SignalWorkflow(BaseWorkflow):
    """Fires _signal_ready() immediately then finishes after a tiny sleep."""

    workflow_id = "signal"

    async def run(self, replica_id: str) -> None:
        await self._signal_ready()
        await asyncio.sleep(0.01)


class HookWorkflow(BaseWorkflow):
    """Records (replica_id, final_state) tuples in on_replica_done."""

    workflow_id = "hook"
    calls: list = []

    async def run(self, replica_id: str) -> None:
        pass

    async def on_replica_done(self, replica_id, cm, final_state):
        HookWorkflow.calls.append((replica_id, final_state))


# Sync variants for CampaignManager (thread-pool) tests


class SyncRecordingWorkflow(BaseWorkflow):
    workflow_id = "sync_rec"
    ran: list = []

    def run(self, replica_id: str) -> None:
        SyncRecordingWorkflow.ran.append(replica_id)


class SyncHookWorkflow(BaseWorkflow):
    workflow_id = "sync_hook"
    calls: list = []

    def run(self, replica_id: str) -> None:
        pass

    def on_replica_done(self, replica_id, cm, final_state):
        SyncHookWorkflow.calls.append((replica_id, final_state))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_class_state():
    """Clear class-level recording lists before every test."""
    RecordingWorkflow.ran = []
    HookWorkflow.calls = []
    SyncRecordingWorkflow.ran = []
    SyncHookWorkflow.calls = []
    yield


@pytest.fixture
async def acm():
    """AsyncCampaignManager with asyncflow initialization mocked out."""
    cm = AsyncCampaignManager()
    mock_af = AsyncMock()

    async def _fake_init():
        cm._asyncflow = mock_af

    cm._init_asyncflow = _fake_init
    yield cm
    await cm.close()


# ---------------------------------------------------------------------------
# BaseWorkflow
# ---------------------------------------------------------------------------


class TestBaseWorkflow:
    async def test_signal_ready_no_callback_is_noop(self):
        wf = NullWorkflow()
        await wf._signal_ready()  # must not raise

    async def test_signal_ready_calls_sync_callback(self):
        called = []
        wf = NullWorkflow(on_ready=lambda: called.append(1))
        await wf._signal_ready()
        assert called == [1]

    async def test_signal_ready_awaits_async_callback(self):
        called = []

        async def cb():
            called.append(1)

        wf = NullWorkflow(on_ready=cb)
        await wf._signal_ready()
        assert called == [1]

    def test_base_run_raises_not_implemented(self):
        wf = BaseWorkflow()
        with pytest.raises(NotImplementedError):
            wf.run("r0")


# ---------------------------------------------------------------------------
# AsyncCampaignManager
# ---------------------------------------------------------------------------


class TestAsyncCampaignManager:
    async def test_single_replica_completes(self, acm):
        acm.register_group("a", NullWorkflow, replicas=1)
        await acm.start()
        assert await acm.wait(timeout=3.0)
        s = acm.status()
        assert s["groups"]["a"]["status"] == "done"
        assert s["groups"]["a"]["replicas_finished"] == 1

    async def test_all_replicas_run(self, acm):
        acm.register_group("a", RecordingWorkflow, replicas=4, max_replicas=4)
        await acm.start()
        assert await acm.wait(timeout=3.0)
        assert sorted(RecordingWorkflow.ran) == ["a_0", "a_1", "a_2", "a_3"]

    async def test_max_replicas_cap_respected(self, acm):
        """Concurrent running count must never exceed max_replicas."""
        peak = []

        class PeakObserver(BaseWorkflow):
            workflow_id = "peak"
            _active = 0

            async def run(self, replica_id: str) -> None:
                PeakObserver._active += 1
                peak.append(PeakObserver._active)
                await asyncio.sleep(0.02)
                PeakObserver._active -= 1

        acm.register_group("a", PeakObserver, replicas=6, max_replicas=2)
        await acm.start()
        assert await acm.wait(timeout=5.0)
        assert max(peak) <= 2

    async def test_dependency_count_based(self, acm):
        """Group B must not start until A has dep_threshold finished replicas."""
        order = []

        class A(BaseWorkflow):
            workflow_id = "A"

            async def run(self, replica_id: str) -> None:
                order.append(("A", replica_id))

        class B(BaseWorkflow):
            workflow_id = "B"

            async def run(self, replica_id: str) -> None:
                order.append(("B", replica_id))

        acm.register_group("a", A, replicas=2)
        acm.register_group("b", B, replicas=1, dependencies=["a"], dep_threshold=2)
        await acm.start()
        assert await acm.wait(timeout=3.0)

        b_idx = next(i for i, (wf, _) in enumerate(order) if wf == "B")
        assert all(wf == "A" for wf, _ in order[:b_idx])

    async def test_dependency_via_signal_ready(self, acm):
        """_signal_ready() unblocks B even before all of A's replicas finish."""
        acm.register_group("a", SignalWorkflow, replicas=1)
        acm.register_group(
            "b", NullWorkflow, replicas=1,
            dependencies=["a"],
            dep_threshold=999,  # count-based fallback would never fire
        )
        await acm.start()
        assert await acm.wait(timeout=3.0)

        s = acm.status()
        assert s["groups"]["a"]["ready"] is True
        assert s["groups"]["b"]["status"] == "done"

    async def test_on_replica_done_hook_called(self, acm):
        acm.register_group("a", HookWorkflow, replicas=2)
        await acm.start()
        assert await acm.wait(timeout=3.0)
        assert len(HookWorkflow.calls) == 2
        assert {rid for rid, _ in HookWorkflow.calls} == {"a_0", "a_1"}
        assert all(st == "done" for _, st in HookWorkflow.calls)

    async def test_empty_campaign_finishes_immediately(self, acm):
        await acm.start()
        assert await acm.wait(timeout=1.0)

    async def test_status_snapshot_fields(self, acm):
        acm.register_group("a", NullWorkflow, replicas=2, max_replicas=1, priority=7)
        s = acm.status()["groups"]["a"]
        assert s["status"] == "pending"
        assert s["replicas_total"] == 2
        assert s["max_replicas"] == 1
        assert s["priority"] == 7
        assert s["dependencies"] == []

    async def test_stats_reflect_finished_count(self, acm):
        acm.register_group("a", NullWorkflow, replicas=3)
        await acm.start()
        assert await acm.wait(timeout=3.0)
        st = acm.stats()
        assert st["a"].replicas_started == 3
        assert st["a"].replicas_finished == 3
        assert isinstance(st["a"], WorkflowStats)

    async def test_from_config_registers_groups(self):
        config = {
            "workflows": {
                "x": {"replicas": 2, "max_replicas": 1, "priority": 3},
                "y": {"replicas": 1, "dependencies": ["x"], "dependency_threshold": 2},
            }
        }
        cm = AsyncCampaignManager.from_config(config, {"x": NullWorkflow, "y": NullWorkflow})
        s = cm.status()["groups"]
        assert s["x"]["replicas_total"] == 2
        assert s["x"]["max_replicas"] == 1
        assert s["x"]["priority"] == 3
        assert s["y"]["dependencies"] == ["x"]
        assert s["y"]["dep_threshold"] == 2

    async def test_add_replicas_updates_total(self, acm):
        """add_replicas raises group.replicas up to configured_replicas cap."""
        acm.register_group("a", NullWorkflow, replicas=1)
        acm._groups["a"].configured_replicas = 3  # widen the cap
        assert acm.status()["groups"]["a"]["replicas_total"] == 1

        await acm.add_replicas("a", 2)

        assert acm.status()["groups"]["a"]["replicas_total"] == 3

    async def test_add_replicas_ignores_beyond_cap(self, acm):
        """Requests beyond configured_replicas are silently ignored."""
        acm.register_group("a", NullWorkflow, replicas=2)
        # configured_replicas == replicas == 2, so add_replicas is a no-op
        await acm.add_replicas("a", 5)
        assert acm.status()["groups"]["a"]["replicas_total"] == 2

    async def test_unknown_group_skipped_in_from_config(self):
        config = {"workflows": {"unknown": {"replicas": 1}}}
        cm = AsyncCampaignManager.from_config(config, {})  # empty registry
        assert "unknown" not in cm.status()["groups"]


# ---------------------------------------------------------------------------
# CampaignManager (sync / thread-pool)
# ---------------------------------------------------------------------------


class TestCampaignManager:
    @pytest.fixture
    def cm(self):
        manager = CampaignManager()
        yield manager
        manager.close()

    def test_single_replica_runs(self, cm):
        cm.register_group("a", SyncRecordingWorkflow, replicas=1)
        cm.start()
        assert cm.wait(timeout=5.0)
        assert SyncRecordingWorkflow.ran == ["a_0"]

    def test_multiple_replicas_all_run(self, cm):
        cm.register_group("a", SyncRecordingWorkflow, replicas=3)
        cm.start()
        assert cm.wait(timeout=5.0)
        assert sorted(SyncRecordingWorkflow.ran) == ["a_0", "a_1", "a_2"]

    def test_sliding_window_max_replicas(self, cm):
        cm.register_group("a", SyncRecordingWorkflow, replicas=4, max_replicas=2)
        cm.start()
        assert cm.wait(timeout=5.0)
        assert sorted(SyncRecordingWorkflow.ran) == ["a_0", "a_1", "a_2", "a_3"]

    def test_dependency_respected(self, cm):
        """Group B must start only after group A completes."""
        order = []

        class A(BaseWorkflow):
            workflow_id = "A"

            def run(self, replica_id: str) -> None:
                order.append(("A", replica_id))

        class B(BaseWorkflow):
            workflow_id = "B"

            def run(self, replica_id: str) -> None:
                order.append(("B", replica_id))

        cm.register_group("a", A, replicas=2)
        cm.register_group("b", B, replicas=1, dependencies=["a"])
        cm.start()
        assert cm.wait(timeout=5.0)

        b_idx = next(i for i, (wf, _) in enumerate(order) if wf == "B")
        assert all(wf == "A" for wf, _ in order[:b_idx])

    def test_on_replica_done_hook_called(self, cm):
        cm.register_group("a", SyncHookWorkflow, replicas=2)
        cm.start()
        assert cm.wait(timeout=5.0)
        assert len(SyncHookWorkflow.calls) == 2
        assert {rid for rid, _ in SyncHookWorkflow.calls} == {"a_0", "a_1"}

    def test_status_snapshot_fields(self, cm):
        cm.register_group("a", SyncRecordingWorkflow, replicas=1, priority=5, max_replicas=1)
        s = cm.status()["groups"]["a"]
        assert s["status"] == "pending"
        assert s["priority"] == 5
        assert s["replicas_total"] == 1
        assert s["max_replicas"] == 1

    def test_stats_reflect_finished_count(self, cm):
        cm.register_group("a", SyncRecordingWorkflow, replicas=3)
        cm.start()
        assert cm.wait(timeout=5.0)
        st = cm.stats()
        assert st["a"].replicas_finished == 3
        assert isinstance(st["a"], WorkflowStats)

    def test_from_config_registers_groups(self):
        config = {
            "workflows": {
                "alpha": {"replicas": 3, "max_replicas": 2, "priority": 7},
            }
        }
        cm = CampaignManager.from_config(config, {"alpha": SyncRecordingWorkflow})
        s = cm.status()["groups"]
        cm.close()
        assert "alpha" in s
        assert s["alpha"]["replicas_total"] == 3
        assert s["alpha"]["max_replicas"] == 2
        assert s["alpha"]["priority"] == 7

    def test_unknown_group_skipped_in_from_config(self):
        config = {"workflows": {"ghost": {"replicas": 1}}}
        cm = CampaignManager.from_config(config, {})
        cm.close()
        assert "ghost" not in cm.status()["groups"]


# ---------------------------------------------------------------------------
# ResourcePool unit tests
# ---------------------------------------------------------------------------


class TestResourcePool:
    def test_initial_available_equals_total(self):
        rp = ResourcePool(total_cpus=16, total_gpus=4)
        assert rp.available_cpus == 16
        assert rp.available_gpus == 4

    def test_can_fit_within_budget(self):
        rp = ResourcePool(total_cpus=8, total_gpus=2)
        assert rp.can_fit(8, 2)
        assert rp.can_fit(1, 0)
        assert rp.can_fit(0, 1)

    def test_cannot_fit_over_budget(self):
        rp = ResourcePool(total_cpus=4, total_gpus=1)
        assert not rp.can_fit(5, 0)
        assert not rp.can_fit(0, 2)

    def test_zero_total_means_unlimited(self):
        rp = ResourcePool(total_cpus=0, total_gpus=0)
        assert rp.can_fit(9999, 9999)

    def test_allocate_decrements_available(self):
        rp = ResourcePool(total_cpus=16, total_gpus=4)
        rp.allocate(4, 1)
        assert rp.available_cpus == 12
        assert rp.available_gpus == 3

    def test_release_increments_available(self):
        rp = ResourcePool(total_cpus=16, total_gpus=4)
        rp.allocate(4, 1)
        rp.release(4, 1)
        assert rp.available_cpus == 16
        assert rp.available_gpus == 4

    def test_as_dict_keys(self):
        rp = ResourcePool(total_cpus=8, total_gpus=2)
        d = rp.as_dict()
        assert set(d) == {"total_cpus", "available_cpus", "total_gpus", "available_gpus"}


# ---------------------------------------------------------------------------
# Resource-aware scheduling tests (AsyncCampaignManager)
# ---------------------------------------------------------------------------


class TestAsyncCampaignManagerResources:
    @pytest.fixture
    async def racm(self):
        """AsyncCampaignManager with 4 CPUs and 2 GPUs, asyncflow mocked."""
        cm = AsyncCampaignManager(total_cpus=4, total_gpus=2)
        mock_af = AsyncMock()

        async def _fake_init():
            cm._asyncflow = mock_af

        cm._init_asyncflow = _fake_init
        yield cm
        await cm.close()

    async def test_resource_limits_concurrency(self, racm):
        """With 2 GPUs and 1 GPU/replica, at most 2 replicas run concurrently."""
        peak = []

        class GpuWorkflow(BaseWorkflow):
            workflow_id = "gpu"
            _active = 0

            async def run(self, replica_id: str) -> None:
                GpuWorkflow._active += 1
                peak.append(GpuWorkflow._active)
                await asyncio.sleep(0.02)
                GpuWorkflow._active -= 1

        racm.register_group("g", GpuWorkflow, replicas=6, max_replicas=6, required_gpus=1)
        await racm.start()
        assert await racm.wait(timeout=5.0)
        assert max(peak) <= 2  # only 2 GPUs available

    async def test_resources_released_after_replica(self, racm):
        """Available resources return to full after all replicas complete."""
        racm.register_group("g", NullWorkflow, replicas=2, required_cpus=2, required_gpus=1)
        await racm.start()
        assert await racm.wait(timeout=3.0)
        s = racm.status()["resources"]
        assert s["available_cpus"] == 4   # total_cpus restored
        assert s["available_gpus"] == 2   # total_gpus restored

    async def test_status_includes_resource_snapshot(self, racm):
        racm.register_group("g", NullWorkflow, replicas=1, required_cpus=2, required_gpus=1)
        s = racm.status()
        assert "resources" in s
        assert s["resources"]["total_cpus"] == 4
        assert s["resources"]["total_gpus"] == 2
        assert s["resources"]["available_cpus"] == 4
        assert s["resources"]["available_gpus"] == 2

    async def test_from_config_parses_resources(self):
        config = {
            "resources": {"total_cpus": 64, "total_gpus": 8},
            "workflows": {
                "a": {"replicas": 1, "required_cpus": 4, "required_gpus": 2},
            },
        }
        cm = AsyncCampaignManager.from_config(config, {"a": NullWorkflow})
        s = cm.status()
        assert s["resources"]["total_cpus"] == 64
        assert s["resources"]["total_gpus"] == 8
        assert s["groups"]["a"]["required_cpus"] == 4
        assert s["groups"]["a"]["required_gpus"] == 2

    async def test_resource_constrained_priority_ordering(self, racm):
        """High-priority group fills available GPU slots before low-priority."""
        started_order = []

        class TrackWorkflow(BaseWorkflow):
            workflow_id = "track"

            async def run(self, replica_id: str) -> None:
                started_order.append(replica_id)
                await asyncio.sleep(0.01)

        racm.register_group("lo", TrackWorkflow, replicas=2, priority=1, required_gpus=1)
        racm.register_group("hi", TrackWorkflow, replicas=2, priority=9, required_gpus=1)
        await racm.start()
        assert await racm.wait(timeout=3.0)
        # First two started should be the high-priority group
        assert started_order[0].startswith("hi")
        assert started_order[1].startswith("hi")


# ---------------------------------------------------------------------------
# Resource-aware scheduling tests (CampaignManager sync)
# ---------------------------------------------------------------------------


class TestCampaignManagerResources:
    @pytest.fixture
    def rcm(self):
        cm = CampaignManager(total_cpus=4, total_gpus=2)
        yield cm
        cm.close()

    def test_resource_limits_concurrency(self, rcm):
        """With 2 GPUs and 1 GPU/replica, at most 2 run concurrently."""
        import threading
        peak = []
        lock = threading.Lock()

        class GpuWorkflow(BaseWorkflow):
            workflow_id = "gpu"
            _active = 0

            def run(self, replica_id: str) -> None:
                with lock:
                    GpuWorkflow._active += 1
                    peak.append(GpuWorkflow._active)
                import time; time.sleep(0.02)
                with lock:
                    GpuWorkflow._active -= 1

        rcm.register_group("g", GpuWorkflow, replicas=6, max_replicas=6, required_gpus=1)
        rcm.start()
        assert rcm.wait(timeout=5.0)
        assert max(peak) <= 2

    def test_resources_released_after_replica(self, rcm):
        rcm.register_group("g", SyncRecordingWorkflow, replicas=2,
                            required_cpus=2, required_gpus=1)
        rcm.start()
        assert rcm.wait(timeout=3.0)
        s = rcm.status()["resources"]
        assert s["available_cpus"] == 4
        assert s["available_gpus"] == 2

    def test_status_includes_resource_snapshot(self, rcm):
        rcm.register_group("g", SyncRecordingWorkflow, replicas=1,
                            required_cpus=1, required_gpus=0)
        s = rcm.status()
        assert "resources" in s
        assert s["resources"]["total_cpus"] == 4
        assert s["resources"]["total_gpus"] == 2

    def test_from_config_parses_resources(self):
        config = {
            "resources": {"total_cpus": 32, "total_gpus": 4},
            "workflows": {
                "a": {"replicas": 1, "required_cpus": 8, "required_gpus": 1},
            },
        }
        cm = CampaignManager.from_config(config, {"a": SyncRecordingWorkflow})
        s = cm.status()
        cm.close()
        assert s["resources"]["total_cpus"] == 32
        assert s["resources"]["total_gpus"] == 4
        assert s["groups"]["a"]["required_cpus"] == 8
        assert s["groups"]["a"]["required_gpus"] == 1
