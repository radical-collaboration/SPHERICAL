"""
NvmlMonitor — lightweight GPU/CPU monitor that runs directly in the main
process via a background thread, using pynvml (NVML C bindings).

Produces checkpoint files in the same JSON format as DragonTelemetryCollector
so the two datasets can be directly compared.

Usage
-----
    from src.utils.nvml_monitor import NvmlMonitor

    monitor = NvmlMonitor(
        output_dir="data/nvml-telemetry",
        collection_rate=1.0,      # seconds between samples
        checkpoint_interval=30.0, # seconds between checkpoint flushes
    )
    monitor.start()
    ...
    monitor.stop()
"""

import json
import os
import socket
import threading
import time
from pathlib import Path


def _try_nvml_import():
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


class NvmlMonitor:
    """Collects GPU and CPU metrics in a background thread via pynvml."""

    def __init__(
        self,
        output_dir: str = "data/nvml-telemetry",
        collection_rate: float = 1.0,
        checkpoint_interval: float = 30.0,
        metric_prefix: str = "SPHERICAL-nvml",
    ):
        self._output_dir = Path(output_dir)
        self._collection_rate = collection_rate
        self._checkpoint_interval = checkpoint_interval
        self._metric_prefix = metric_prefix
        self._hostname = socket.getfqdn()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._nvml = _try_nvml_import()
        self._nvml_ok = False
        self._gpu_handles: list = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._init_nvml()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="NvmlMonitor")
        self._thread.start()
        print(f"[NvmlMonitor] started → {self._output_dir}  "
              f"(GPUs={len(self._gpu_handles)}  rate={self._collection_rate}s)")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._collection_rate * 3)
        self._thread = None
        if self._nvml_ok:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False
        print("[NvmlMonitor] stopped")

    # ------------------------------------------------------------------
    def _init_nvml(self) -> None:
        if self._nvml is None:
            print("[NvmlMonitor] pynvml not available — GPU metrics disabled")
            return
        try:
            self._nvml.nvmlInit()
            n = self._nvml.nvmlDeviceGetCount()
            self._gpu_handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._nvml_ok = True
        except Exception as exc:
            print(f"[NvmlMonitor] nvmlInit failed: {exc} — GPU metrics disabled")

    # ------------------------------------------------------------------
    def _collect_sample(self) -> dict:
        ts = int(time.time())
        metrics: dict = {}

        # ── CPU ────────────────────────────────────────────────────────
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            la = os.getloadavg()
            metrics["load_average_1m"]  = round(la[0], 3)
            metrics["load_average_5m"]  = round(la[1], 3)
            metrics["load_average_15m"] = round(la[2], 3)
        except Exception:
            pass

        # ── GPU ────────────────────────────────────────────────────────
        if self._nvml_ok:
            for i, handle in enumerate(self._gpu_handles):
                prefix = f"gpu_{i}"
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"{prefix}_utilization"]        = util.gpu
                    metrics[f"{prefix}_memory_utilization"] = util.memory
                except Exception:
                    metrics[f"{prefix}_utilization"]        = -1
                    metrics[f"{prefix}_memory_utilization"] = -1

                try:
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gb = mem.total / (1024 ** 3)
                    used_gb  = mem.used  / (1024 ** 3)
                    metrics[f"{prefix}_memory_total_gb"]   = round(total_gb, 4)
                    metrics[f"{prefix}_memory_used_gb"]    = round(used_gb,  4)
                    metrics[f"{prefix}_memory_percent"]    = round(100 * mem.used / mem.total, 4) if mem.total else 0
                except Exception:
                    pass

                try:
                    power_mw = self._nvml.nvmlDeviceGetPowerUsage(handle)
                    metrics[f"{prefix}_power_watts"] = round(power_mw / 1000, 3)
                except Exception:
                    pass

        return {"timestamp": ts, "hostname": self._hostname, "metrics": metrics}

    # ------------------------------------------------------------------
    def _flush_checkpoint(self, samples: list, checkpoint_id: int) -> None:
        ts = int(time.time())
        data = {
            "checkpoint_metadata": {
                "hostname":          self._hostname,
                "timestamp":         ts,
                "collection_rate":   self._collection_rate,
                "metric_prefix":     self._metric_prefix,
                "checkpoint_id":     checkpoint_id,
                "num_samples":       len(samples),
            },
            "metrics": samples,
        }
        path = self._output_dir / f"nvml_checkpoint_{self._hostname}_{ts}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            print(f"[NvmlMonitor] checkpoint write failed: {exc}")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        samples: list = []
        checkpoint_id  = 0
        last_flush     = time.monotonic()

        # Prime psutil CPU measurement (first call returns 0.0)
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            try:
                samples.append(self._collect_sample())
            except Exception as exc:
                print(f"[NvmlMonitor] sample error: {exc}")

            elapsed = time.monotonic() - last_flush
            if elapsed >= self._checkpoint_interval and samples:
                self._flush_checkpoint(samples, checkpoint_id)
                checkpoint_id += 1
                samples = []
                last_flush = time.monotonic()

            # sleep for the remainder of the collection interval
            sleep_time = self._collection_rate - (time.monotonic() - t0)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

        # flush any remaining samples on shutdown
        if samples:
            self._flush_checkpoint(samples, checkpoint_id)
