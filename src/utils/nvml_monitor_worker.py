#!/usr/bin/env python3
"""
Standalone NvmlMonitor worker — launched as a subprocess by start_node_telemetry.

Runs independently of Dragon worker processes so it is not killed when Dragon
recycles the worker that started it.  Collects GPU metrics until SIGTERM/SIGINT,
writing a final checkpoint before exit.  Periodic checkpoints are also written
every --checkpoint-interval seconds so data survives even if the signal is never
received (e.g. SLURM wall-time kill via SIGKILL).

Usage (internal — called by start_node_telemetry):
    python nvml_monitor_worker.py \
        --outdir /path/to/nvml-telemetry \
        --rate 1.0 \
        --checkpoint-interval 30.0 \
        --spherical-dir /path/to/SPHERICAL
"""
import argparse
import os
import signal
import socket
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="NvmlMonitor worker subprocess")
    parser.add_argument("--outdir", required=True, help="Output directory for checkpoint files")
    parser.add_argument("--rate", type=float, default=1.0, help="Seconds between NVML samples")
    parser.add_argument("--checkpoint-interval", type=float, default=30.0,
                        help="Seconds between checkpoint flushes")
    parser.add_argument("--spherical-dir", default="",
                        help="SPHERICAL repo root (added to sys.path)")
    args = parser.parse_args()

    if args.spherical_dir and args.spherical_dir not in sys.path:
        sys.path.insert(0, args.spherical_dir)

    from src.utils.nvml_monitor import NvmlMonitor

    mon = NvmlMonitor(
        output_dir=args.outdir,
        collection_rate=args.rate,
        checkpoint_interval=args.checkpoint_interval,
    )

    def _shutdown(signum, frame):
        mon.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    mon.start()

    # Write PID file so stop_node_telemetry can signal us from any Dragon worker.
    hostname = socket.gethostname()
    pid_file = os.path.join(args.outdir, f"nvml_pid_{hostname}.txt")
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        print(f"[nvml_monitor_worker] WARNING: could not write PID file {pid_file}: {exc}")

    # Block until signaled.
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        mon.stop()
        try:
            os.remove(pid_file)
        except OSError:
            pass


if __name__ == "__main__":
    main()
