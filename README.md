# Spherical

Multi-GPU Inference Service Framework with Worker Pool Management.

## Features

- **Multi-GPU Support**: Automatic load balancing across multiple GPUs
- **Automatic Device Detection**: Detects CUDA GPUs if available, falls back to CPU
- **Worker Pool Management**: Configurable workers per device
- **Async Architecture**: Built on asyncio for high throughput
- **HTTP Server/Client**: aiohttp-based server with health checks
- **Dragon/Asyncflow Integration**: Optional HPC runtime support for distributed execution
- **Metrics Collection**: Real-time throughput and device utilization tracking
- **Extensible**: Base classes for adding new model types

## Installation

```bash
# Basic installation
pip install -e .

# With ESM2 model support
pip install -e ".[esm2]"

# With Dragon/RADICAL support
pip install -e ".[dragon]"

# With development dependencies
pip install -e ".[dev]"

# Full installation
pip install -e ".[esm2,dragon,dev,plotting]"
```

## Quick Start

### Running the ESM2 Example

```bash
# Start server mode (with HTTP endpoints)
python example/esm2/run_esm2_inference.py --mode server --config_file example/esm2/config.yaml

# Run local inference (no server)
python example/esm2/run_esm2_inference.py --mode local --config_file example/esm2/config.yaml
```

### Configuration

Edit `example/esm2/config.yaml` to configure:

```yaml
# Model Settings
model_path: "facebook/esm2_t33_650M_UR50D"

# GPU Configuration
num_services: 1
num_gpus_per_service: 4
num_workers_per_gpu: 2

# Server Settings
server_port: 8000

# Batch Settings
num_batches: 200
max_batch_tokens: 16000

# Execution Settings
debug: true
engine: dragon      # Enable Dragon HPC runtime
```

## Architecture

```
spherical/
├── src/                       # Core library
│   ├── inference_service.py   # Base inference service + GPU workers
│   ├── server.py              # HTTP server endpoints
│   ├── orchestrator.py        # Multi-node coordination
│   ├── logger.py              # Logging utilities
│   └── utils.py               # Helper functions
├── example/
│   └── esm2/                  # ESM2 example
│       ├── client.py          # HTTP client with load balancing
│       ├── esm2_service.py    # ESM2 service (re-export)
│       ├── run_esm2_inference.py  # Entry point
│       └── config.yaml        # Configuration
├── tests/                     # Unit tests
└── doc/                       # Documentation
```

## Extending for New Models

Create a new service by extending `InferenceService`:

```python
from src.inference_service import InferenceService

class MyModelService(InferenceService):
    def _load_models(self):
        """Load your model onto GPUs."""
        for device in self.devices:
            self.models[device] = load_model().to(device)

    def process_batch_sync(self, batch_id: int, device: str):
        """Run inference on a batch."""
        model = self.models[device]
        # Process batch...
        self.reply_store[batch_id] = results
        self.processed_queue.put_nowait(batch_id)

    async def generate_batch(self) -> tuple:
        """Generate batches from input queue."""
        seq = await self.input_queue.get()
        if seq is None:
            raise StopAsyncIteration
        batch = tokenize(seq)
        return len(batch), batch
```

## Dragon/Asyncflow Support

For HPC environments, Spherical supports Dragon runtime with asyncflow:

```yaml
# Enable in config.yaml
engine: dragon
dragon_workers: 100
```

Run with Dragon:
```bash
dragon -w ssh --network-config slurm.yaml run_esm2_infern.py
```

## Metrics & Visualization

Two plotting scripts live in `src/plot/`:

| Script | Input | Use case |
|--------|-------|----------|
| `src/plot/plot_dragon.py` | Dragon telemetry JSON (`checkpoint_metadata` + `metrics[]`) and/or inference `metrics_*.json` | ESM2 inference runs, Dragon campaign runs |
| `src/plot/plot_nvml.py` | NVML telemetry JSON (`nvml_checkpoint_*.json`) | SGDES and any workflow using `NvmlMonitor` |

### Dragon telemetry — standalone mode

Plot GPU/CPU utilization from a single telemetry directory:

```bash
python src/plot/plot_dragon.py --telemetry-dir outputs/telemetry-results
```

### Dragon telemetry — multi-run mode

Scan a parent directory for per-run output subdirectories, generate one plot
per run and a throughput-vs-GPUs summary chart:

```bash
python src/plot/plot_dragon.py --output-dirs outputs --plots-dir plots
```

Each subdirectory may contain `metrics_*.json` (throughput timeseries) and/or
a `telemetry-results/` subdirectory (GPU/CPU utilization).

### NVML telemetry

Plot GPU utilization and memory from NVML checkpoint files:

```bash
python src/plot/plot_nvml.py --telemetry-dir nvml-telemetry --output gpu_util.png
```

Prints a per-GPU summary table (mean/max utilization and memory) to stdout.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Lint and format code
ruff check .
ruff format .
```

## License

MIT License
