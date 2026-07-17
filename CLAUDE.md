# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Note**: The `AsyncCampaignManager` and all campaign orchestration code has been moved to
> `/scratch/bblj/mgoliyad1/campaign_manager`. This repo now contains only the
> multi-GPU inference service framework and SGDES workflow.

## Quick Start

### Installation

```bash
# Core installation (inference + utils)
pip install -e .

# With ESM2 model support (torch + transformers)
pip install -e ".[esm2]"

# With Dragon/RADICAL HPC support
pip install -e ".[dragon]"

# Full dev setup
pip install -e ".[esm2,dragon,dev]"
```

Requires **Python ≥ 3.10**.

### Common Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run a single test file or test
pytest tests/test_inference_service.py
pytest tests/test_inference_service.py::TestClass::test_method

# Lint and format check
ruff check .
ruff format --check .

# Auto-format code
ruff format .
ruff check . --fix
```

### Key Entry Points

- **ESM2 inference (standalone)**: `python workflows/esm2_inference/run_esm2_infern.py --config workflows/esm2_inference/inference.yaml --mode local`
- **SGDES workflow**: `python workflows/sgdes/run_workflow.py`
- **SLURM submission (Delta GPU)**: `sbatch workflows/esm2_inference/delta_gpu_sbatch.sh`

---

## Architecture Overview

SPHERICAL provides a **multi-GPU inference service framework** for ESM2 protein language model embeddings,
built on `radical.asyncflow`. Workflows submit sequences to inference services running on one or more GPUs;
results feed downstream molecular dynamics or ML pipelines.

### High-Level Design

```
src/inference/
├── esm2_service/
│   ├── esm2_service.py      ESM2InferenceService — loads model, runs batched inference
│   └── esm2_client.py       ESM2Client — submits sequences, collects embeddings
├── inference_service.py     BaseInferenceService — async queue-based service skeleton
├── inference_client.py      BaseInferenceClient — client protocol
├── orchestrator.py          start_services() / start_services_local() — spin up N service instances
├── server.py                aiohttp HTTP server wrapping a service
├── dragon_launcher.py       Dragon backend launcher for HPC nodes
├── utils.py                 shared helpers (queue drain, batch sizing, etc.)
└── kill_all.py              emergency cleanup for zombie processes

src/utils/
├── logger.py                colored structured logging with metrics recording
└── workflow.py              _expand_env(), load_config(), find_gpus(), make_policies()
```

### Core Concepts

#### ESM2 Inference Service

Each `ESM2InferenceService` instance owns one GPU. The orchestrator starts N instances (one per GPU):

```python
handles = await start_services_local(config, ESM2InferenceService)
# handles[i].service  → ESM2InferenceService on GPU i
# handles[i].endpoint → HTTP URL (server mode)
# handles[i].close()  → teardown
```

Services expose three asyncio queues:
- `input_queue`: caller puts `(batch_id, sequences)`
- `processed_queue`: service puts `(batch_id, embeddings)` after inference
- `work_queue`: internal task tracking (joined for backpressure)

#### ESM2 Client

`ESM2Client` serialises access to one service via an `asyncio.Lock`. It:
1. Drains any in-flight work (`work_queue.join()`, `processed_queue.join()`)
2. Resets queue state for a clean run
3. Submits sequences → collects embeddings
4. Writes outputs to `config["output_dir"]`

#### Server Mode

`server.py` wraps a service with an aiohttp HTTP interface. Clients POST sequences and GET results.
Used for cross-node communication in Dragon/HPC deployments.

#### Resource Helpers (`src/utils/workflow.py`)

- `_expand_env(value)` — expands `${VAR}` in config strings
- `load_config(path)` — loads YAML with env expansion
- `find_gpus(node)` — enumerates GPUs on a Dragon node
- `make_policies(gpu_ids)` — builds Dragon `Policy` objects for GPU affinity

---

## Key File Organization

### Inference Framework (`src/inference/`)

| File | Purpose |
|------|---------|
| `esm2_service/esm2_service.py` | ESM2 model loading + batched GPU inference |
| `esm2_service/esm2_client.py` | HTTP / in-process client for ESM2 service |
| `inference_service.py` | Abstract async queue-based service |
| `inference_client.py` | Abstract client protocol |
| `orchestrator.py` | `start_services()` / `start_services_local()` |
| `server.py` | aiohttp HTTP server |
| `dragon_launcher.py` | Dragon HPC backend launcher |
| `utils.py` | Queue helpers, batch sizing |
| `kill_all.py` | Process cleanup |

### Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `logger.py` | Colored structured logging, metrics recording |
| `workflow.py` | Config loading, GPU enumeration, Dragon policies |

### Workflows

- **`workflows/esm2_inference/`**: Standalone ESM2 inference workflow
  - `run_esm2_infern.py`: entry point (local or server mode)
  - `inference.yaml`: config (model path, GPU count, output dir)
  - `delta_gpu_sbatch.sh`: SLURM script for Delta HPC

- **`workflows/sgdes/`**: SGDES protein engineering workflow
  - `sgdes_workflow.py`: main workflow class
  - `run_workflow.py`: entry point
  - `config.yaml`: campaign configuration

- **`workflows/plot_telemetry.sh`**: plots asyncflow JSONL telemetry to PNG

---

## YAML Config Features

### Environment Variable Expansion

All config files support `${VAR}` and `$VAR` references, expanded at load time by `_expand_env()`:

```yaml
service_python: "${VE_HOME}/esm2/bin/python"
outdir: "${SPHERICAL_DIR}/workflows/sgdes/output"
model_path: "${HF_HOME}/models/esm2_t33_650M_UR50D"
```

Unset variables raise `KeyError` at load time.

### Inference Config Example

```yaml
mode: local          # "local" (in-process) or "server" (HTTP)
num_services: 2      # one per GPU
model_name: esm2_t33_650M_UR50D
batch_size: 32
output_dir: outputs/embeddings
metrics_dir: outputs/metrics
stub_sleep_s: 0.1    # debug: bypass real inference, sleep instead
```

---

## Testing

### Test Organization

| File | What it tests |
|------|--------------|
| `tests/test_inference_service.py` | Multi-GPU service orchestration |
| `tests/test_server.py` | aiohttp server endpoints |
| `tests/test_client.py` | HTTP client interface |
| `tests/test_logger.py` | Structured logging utilities |
| `tests/test_utils.py` | Config loading and GPU helpers |
| `tests/test_sgdes_workflow.py` | SGDES protein engineering workflow |

### Test Markers

```bash
pytest -m "not slow"       # skip slow integration tests
pytest -m integration      # only integration tests
pytest -m gpu              # only GPU tests (if CUDA available)
```

### Async Tests

All async tests use `anyio`:

```python
import pytest
pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

async def test_something():
    ...
```

---

## Deployment

### Local Mode (no GPU required)

```bash
python workflows/esm2_inference/run_esm2_infern.py \
  --config workflows/esm2_inference/inference.yaml \
  --mode local
```

### Delta HPC (SLURM + Dragon)

Set `SBATCH_ACCOUNT` and `HF_TOKEN` before submitting:

```bash
export SBATCH_ACCOUNT=your-project-id
export HF_TOKEN=hf_...
sbatch workflows/esm2_inference/delta_gpu_sbatch.sh
```

The batch script reads `$SBATCH_ACCOUNT` for the `#SBATCH -A` directive.

### Telemetry Visualization

```bash
bash workflows/plot_telemetry.sh \
  workflows/sgdes/telemetry_output/out.jsonl \
  --out-dir plots/sgdes
```
