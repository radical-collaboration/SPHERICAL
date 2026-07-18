# SPHERICAL

HPC workflow orchestration framework for multi-GPU protein inference and engineering campaigns.

## Features

- **Multi-GPU Inference** — worker pool per GPU with automatic load balancing; aiohttp HTTP server/client
- **ESM2 Inference Workflow** — standalone or campaign-embedded ESM2-650M embedding service
- **SGDES Workflow** — Structure-Guided Deep Evolution Solver for iterative protein sequence optimisation
- **Dragon/Asyncflow Integration** — HPC runtime for distributed multi-node execution via DragonHPC
- **Automatic Device Detection** — CUDA GPUs if available, CPU fallback
- **YAML Config with Env-Var Expansion** — `${VAR}` references in config files are resolved at load time
- **Telemetry & Visualization** — asyncflow native JSONL telemetry with workflow dashboard plots

---

## Repository Layout

```
spherical/
├── src/
│   ├── inference/                   # InferenceService base, orchestrator, server
│   │   ├── esm2_service/            # ESM2InferenceService + ESM2Client
│   │   ├── inference_client.py
│   │   ├── orchestrator.py
│   │   ├── server.py
│   │   └── utils.py                 # load_config, ensure_dir, export_metrics
│   └── utils/
│       ├── logger.py                # structured Logger with colours
│       └── workflow.py              # _expand_env, load_config, find_gpus, make_policies
├── workflows/
│   ├── plot_telemetry.sh            # Plot asyncflow JSONL telemetry → workflow dashboard PNG
│   ├── esm2_inference/              # Standalone ESM2 inference runner
│   │   ├── run_esm2_infern.py
│   │   └── config.yaml
│   └── sgdes/                       # SGDES protein engineering
│       ├── run_workflow.py
│       ├── sgdes_workflow.py
│       └── config.yaml
└── tests/
    ├── test_inference_service.py
    ├── test_client.py
    ├── test_server.py
    ├── test_sgdes_workflow.py
    ├── test_logger.py
    └── test_utils.py
```

---

## Installation

```bash
# Core
pip install -e .

# With ESM2 model support (torch + transformers)
pip install -e ".[esm2]"

# With Dragon/RADICAL HPC support
pip install -e ".[dragon]"

# Full (includes dev and plotting extras)
pip install -e ".[esm2,dragon,dev]"
```

---

## Quick Start

### ESM2 Inference (standalone)

```bash
# Local mode — model runs in-process, no HTTP server
python workflows/esm2_inference/run_esm2_infern.py \
    --config_file workflows/esm2_inference/config.yaml \
    --mode local

# Server mode (default) — model hosted via HTTP; use dragon for multi-node
dragon -m workflows/esm2_inference/run_esm2_infern.py \
    --config_file workflows/esm2_inference/config.yaml
```

Key config options (`workflows/esm2_inference/config.yaml`):

```yaml
model_path: "facebook/esm2_t33_650M_UR50D"
num_services: 1
num_workers_per_gpu: 4
server_port: 8000
num_batches: 32
max_batch_tokens: 20000
mode: local           # "local" or "server"
engine: concurrent    # "concurrent" or "dragon"
service_python: "${VE_HOME}/esm2/bin/python"   # resolved at load time
```

### SGDES Protein Engineering

```bash
sbatch workflows/sgdes/delta_gpu_sbatch.sh
```

See [workflows/sgdes/README.md](workflows/sgdes/README.md) for full setup, configuration, and scaling results.

---

## Extending for New Model Types

Subclass `InferenceService` from `src.inference.inference_service`:

```python
from src.inference.inference_service import InferenceService

class MyModelService(InferenceService):
    def _load_models(self):
        for device in self.devices:
            self.models[device] = load_model().to(device)

    def process_batch_sync(self, batch_id: int, device: str):
        results = self.models[device](self.reply_store[batch_id])
        self.reply_store[batch_id] = results
        self.processed_queue.put_nowait(batch_id)

    async def generate_batch(self) -> tuple:
        seq = await self.input_queue.get()
        if seq is None:
            raise StopAsyncIteration
        batch = tokenize(seq)
        return len(batch), batch
```

---

## Environment Variables in YAML Configs

All config files support `${VAR}` and `$VAR` shell-style references.
They are expanded by `_expand_env` (in `src/utils/workflow.py`) at load time,
so paths like the following work without any Python-side substitution:

```yaml
outdir:         "${SPHERICAL_DIR}/workflows/sgdes/mayv_output"
service_python: "${VE_HOME}/esm2/bin/python"
```

If a variable is not set at runtime the literal string is preserved, producing a clear `FileNotFoundError` rather than an obscure downstream crash.

---

## Visualization

### Asyncflow workflow telemetry dashboard

`workflows/plot_telemetry.sh` plots the native JSONL telemetry produced by
`asyncflow.start_telemetry()` (enabled via `collect_telemetry: true` in the
workflow config).  It calls
`radical.asyncflow/workflows/telemetry/plot_workflow_dashboard.py` and saves a
PNG to `workflows/plots/<wf_name>/`.

```bash
bash workflows/plot_telemetry.sh <telemetry.jsonl> [--out-dir DIR]
```

The workflow name is inferred from the directory layout
(`<wf_name>/telemetry-output/<file>.jsonl`); pass `--out-dir` to override.
Output file: `workflow_dashboard_<YYYYMMDD_HHMMSS>.png`.

**Example** (SGDES run):
```bash
bash workflows/plot_telemetry.sh \
    workflows/sgdes/telemetry_output/out.jsonl \
    --out-dir plots/sgdes
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Lint and format
ruff check .
ruff format .
```

---

## License

MIT License
