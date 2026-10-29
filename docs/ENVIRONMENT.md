# Environment inspection and setup log

Recorded 2026-10-28. This project is isolated at
`/workspace/cross_domain_prompt_adaptation_v2`; the earlier project at
`/workspace/qwen_cross_domain_prompt_adaptation` was not modified, overwritten,
or moved (verified read-only inspection only; its git branch/remote untouched).

## Hardware

- GPU: 1x NVIDIA B200, presented as a MIG slice `1g.45gb` (`NVIDIA_VISIBLE_DEVICES`
  pins the container to MIG device `MIG-61af099f-6174-5671-a471-31119348b4d6`).
  Reported by `torch.cuda.get_device_name(0)`: `NVIDIA B200 MIG 1g.45gb`.
  Compute capability `(10, 0)` (Blackwell / sm_100).
- **This pod is shared.** At inspection time, `nvidia-smi` showed another
  process (PID 488814, an unrelated `.venv/bin/python`) already holding
  ~12.8GB on this same MIG slice. `torch.cuda.mem_get_info(0)` measured
  **~33.7GB free / 47.5GB total** at that moment. Budgets for this project are
  chosen against *measured free memory*, not total slice size, and are
  re-checked before any large run since the other process's usage can change.
  No GPU processes belonging to this project interfere with that other PID.
- Driver 570.124.06, CUDA runtime 12.8 (`nvcc` 12.8.93), `CUDA_VERSION=12.8.1.012`.
- Host: 224 CPUs, 2.0TB RAM (907GB available at inspection time), swap disabled.
- Disk: 278GB free of 1.8TB on `/` (overlay fs), shared across all pod projects.

## Software

- OS: Ubuntu 24.04.1 LTS, kernel 6.8.0-57-generic.
- System Python: 3.12.3 (`/usr/bin/python3`), base NVIDIA PyTorch container image
  (`NVIDIA_PYTORCH_VERSION=25.03`).
- System-level `torch==2.7.0a0+7c8ec84dab.nv25.3` is an NVIDIA-internal build
  (not reinstallable from public PyPI) with `TORCH_CUDA_ARCH_LIST` including
  `10.0`, i.e. explicit Blackwell/B200 support. **Verified** with a real CUDA
  op before any model loading: a 2048x2048 bf16 matmul on `cuda:0` completed
  and `torch.cuda.mem_get_info` returned sane free/total values.

## Project virtual environment

- Created at `.venv` with `python3 -m venv --system-site-packages .venv` so it
  reuses the system NVIDIA-built `torch`/`torchvision`/CUDA stack (cannot and
  should not be reinstalled) while keeping all NLP dependencies
  (`transformers`, `datasets`, `accelerate`, ...) isolated from both the
  system Python and from the old project's separate `.venv`.
- Installing modern `datasets` required overriding the container-wide
  `PIP_CONSTRAINT=/etc/pip/constraint.txt` (pins `pyarrow==19.0.1` for an
  unrelated system RAPIDS/cuDF install) for this venv's install commands only
  (`PIP_CONSTRAINT= pip install ...`). This installs `pyarrow==24.0.0` and
  `huggingface_hub==1.33.0` into `.venv`'s own site-packages; pip explicitly
  refused to touch/uninstall the system-level `pyarrow==19.0.1` (different
  environment), and a direct check after install confirmed
  `/usr/bin/python3 -c "import pyarrow"` still reports `19.0.1`. See
  `requirements.txt` for the exact pinned, resolved, working version set and
  install command.
- `pip check` inside `.venv` reports only: `cudf`/`pylibcudf` (system RAPIDS
  packages, visible via `--system-site-packages`, never imported by this
  project) want `pyarrow<20`. This is an expected, harmless side effect of the
  override above, not a broken install for anything this project uses.
- Verified inside `.venv`: `torch` (CUDA available, bf16 matmul succeeds, same
  free/total memory readout as above), `transformers==5.3.0`,
  `datasets==5.1.0`, `accelerate==1.15.0`, `scikit-learn==1.6.1`,
  `pandas==2.2.3`, `numpy==1.26.4`.

## Existing caches available for reuse (reuse permitted; predictions/caches are not)

`~/.cache/huggingface/hub` already contains (42GB total):
- `models--Qwen--Qwen2.5-0.5B-Instruct`, `models--Qwen--Qwen2.5-1.5B-Instruct`
  (the two candidate screening checkpoints from the brief), plus several
  unrelated cached models from other projects on this pod
  (`Qwen2.5-Coder-*`, `Qwen3-*`, `DeepSeek-R1-Distill-Qwen-1.5B`).
- `datasets--stanfordnlp--imdb`, `datasets--yelp_polarity`,
  `datasets--takala--financial_phrasebank`, `datasets--google-research-datasets--mbpp`.
- **Not yet cached**: SST (any revision) and Yelp Review Full -- both will be
  downloaded fresh as part of Section 5 dataset verification, and their
  resolved revisions/hashes recorded in `data/DATASET_README.md`.

## Reproduction

```bash
cd /workspace/cross_domain_prompt_adaptation_v2
python3 -m venv --system-site-packages .venv
PIP_CONSTRAINT= .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```
