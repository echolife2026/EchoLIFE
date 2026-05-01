# EchoLIFE project

This repository starts from precomputed `DeltaE [T, D]` npz files and runs the complete downstream pipeline:

1. **Segmentation / feature extraction**: `DeltaE -> Z -> S/centroid/spread -> room segments -> segment_summaries.json`
2. **Activity recognition**: `segment_summaries.json -> activity labels`

## Expected data layout

```text
dataset/p1/
├── 1/
│   ├── deltae.npz
│   └── label.json
├── 2/
│   ├── deltae.npz
│   └── label.json
├── layout.json                    # optional
└── profile.txt                    # optional
```

Each `deltae.npz` should contain room keys such as `kitchen`, `living`, `bedroom`, `bathroom`, `study`, `dining`. Each array must be `float16`-compatible with shape `[T, D]`.

## Install

```bash
pip install -r requirements.txt
```

For LLM modes, set your API key in the environment instead of hardcoding it:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_API_BASE="api_base"
export ECHOLIFE_LLM_MODEL="llm_model"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_API_BASE="api_base"
$env:ECHOLIFE_LLM_MODEL="llm_model"
```

## Option A: run the complete downstream pipeline at once

```bash
python scripts/run_full_pipeline.py \
  --root-parent ../dataset/p1 \
  --deltae-name deltae.npz \
  --label-name label.json \
  --modes rulebook_only \
  --plots
```

This produces segmentation outputs under each day folder:

```text
1/echolife_segmentation_out/segment_summaries.json
1/echolife_segmentation_out/day_meta.json
1/echolife_segmentation_out/segments.json
```

and recognition outputs under:

```text
echolife_recognition_out/rulebook_only/
```

## Option B: run segmentation/features and recognition separately

First run segmentation/features:

```bash
python scripts/run_segmentation.py \
  --root-parent ../dataset/p1 \
  --deltae-name deltae.npz \
  --label-name label.json \
  --plots
```

Then run recognition from the saved `segment_summaries.json` files:

```bash
python scripts/run_recognition.py \
  --root-parent ../dataset/p1 \
  --segment-output-dir echolife_segmentation_out \
  --modes rulebook_only
```

## Recognition modes

Supported modes:

- `rulebook_only`: deterministic rule-based recognition; no API key required.
- `llm_only`: LLM directly classifies each segment; requires `OPENAI_API_KEY`.
- `rulebook_then_llm_calibration`: rulebook first, then LLM calibrates low-margin decisions; requires `OPENAI_API_KEY`.

Example with all modes:

```bash
python scripts/run_full_pipeline.py \
  --root-parent ../dataset/p1 \
  --modes rulebook_only llm_only rulebook_then_llm_calibration \
  --rulebook-strategy fixed
```

Example with LLM-generated rulebook once:

```bash
python scripts/run_recognition.py \
  --root-parent ../dataset/p1 \
  --modes rulebook_only rulebook_then_llm_calibration \
  --rulebook-strategy llm_once
```

## Evaluation

If every day folder has `label.json`, add `--eval`:

```bash
python scripts/run_recognition.py \
  --root-parent ../dataset/output/p1 \
  --modes rulebook_only \
  --eval
```

Evaluation outputs are saved under:

```text
echolife_recognition_out/evaluation/
```

## PyCharm parameters

### Full downstream pipeline

**Script path**:

```text
.../echolife_project/scripts/run_full_pipeline.py
```

**Working directory**:

```text
.../echolife_project
```

**Parameters**:

```bash
--root-parent D:\your_project\dataset\p1 --deltae-name deltae.npz --label-name label.json --modes rulebook_only --plots
```

### Segmentation/features only

```bash
--root-parent D:\your_project\dataset\p1 --deltae-name deltae.npz --label-name label.json --plots
```

### Recognition only

```bash
--root-parent D:\your_project\dataset\p1 --segment-output-dir echolife_segmentation_out --modes rulebook_only
```
