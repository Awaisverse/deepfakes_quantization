# DeepFake Model Quantization (CPU-first)

This repo contains a Kaggle notebook (`notebookfff027f37c.ipynb`) that loads your DeepFake classifier checkpoint, generates multiple optimized model artifacts, and benchmarks them for **CPU inference**.

## What the notebook produces

Typical outputs in `/kaggle/working`:

- `model_original_fp32.pth` — baseline PyTorch FP32 weights.
- `model_dynamic_int8.pth` — dynamic quantized PyTorch model (easy PyTorch CPU deployment).
- `model_static_ptq_int8.pth` — static PTQ output (eager PTQ when supported, FX fallback when eager kernels fail).
- `model_fx_graph_int8.pth` — FX graph quantized model.
- `model_torchscript_int8.pt` — TorchScript optimized model.
- `model_fp16.pth` — half-precision model (mainly size benefit; CPU speedup is workload-dependent).
- `model_fp32.onnx` and `model_onnx_int8.onnx` — ONNX export + ORT dynamic INT8.
- `quantization_summary.csv` — benchmark table (latency/size/speedup).
- `artifact_manifest.json` — run metadata + produced files list.

## Which file should I use?

Use this priority for CPU users:

1. **`model_onnx_int8.onnx`** (best default for CPU serving if ONNX Runtime is available).
2. **`model_dynamic_int8.pth`** (best simple PyTorch-only deployment).
3. **`model_torchscript_int8.pt`** (PyTorch runtime with TorchScript graph optimizations).
4. **`model_static_ptq_int8.pth`** (works when eager static PTQ path is compatible; notebook includes FX fallback).
5. **`model_original_fp32.pth`** as compatibility fallback.

Always pick based on your measured latency + accuracy from `quantization_summary.csv`.

## Quick inference examples

### ONNX Runtime (recommended for CPU)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model_onnx_int8.onnx", providers=["CPUExecutionProvider"])
# image_np shape: [1, 3, 224, 224], dtype float32, normalized same as training
logits = sess.run(None, {"input": image_np})[0]
pred = int(np.argmax(logits, axis=1)[0])
```

### PyTorch Dynamic INT8

```python
import torch

model = torch.load("model_dynamic_int8.pth", map_location="cpu")
model.eval()
with torch.no_grad():
    logits = model(image_tensor.unsqueeze(0))
    pred = int(logits.argmax(dim=1).item())
```

## Practical notes

- Use the **same preprocessing** as training (resize, tensor conversion, normalization).
- Validate on a real validation set before shipping.
- CPU quantization support depends on operator availability in your PyTorch build.
- The notebook now auto-selects quantization backend and includes fallback handling for unsupported eager PTQ residual ops.

## Suggested deployment decision flow

1. Run notebook once on your target-like CPU environment.
2. Open `quantization_summary.csv`.
3. Select candidate with lowest latency that preserves acceptable accuracy.
4. Deploy that artifact format (ONNX Runtime or PyTorch runtime) consistently in production.
