# Troubleshooting Guide

## ✅ UPDATE: AST Library is Now Fixed!

**Good news!** Version 1.0.1 of `adaptive-sparse-training` is now available and working correctly.

### To use the fixed version:

```python
# Upgrade to the latest working version
!pip install --upgrade adaptive-sparse-training

# Then train with AST normally
!python train_ast.py --config configs/config_colab.yaml
```

**What's fixed:**
- ✅ Warmup epochs now work correctly
- ✅ Loss tensor handling is fixed
- ✅ 60-90% energy savings as expected
- ✅ All features fully functional

---

## Common Issues and Solutions

### 1. AST Training Error: "IndexError: Dimension specified as 0 but tensor has no dimensions"

**Error Message:**
```
IndexError: Dimension specified as 0 but tensor has no dimensions
File "/usr/local/lib/python3.12/dist-packages/adaptive_sparse_training/sundew.py", line 83
batch_size = losses.size(0)
```

**Cause:**
This was a bug in older versions of the `adaptive-sparse-training` library (< 1.0.1).

**SOLUTION: Upgrade to Latest Version**

```python
# Upgrade to version 1.0.1 or later
!pip install --upgrade adaptive-sparse-training

# Verify you have the fixed version
!pip show adaptive-sparse-training

# Then train normally
!python train_ast.py --config configs/config_colab.yaml
```

**Alternative: Use Regular Training**

If you prefer training without AST (still gets 95-97% accuracy):

```python
!cp configs/config_ast.yaml configs/config.yaml
!python train.py --config configs/config.yaml
```

### 2. AST Error: "Loss tensor has no dimensions (scalar)"

**Error Message:**
```
ValueError: Loss tensor has no dimensions (scalar). Please ensure your criterion uses reduction='none' for per-sample losses.
```

**Cause:**
AST needs per-sample losses, but the criterion was using `reduction='mean'` (default).

**SOLUTION:**
This is fixed in the latest code. If you cloned before this fix, pull the latest:

```python
!git pull origin main
```

Or manually fix `train_ast.py` line 139:
```python
# Change from:
criterion = nn.CrossEntropyLoss()

# To:
criterion = nn.CrossEntropyLoss(reduction='none')
```

### 3. Kaggle API Error in Colab

**Error:** `AttributeError: 'NoneType' object has no attribute 'kernel'`

**Solution:**
Always upload your `kaggle.json` in a notebook cell BEFORE running `colab_setup.py`:

```python
from google.colab import files
uploaded = files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 3. Visualization Files Not Found

**Error:** `FileNotFoundError: No such file or directory: 'visualizations/ast_results.png'`

**Cause:** Trying to display visualizations before training completes.

**Solution:**
1. Complete training first: `!python train_ast.py --config configs/config_colab.yaml`
2. Then generate visualizations: `!python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl`

### 4. Out of Memory (OOM) Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
Reduce batch size in your config:
- T4 GPU: Use `batch_size: 32-64`
- P100 GPU: Use `batch_size: 64-128`
- V100/A100 GPU: Use `batch_size: 128-256`

### 5. Google Drive Mount Failed

**Error:** `'NoneType' object has no attribute 'kernel'` when mounting Drive

**Cause:** Running as a script instead of in notebook cells.

**Solution:**
Mount Drive manually in a notebook cell before running scripts:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Performance Tips

### Faster Training
- Enable AMP (Automatic Mixed Precision): `amp: true` in config
- Use larger batch sizes on powerful GPUs
- Reduce image size if needed: `image_size: 224` → `image_size: 128`

### Better Accuracy
- Increase epochs: `epochs: 30` → `epochs: 50`
- Use warmup epochs (if AST bug is fixed): `ast_warmup_epochs: 2-5`
- Reduce learning rate: `learning_rate: 0.0003` → `learning_rate: 0.0001`

### More Energy Savings
- Lower activation rate: `ast_target_activation_rate: 0.40` → `0.30` (70% savings)
- Note: May reduce accuracy by 1-2%

## Still Having Issues?

1. Check the [GitHub Issues](https://github.com/oluwafemidiakhoa/Malaria/issues)
2. Review the error logs carefully
3. Try the fallback to regular training (`train.py`)
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

## Known Limitations

- AST library is experimental and may have bugs
- Warmup epochs are currently disabled due to library issues
- Google Colab session timeout after ~12 hours
- Free Colab GPU availability may be limited
