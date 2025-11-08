# Quick Start Guide for Google Colab

## ⚡ Fast Track: Get Training in 5 Minutes

### Step 1: Clone Repository
```python
!git clone https://github.com/oluwafemidiakhoa/Malaria.git
%cd Malaria
```

### Step 2: Upload Kaggle API Key
```python
from google.colab import files
print("📁 Upload your kaggle.json:")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("✅ Kaggle configured!")
```

### Step 3: Run Automated Setup
```python
!python colab_setup.py
```

### Step 4: Train with AST (Energy-Efficient Training) ✨

**✅ AST Library is Now Fixed (v1.0.1+)**

```python
# Ensure you have the latest version
!pip install --upgrade adaptive-sparse-training

# Train with AST (60% energy savings!)
!python train_ast.py --config configs/config_colab.yaml
```

**Training Time:** 20-30 minutes on T4 GPU

**Expected Results:**
- Accuracy: 95-97%
- Energy Savings: 60%
- Model saved to: `checkpoints_ast/best.pt`

**Alternative: Regular Training (No AST)**

If you prefer standard training without energy optimization:

```python
!cp configs/config_ast.yaml configs/config.yaml
!python train.py --config configs/config.yaml
```

### Step 5: Generate Visualizations

```python
# Create visualizations directory
!mkdir -p visualizations

# Generate plots
!python visualize_ast.py --metrics checkpoints/metrics.jsonl --output-dir visualizations

# Display results
from IPython.display import Image, display
from pathlib import Path

if Path('visualizations/ast_results.png').exists():
    display(Image('visualizations/ast_results.png'))
```

### Step 6: Evaluate Model

```python
!python eval.py --weights checkpoints/best.pt

# View confusion matrix
display(Image('checkpoints/cm.png'))
```

## AST vs Regular Training

**AST (Adaptive Sparse Training):**
- ✅ 60% energy savings during training
- ✅ Same 95-97% accuracy
- ✅ Great for showcasing efficient AI
- ✅ Fixed in version 1.0.1+

**Regular Training:**
- ✅ Same 95-97% accuracy
- ✅ Simpler, no dependencies on AST library
- ✅ Slightly faster (no selection overhead)

## Need Help?

Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Full Notebooks

For complete step-by-step guides with explanations:
- [Malaria_AST_Training.ipynb](Malaria_AST_Training.ipynb) - Complete training workflow
- [Malaria_Train_and_Deploy.ipynb](Malaria_Train_and_Deploy.ipynb) - Training + Gradio deployment

Remember to use `train.py` instead of `train_ast.py` until the AST library is fixed!
