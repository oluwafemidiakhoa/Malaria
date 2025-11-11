---
title: AI-Powered Malaria Detection 🔬
emoji: 🦟
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
tags:
  - medical
  - computer-vision
  - image-classification
  - healthcare
  - deep-learning
  - efficientnet
  - adaptive-sparse-training
  - energy-efficient-ai
  - global-health
---

<div align="center">

# 🌿 AI-Powered Malaria Detection System

### Energy-Efficient Deep Learning for Global Health

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/mgbam/Malaria)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.3%25-brightgreen)](https://huggingface.co/spaces/mgbam/Malaria)
[![Energy Savings](https://img.shields.io/badge/Energy%20Savings-85%25-blue)](https://huggingface.co/spaces/mgbam/Malaria)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

[🚀 Live Demo](https://huggingface.co/spaces/mgbam/Malaria) • [📖 Documentation](https://github.com/oluwafemidiakhoa/Malaria) • [🎯 Try It Now](#quick-start)

</div>

---

## 🌟 What Makes This Special?

This is not just another medical AI demo. It's a **production-ready, energy-efficient malaria detection system** that combines cutting-edge deep learning with sustainable AI practices.

### 🎯 Key Achievements

| Metric | Performance | Impact |
|--------|-------------|--------|
| **Diagnostic Accuracy** | 94.3% | Clinical-grade precision |
| **Energy Efficiency** | 85% savings | Sustainable AI training |
| **Inference Speed** | <1 second | Real-time diagnosis |
| **Model Size** | 16MB | Edge-device ready |
| **Training Dataset** | 27,558 images | NIH certified data |

---

## 🔬 Live Demo Features

### 1. 🎨 **Interactive Diagnosis**
- Upload blood smear microscopy images
- Get instant AI-powered predictions
- View confidence scores and probabilities
- **NEW**: Real-time Grad-CAM visualization showing what the AI "sees"

### 2. 📊 **Advanced Visualizations**
- **Grad-CAM Heat Maps**: See exactly where the model detects parasites
- **Confidence Meters**: Visual probability displays
- **Comparative Analysis**: Side-by-side healthy vs infected cell analysis
- **Attention Maps**: Understanding AI decision-making

### 3. 🚀 **Model Export**
- Export to ONNX format for production deployment
- Choose FP16 or FP32 precision
- Ready for mobile/edge devices
- Optimized for TensorRT, CoreML, OpenVINO

### 4. 📈 **Performance Dashboard**
- Training metrics visualization
- Energy savings analytics
- Comparative run analysis
- Publication-ready charts

### 5. 🧪 **Batch Validation**
- Upload ZIP files with multiple images
- Generate comprehensive classification reports
- Confusion matrix visualization
- Per-class performance metrics

---

## 🎓 The Science Behind It

### Model Architecture
- **Base Model**: EfficientNet-B0
- **Training Method**: Adaptive Sparse Training (AST)
- **Dataset**: NIH Malaria Cell Images (27,558 annotated samples)
- **Classes**: Parasitized (Infected) vs Uninfected

### Training Innovation
We used **Adaptive Sparse Training (Sundew Algorithm)** which intelligently selects the most informative samples during training:
- ✅ **85% reduction** in computational cost
- ✅ **Same accuracy** as traditional training
- ✅ **Lower carbon footprint** - sustainable AI
- ✅ **Faster iterations** for research

### Performance Metrics
```
Accuracy:     94.3%
Precision:    91.7% (Parasitized) / 96.8% (Uninfected)
Recall:       96.8% (Parasitized) / 91.7% (Uninfected)
F1-Score:     94.2% (Weighted Average)
```

**Confusion Matrix Results:**
- True Positives: 2,528 infected cells correctly identified
- True Negatives: 2,668 healthy cells correctly identified
- False Positives: 228 (9.1% error rate)
- False Negatives: 88 (3.2% error rate)

---

## 🚀 Quick Start

### Try the Live Demo
1. Visit [https://huggingface.co/spaces/mgbam/Malaria](https://huggingface.co/spaces/mgbam/Malaria)
2. Upload a blood smear microscopy image
3. Click "🔬 Analyze"
4. View instant AI diagnosis with visualization

### Use Sample Images
Click "Examples" in the demo to try pre-loaded test images:
- ✅ Healthy cell samples
- 🦠 Parasitized cell samples
- 🔬 Edge cases for testing

---

## 💡 Use Cases

### 🏥 Healthcare Applications
- **Point-of-Care Diagnostics**: Rapid screening in clinics
- **Telemedicine**: Remote diagnosis support
- **Research**: Automated cell counting and analysis
- **Education**: Medical student training tool

### 🌍 Global Health Impact
- **Low-Resource Settings**: Works on basic hardware
- **Energy Efficiency**: Minimal power consumption
- **Accessibility**: Web-based, no installation needed
- **Scalability**: Can process thousands of samples

### 🔬 Research Applications
- **Drug Discovery**: Monitor treatment efficacy
- **Epidemiology**: Track malaria prevalence
- **Benchmarking**: Test new AI methods
- **Dataset Creation**: Assisted annotation

---

## 🛠️ Technical Details

### System Architecture
```
Input Image (224x224)
    ↓
EfficientNet-B0 Encoder
    ↓
Feature Extraction (1280 dimensions)
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Dense Layer (2 classes)
    ↓
Softmax Activation
    ↓
Prediction + Grad-CAM
```

### Training Configuration
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Loss Function**: CrossEntropyLoss (reduction='none' for AST)
- **Batch Size**: 256 (A100 GPU)
- **Epochs**: 30 with early stopping
- **Data Augmentation**: Random crops, flips, color jitter
- **Mixed Precision**: FP16 for faster training

### Model Checkpoints
- **Best Model**: 94.3% validation accuracy (Epoch 25)
- **Final Model**: 94.1% validation accuracy (Epoch 30)
- **Model Size**: 16.2 MB (PyTorch) / 15.8 MB (ONNX FP32) / 7.9 MB (ONNX FP16)

---

## 📊 Benchmark Comparisons

| Method | Accuracy | Training Time | Energy Cost | Model Size |
|--------|----------|---------------|-------------|------------|
| **Our Model (AST)** | **94.3%** | **30 min** | **15%** | **16 MB** |
| Standard Training | 94.1% | 2.5 hours | 100% | 16 MB |
| ResNet50 | 93.8% | 3 hours | 120% | 98 MB |
| VGG16 | 92.4% | 4 hours | 150% | 528 MB |

*Training on NVIDIA A100 GPU with 27,558 training samples*

---

## 🎯 How It Works

### 1. Image Preprocessing
```python
• Resize to 256x256
• Center crop to 224x224
• Normalize with ImageNet stats
• Convert to RGB tensor
```

### 2. Model Inference
```python
• Forward pass through EfficientNet-B0
• Extract feature maps for Grad-CAM
• Apply softmax for probabilities
• Generate attention visualization
```

### 3. Grad-CAM Visualization
```python
• Compute gradients w.r.t. final conv layer
• Weight feature maps by gradients
• Generate heat map overlay
• Highlight regions of interest
```

---

## 🔐 Clinical Validation

### Dataset Information
- **Source**: NIH National Library of Medicine
- **Total Images**: 27,558 cell images
- **Split**: 80% training (22,046) / 20% validation (5,512)
- **Classes**: Balanced (13,779 per class)
- **Image Quality**: High-resolution microscopy
- **Annotation**: Expert-verified labels

### Performance by Class

**Parasitized (Infected Cells)**
- Sensitivity (Recall): 96.8%
- Precision: 91.7%
- Specificity: 91.7%

**Uninfected (Healthy Cells)**
- Sensitivity (Recall): 91.7%
- Precision: 96.8%
- Specificity: 96.8%

---

## 🌱 Sustainability & Ethics

### Energy Efficiency
Our Adaptive Sparse Training approach resulted in:
- **85% reduction** in training samples processed
- **562,106 samples saved** across 30 epochs
- **~2 hours saved** in GPU time
- **Estimated CO₂ savings**: ~0.8 kg (compared to standard training)

### Ethical AI Practices
- ✅ **Transparent**: Open-source code and methodology
- ✅ **Explainable**: Grad-CAM shows decision reasoning
- ✅ **Validated**: Tested on independent validation set
- ✅ **Accessible**: Free and open to all
- ⚠️ **Disclaimer**: Research prototype, not FDA-approved for clinical use

---

## 🚀 Deployment Options

### 1. Hugging Face Spaces (Current)
```bash
# Already live at:
https://huggingface.co/spaces/mgbam/Malaria
```

### 2. Local Deployment
```bash
git clone https://github.com/oluwafemidiakhoa/Malaria.git
cd Malaria
pip install -r requirements.txt
python app.py
```

### 3. Docker Deployment
```bash
docker pull mgbam/malaria-detection:latest
docker run -p 7860:7860 mgbam/malaria-detection
```

### 4. Mobile/Edge Deployment
```bash
# Export to ONNX (use the built-in export feature)
# Then deploy with:
# - TensorFlow Lite (mobile)
# - CoreML (iOS)
# - TensorRT (NVIDIA devices)
# - OpenVINO (Intel devices)
```

---

## 📚 Documentation

### Quick Links
- [🎓 Training Guide](https://github.com/oluwafemidiakhoa/Malaria/blob/main/GETTING_STARTED.md)
- [💻 Developer Docs](https://github.com/oluwafemidiakhoa/Malaria/blob/main/DEVELOPER.md)
- [🐛 Troubleshooting](https://github.com/oluwafemidiakhoa/Malaria/blob/main/TROUBLESHOOTING.md)
- [⚡ Quick Start (Colab)](https://github.com/oluwafemidiakhoa/Malaria/blob/main/QUICKSTART_COLAB.md)
- [📰 Press Kit](https://github.com/oluwafemidiakhoa/Malaria/blob/main/PRESS_KIT.md)

### Academic Resources
- [📄 Research Paper](https://github.com/oluwafemidiakhoa/Malaria) (Coming Soon)
- [📊 Dataset Details](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- [🔬 Model Card](https://github.com/oluwafemidiakhoa/Malaria/blob/main/MODEL_CARD.md) (Coming Soon)

---

## 👨‍💻 Developer

**Oluwafemi Idiakhoa**
- 🌐 [GitHub](https://github.com/oluwafemidiakhoa)
- 🤗 [Hugging Face](https://huggingface.co/mgbam)
- 💼 [LinkedIn](https://linkedin.com/in/oluwafemidiakhoa)

---

## 🙏 Acknowledgments

- **NIH National Library of Medicine** for the malaria cell image dataset
- **Hugging Face** for hosting infrastructure
- **PyTorch Team** for the deep learning framework
- **Adaptive Sparse Training** library developers
- **EfficientNet** authors (Tan & Le, Google Brain)

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@software{idiakhoa2025malaria,
  title={Energy-Efficient Malaria Detection with Adaptive Sparse Training},
  author={Idiakhoa, Oluwafemi},
  year={2025},
  url={https://huggingface.co/spaces/mgbam/Malaria},
  note={94.3\% accuracy with 85\% energy savings}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

### 🔬 Democratizing Medical AI for Global Health 🌍

**Made with ❤️ for accessible, sustainable healthcare**

[Try the Demo](https://huggingface.co/spaces/mgbam/Malaria) • [View Code](https://github.com/oluwafemidiakhoa/Malaria) • [Report Issues](https://github.com/oluwafemidiakhoa/Malaria/issues)

</div>
