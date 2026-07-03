# 🔍 Digital Image Forgery Detection using Deep Learning

> **Semester Project for SPU** - An intelligent system for detecting digital image forgery using state-of-the-art deep learning techniques.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Methodologies](#methodologies)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Branches](#branches)
- [Dataset](#dataset)
- [References](#references)

---

## 📝 Overview

Digital image forgery detection is a critical challenge in the era of AI-generated and manipulated media. This project explores **three distinct deep learning methodologies** to detect forged or tampered images.

Each methodology leverages different neural network architectures and input combinations:
- **RGB Images** - Original color information
- **Error Level Analysis (ELA)** - Highlights compression artifacts
- **Spatial Richness Model (SRM)** - Captures subtle noise patterns

---

## 🧠 Methodologies

### **Methodology 1: Dual-Input ResNet50V2** ⭐ (Best Performer)
**Branch:** `main`  
**Architecture:** Dual-stream neural network merging RGB and ELA inputs

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.88% |
| **AUC** | 0.9468 |
| **Recall** | 0.8937 |

✅ **Best balance between accuracy and computational efficiency**

📖 [View Full Documentation](https://github.com/Mariam6600/Digital-image-forgery-detection/blob/main/README.md) | 
🔗 [Colab Notebook](https://colab.research.google.com/drive/1Ih6SUtPCtkl-Ix89bDkqD1srwx5ZbiXg?usp=sharing)

---

### **Methodology 2: Single-Input ResNet101V2**
**Branch:** `experiment2`  
**Architecture:** Single-stream network using only ELA images

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.00% |
| **AUC** | 0.88 |
| **Recall** | 0.85 |

💡 **Lightweight alternative with strong performance**

📖 [View Full Documentation](https://github.com/Mariam6600/Digital-image-forgery-detection/blob/experiment2/README.md) |
🔗 [Colab Notebook](https://colab.research.google.com/drive/1xaS9suZ1JK-IgTXcNYZ3NkUKMWNhHWdA?usp=sharing)

---

### **Methodology 3: Triple-Input EfficientNetV2S**
**Branch:** `experiment3`  
**Architecture:** Triple-stream network using RGB, ELA, and SRM inputs

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.00% |
| **AUC** | 0.93385 |
| **Recall** | 0.95 |

🎯 **Highest recall - optimal for high-sensitivity forgery detection**

📖 [View Full Documentation](https://github.com/Mariam6600/Digital-image-forgery-detection/blob/experiment3/README.md) |
🔗 [Colab Notebook](https://colab.research.google.com/drive/1SWIVMyHJyat0zBGcvfAdcBRXzku4bkX2?usp=sharing)

---

## 📊 Key Results Comparison

```
┌─────────────────────────┬──────────┬────────┬────────┐
│ Methodology             │ Accuracy │  AUC   │ Recall │
├─────────────────────────┼──────────┼────────┼────────┤
│ Dual-Input ResNet50V2   │  87.88%  │ 0.9468 │ 0.8937 │
│ Single-Input ResNet101V2│  87.00%  │ 0.88   │ 0.85   │
│ Triple-Input EfficientV2│  87.00%  │ 0.9338 │ 0.95   │
└─────────────────────────┴──────────┴────────┴────────┘
```

---

## 📝 ⚠️ ملاحظة مهمة

**الأكواد الحالية متاحة على [Google Colab](https://colab.research.google.com/) كـ Jupyter Notebooks**

جاري العمل على تحويل وإضافة نسخة Python قابلة للاستخدام محلياً في المستودع. ستتضمن:
- ✅ ملفات Python منفصلة قابلة للاستخدام
- ✅ سهولة التكامل مع المشاريع الأخرى
- ✅ دعم التثبيت عبر pip
- ✅ أدوات سطر الأوامر للتدريب والتنبؤ

---

## 📁 Project Structure

```
Digital-image-forgery-detection/
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Project setup
├── .gitignore               # Git ignore rules
├── notebooks/
│   ├── methodology1_dual_input.ipynb
│   ├── methodology2_single_input.ipynb
│   └── methodology3_triple_input.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Image preprocessing & ELA generation
│   ├── model_architecture.py     # Neural network models
│   ├── training.py              # Training loop
│   ├── evaluation.py            # Metrics computation
│   └── utils.py                 # Helper functions
├── data/
│   ├── CASIA2/               # Dataset (not included in repo)
│   ├── train/
│   ├── test/
│   └── val/
└── models/
    └── pretrained/           # Saved models
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- GPU (optional but recommended for faster training)

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Mariam6600/Digital-image-forgery-detection.git
cd Digital-image-forgery-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download CASIA 2.0 dataset:**
   - Download from [CASIA Forensics Detection](http://forensics.idealtest.org/index_12.html)
   - Extract to `data/CASIA2/` directory

---

## ⚡ Quick Start

### Running on Google Colab (Recommended)
Each methodology has a dedicated Colab notebook with all code pre-configured:

| Methodology | Colab Link |
|------------|-----------|
| Dual-Input (Methodology 1) | [🔗 Open Colab](https://colab.research.google.com/drive/1Ih6SUtPCtkl-Ix89bDkqD1srwx5ZbiXg?usp=sharing) |
| Single-Input (Methodology 2) | [🔗 Open Colab](https://colab.research.google.com/drive/1xaS9suZ1JK-IgTXcNYZ3NkUKMWNhHWdA?usp=sharing) |
| Triple-Input (Methodology 3) | [🔗 Open Colab](https://colab.research.google.com/drive/1SWIVMyHJyat0zBGcvfAdcBRXzku4bkX2?usp=sharing) |

### Running Locally

```python
from src.model_architecture import DualInputResNet50V2
from src.training import train_model
from src.data_preprocessing import load_dataset

# Load data
train_loader, val_loader, test_loader = load_dataset('data/CASIA2/')

# Initialize model
model = DualInputResNet50V2()

# Train
train_model(model, train_loader, val_loader, epochs=50)
```

---

## 🌿 Branches

| Branch | Methodology | Architecture | Best For |
|--------|-------------|--------------|----------|
| `main` | Methodology 1 | Dual-Input ResNet50V2 | 🌟 Production use |
| `experiment2` | Methodology 2 | Single-Input ResNet101V2 | 💡 Lightweight deployment |
| `experiment3` | Methodology 3 | Triple-Input EfficientNetV2S | 🎯 High-recall scenarios |

**Switch branches:**
```bash
git checkout experiment2    # or experiment3
```

---

## 📦 Dataset

**CASIA 2.0 (Chinese Academy of Sciences):**
- **Total Images:** 21,000+
- **Authentic:** 7,200 images
- **Forged:** 14,400 images
- **Resolution:** 256×256 to 800×800 pixels

📥 [Download CASIA 2.0](http://forensics.idealtest.org/index_12.html)

---

## 🔧 Key Features

✅ **Three distinct methodologies** - Compare different approaches  
✅ **Pre-trained models** - Ready to use for inference  
✅ **Comprehensive evaluation metrics** - Accuracy, Precision, Recall, AUC, F1  
✅ **ELA preprocessing** - Error Level Analysis image generation  
✅ **Google Colab ready** - No local GPU required  
✅ **Visualization tools** - Plot results and confusion matrices  

---

## 📚 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. [arXiv](https://arxiv.org/abs/1512.03385)
2. Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. [arXiv](https://arxiv.org/abs/2104.14294)
3. Fridrich, J., Kodovský, J. (2012). Rich Models for Steganalysis of Digital Images. [IEEE Transactions](https://ieeexplore.ieee.org/)
4. Zhong, J., & Huang, Y. (2007). CASIA2: A Large-Scale Heterogeneous Image Forgery Database. [Forensics DB](http://forensics.idealtest.org/)

---

## 👤 Author

**Mariam Abd Al Aal**  
Student at SPU | Deep Learning Researcher  
📧 [abdmariam900@gmail.com](mailto:abdmariam900@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Future Improvements:**
- [ ] Add Vision Transformer (ViT) methodology
- [ ] Implement model quantization for edge deployment
- [ ] Create REST API for inference
- [ ] Add real-time video forgery detection
- [ ] Implement attention mechanism visualization

---

## ⭐ If you find this project helpful, please consider giving it a star!

**Last Updated:** August 10, 2025  
**Status:** ✅ Active Development
