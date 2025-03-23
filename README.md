# Palm Oil Fruit Ripeness Classifier 🌴📸

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A vision-based AI model for classifying palm oil fruit ripeness using RGB images. Achieves **99.96% validation accuracy** with Vision Transformer architecture.

<div align="center">
  <img src="https://komarev.com/ghpvc/?username=Owono2001&style=flat-square&color=7DF9FF" alt="Profile Views">
  <p style="font-family: 'Space Mono', monospace; color: #7DF9FF; font-size: 1.2em;">Your visit sparks innovation! 🔥</p>
</div>

🍇 Class Samples
## Palm Oil Fruit Classification
#### Empty Bunch
![Empty Bunch](./Images/empty_bunch.jpg)

#### Ripe Fruit
![Ripe](./Images/ripes.jpg)

#### Unripe Fruit
![Unripe](./Images/unripe.jpg)

*Sample classifications: Ripe (🟠), Unripe (🟢), Empty (⚫)*

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Raspberry Pi Setup](#-raspberry-pi-setup)
- [File Structure](#-file-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🌟 Features
- 🧠 High-accuracy Vision Transformer model (99.96% val accuracy)
- 🕒 Real-time classification via webcam
- 📱 Raspberry Pi compatible
- 📊 Three-class classification:
  - 🟠 Ripe Bunch
  - 🟢 Unripe Bunch
  - ⚫ Empty Bunch
- 🔄 ONNX model conversion support

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Webcam-enabled device
- PyTorch 2.0.0

### Setup
```bash
# Clone repository
git clone https://github.com/Owono2001/AI_Model_Palm_Oil_Fruit.git
cd AI_Model_Palm_Oil_Fruit

# Create and activate virtual environment
python -m venv venv

# Activation
# Windows
.\venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

🚀 Usage
Real-time Inference
y
python Inference/DoClassify_RGB.py
Point the camera at palm fruit bunches

Press q to exit

Training

python Training/Train_RGB.py
Model Conversion (ONNX)

python Save_Models/scripts/convert_onnx.py
📐 Model Architecture

ViT(
  (transformer): Sequential(
    (0): TransformerEncoderLayer(...)
    (1-11): 11x TransformerEncoderLayer(...)
  )
  (classifier): Linear(in_features=768, out_features=3, bias=True)
)
Vision Transformer with 12 encoder layers

� Raspberry Pi Setup

# Install ARM-compatible PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
sudo apt install v4l-utils python3-pyopencl
pip3 install onnxruntime opencv-python-headless

🤝 Contributing
We welcome contributions! Please follow this workflow:

Fork the repository

Create your feature branch:

git checkout -b feature/your-feature

Commit changes:

git commit -m 'Add awesome feature'

Push to branch:

git push origin feature/your-feature
Open a pull request

📜 License
MIT License - see LICENSE for details

🙏 Acknowledgments
Vision Transformer architecture

PyTorch framework

OpenCV camera integration
