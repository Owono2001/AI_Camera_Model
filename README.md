AI_Camera_Model

🌴 Palm Oil Fruit Ripeness Classifier 📸

A vision-based AI model for classifying palm oil fruit ripeness using RGB images.

This model can be deployed on a standard camera or webcam for real-time classification.

🚀 Features

Real-time Classification: Works with a webcam for live predictions.

High Accuracy: Vision Transformer model achieving 99.96% validation accuracy.

Three Classification Categories:

✅ Ripe

❌ Unripe

🍂 Empty Bunch

Easy-to-use inference Script for quick classification.

📋 Requirements

Python 3.8+

Webcam (for real-time inference)

PyTorch 2.0.0+

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/Owono2001/AI_Camera_Model.git
cd AI_Camera_Model

2️⃣ Create & Activate Virtual Environment

Windows

python -m venv venv
.env\Scripts\activate

Linux/Mac

python -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

🎯 Usage

🔍 Real-time Inference

Run the following script to start real-time classification:

python Inference/DoClassify_RGB.py

Instructions:

Point your camera at palm fruit bunches.

Press 'q' to quit.

🎓 Training

To train the model on your dataset:

python Training/Train_RGB.py

🔄 Model Conversion (to ONNX)

python Save_Models/scripts/convert_onnx.py


🍓 Raspberry Pi Setup

To deploy the model on Raspberry Pi, follow these steps:

Install ARM-Compatible PyTorch

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Install Additional Dependencies

sudo apt install v4l-utils python3-pyopencl
pip3 install onnxruntime opencv-python-headless

🤝 Contributing

We welcome contributions! Follow these steps:

Fork the repository

Create a feature branch

Submit a pull request

📜 License

This project is licensed under the MIT License.

🙏 Acknowledgements

Vision Transformer Architecture

PyTorch Framework

OpenCV for Camera Integration
