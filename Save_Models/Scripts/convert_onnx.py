import sys
import os
import torch

# ✅ Add project root directory to sys.path to fix import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ✅ Now import after adding sys.path
from config import args
from Models.RGBModel import RGBModel  

# ✅ Correct model path
model_path = "Summary/FusionModel/RGB_Weights/best_rgb_model.pth"
onnx_path = "Summary/FusionModel/RGB_Weights/rgb_model.onnx"  # ✅ Save ONNX model in the same folder

# ✅ Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Trained model not found at {model_path}. Please train & save the model first.")

# ✅ Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RGBModel(args).to(device)

# ✅ Fix FutureWarning for safe model loading
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# ✅ Convert to ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model, dummy_input, onnx_path, export_params=True, opset_version=11, do_constant_folding=True,
    input_names=['input'], output_names=['output']
)

print(f"✅ Model successfully converted to ONNX: {onnx_path}")
