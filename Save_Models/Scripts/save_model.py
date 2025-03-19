import sys
import os
import torch

# 🔥 Fix the ImportError by adding the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Models.RGBModel import RGBModel
from config import args

# ✅ Correct model path
model_path = "Summary/FusionModel/RGB_Weights/best_rgb_model.pth"

# ✅ Ensure the directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

# ✅ Load the trained model
model = RGBModel(args)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))  # 👈 Use `weights_only=True` to prevent security warning
model.eval()

# ✅ Save model in a safer location (optional)
safe_model_path = "Save_Models/best_rgb_model.pth"
os.makedirs(os.path.dirname(safe_model_path), exist_ok=True)
torch.save(model.state_dict(), safe_model_path)

print(f"✅ Model loaded from {model_path} and re-saved at {safe_model_path}")
