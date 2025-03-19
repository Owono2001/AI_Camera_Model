import sys
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ✅ Add project root to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Models.RGBModel import RGBModel  # ✅ Fix import issue
from DataLoader.RGBLoader import get_rgb_dataloader
from config import args

# ✅ Correct the model path
model_path = "Summary/FusionModel/RGB_Weights/best_rgb_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Trained model not found at {model_path}. Please train & save the model first.")

# ✅ Main execution (necessary for Windows multiprocessing)
if __name__ == "__main__":
    # ✅ Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBModel(args).to(device)

    # ✅ Fix FutureWarning by explicitly setting `weights_only=True`
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ✅ Load validation dataset (set `num_workers=0` for Windows)
    val_loader = get_rgb_dataloader(args.dataset_rgb_dir, batch_size=32, num_workers=0, train=False, pin_memory=True)

    # ✅ Get true labels and predictions
    true_labels, pred_labels = [], []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

    # ✅ Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=args.label_name, yticklabels=args.label_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ✅ Print classification report
    print("🔍 Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=args.label_name))
