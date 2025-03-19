import os
import cv2
import torch
from PIL import Image
from DataLoader.utils import get_rgb_transform, get_nir_transform
from Models.RGBModel import RGBModel
from Models.NIRModel import NIRModel
from config import args

def preprocess_frame(frame, transform):
    """Preprocess a single frame for inference."""
    image = Image.fromarray(frame)
    return transform(image).unsqueeze(0)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize cameras
    rgb_camera = cv2.VideoCapture(args.rgb_device)
    nir_camera = cv2.VideoCapture(args.nir_device)

    if not rgb_camera.isOpened() or not nir_camera.isOpened():
        print("❌ Error: Unable to access cameras.")
        return

    # Load models
    rgb_model = RGBModel(args).to(device)
    nir_model = NIRModel(args).to(device)

    rgb_model.eval()
    nir_model.eval()

    # Load model weights with error handling
    try:
        rgb_model.load_state_dict(torch.load("Summary/RGBModel/weights/best_model.pth", map_location=device))
        nir_model.load_state_dict(torch.load("Summary/NIRModel/weights/best_model.pth", map_location=device))
        print("✅ Fusion models loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error loading model weights: {e}")
        return

    print("🚀 Starting live classification with fusion...")

    # Transformation functions
    rgb_transform = get_rgb_transform(train=False)
    nir_transform = get_nir_transform(train=False)

    while True:
        # Read and preprocess RGB frame
        ret_rgb, frame_rgb = rgb_camera.read()
        if not ret_rgb:
            print("❌ Error: Unable to read from RGB camera.")
            break
        image_rgb = preprocess_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB), rgb_transform).to(device)

        # Read and preprocess NIR frame
        ret_nir, frame_nir = nir_camera.read()
        if not ret_nir:
            print("❌ Error: Unable to read from NIR camera.")
            break
        frame_nir = cv2.cvtColor(frame_nir, cv2.COLOR_BGR2GRAY)
        image_nir = preprocess_frame(frame_nir, nir_transform).to(device)

        # Perform inference
        with torch.no_grad():
            rgb_pred = torch.softmax(rgb_model(image_rgb), dim=1)
            nir_pred = torch.softmax(nir_model(image_nir), dim=1)

            # Fuse predictions (Weighted Average)
            fusion_weight_rgb = 0.6  # Can be adjusted via args
            fusion_weight_nir = 0.4  # Can be adjusted via args
            fused_pred = (fusion_weight_rgb * rgb_pred + fusion_weight_nir * nir_pred)
            predicted_class = torch.argmax(fused_pred).item()
            predicted_label = args.label_name[predicted_class]
            confidence = fused_pred[0][predicted_class].item() * 100  # Confidence percentage

        # Display results with confidence
        cv2.putText(frame_rgb, f"Fusion: {predicted_label} ({confidence:.2f}%)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Fusion Classification", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    rgb_camera.release()
    nir_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
