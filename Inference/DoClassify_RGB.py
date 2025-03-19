import sys
import os
import torch
import cv2
from PIL import Image
from datetime import datetime
from torchvision import transforms

# Add project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import from config and other modules
from config import args
from DataLoader.RGBLoader import get_rgb_transform
from Models.RGBModel import RGBModel

# ---------- Class Explanations ----------
class_explanations = {
    "empty_bunch": "The image shows an empty palm oil bunch with no visible fruits.",
    "ripe": "Vibrant orange-red color indicates optimal ripeness for harvesting.",
    "unripe": "Predominant green color signifies immature fruits."
}

# ---------- Helper Functions ----------
def find_available_camera(max_check=3):
    """Check for available cameras up to a specified index."""
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"📷 Found camera at index {i}")
            return cap
        cap.release()
    raise RuntimeError("🚨 No available cameras detected.")

def preprocess_frame(frame, transform):
    """Convert and transform an OpenCV frame for model input."""
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise ValueError("Error processing frame image: " + str(e))
    return transform(image)

# ---------- Main Function ----------
def main():
    try:
        # Setup device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Using device: {device}")
    except Exception as e:
        print("Error detecting GPU: ", e)
        return

    try:
        # Initialize camera
        camera = find_available_camera()
    except RuntimeError as e:
        print(e)
        return

    # Load the trained model
    model_path = os.path.join(project_root, "Summary", "FusionModel", "RGB_Weights", "best_rgb_model.pth")
    if not os.path.exists(model_path):
        print(f"🔴 Model weights not found at {model_path}. Exiting...")
        return

    try:
        model = RGBModel(args).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully.")
    except RuntimeError as e:
        print("🔴 GPU memory issue during model loading: ", e)
        return
    except Exception as e:
        print("🔴 Failed to load model: ", e)
        return

    # Prepare the directory to save classified images
    save_dir = os.path.join(project_root, "Captured_Images", "RGB")
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print("Error creating save directory: ", e)
        return

    # Start live classification
    print("\n🎥 Starting live classification (Press 'Q' to quit)...")
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("⚠️ Failed to capture frame.")
                continue

            try:
                # Preprocess frame and perform prediction
                with torch.no_grad():
                    transform = get_rgb_transform(train=False)
                    input_tensor = preprocess_frame(frame, transform).unsqueeze(0).to(device)
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs).item()
                    label = args.label_name[pred_class]
                    confidence = probs[0][pred_class].item()
                    explanation = class_explanations.get(label, "No explanation available.")
            except RuntimeError as e:
                print("🔴 GPU memory error during inference: ", e)
                continue  # Skip this frame and try the next
            except Exception as e:
                print("Error during prediction: ", e)
                continue

            # Display prediction and explanation on the video feed
            display_text = f"{label} ({confidence:.2%})"
            cv2.putText(frame, display_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, explanation, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Palm Oil Fruit Classifier", frame)

            # Save frame if confidence is above updated threshold
            if confidence > 0.7:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"{timestamp}_{label}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Image saved: {save_path}")
                except Exception as e:
                    print("Error saving image: ", e)

            # Quit if 'Q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        camera.release()
        cv2.destroyAllWindows()
        print("\n🛑 Resources released. Exiting...")

# ---------- Entry Point ----------
if __name__ == "__main__":
    main()
