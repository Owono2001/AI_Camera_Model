import sys
import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Dynamically add project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

# Import from your project modules
from Models.RGBModel import RGBModel
from config import args

# Set device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = os.path.join(project_root, "Summary", "FusionModel", "RGB_Weights", "best_rgb_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model weights not found at {model_path}.")

model = RGBModel(args)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Class information and explanations
class_info = {
    "empty_bunch": ("Empty Bunch", "🌴❌"),
    "ripe": ("Ripe", "🌴✔️"),
    "unripe": ("Unripe", "🌴🟩")
}

class_explanations = {
    "empty_bunch": "The image shows an empty palm oil bunch with no visible fruits, indicating no harvestable produce.",
    "ripe": "The fruits display a vibrant orange-red hue and uniform color, indicating optimal ripeness for harvesting.",
    "unripe": "Predominant green color and firm texture signify immature fruits not ready for harvest."
}

# Load and preprocess a test image
try:
    image_path = os.path.join(project_root, "Captured_Images", "Test", "test_image.jpg")
    image = Image.open(image_path).convert("RGB")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare image tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidences, indices = torch.topk(probs, len(class_info))

    # Convert to Python values
    confidences = confidences.squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    class_names = list(class_info.keys())

    # Get top class explanation
    top_class_key = class_names[indices[0]]

    # Prepare results
    results = []
    for i in range(len(indices)):
        class_idx = indices[i]
        results.append((
            class_info[class_names[class_idx]][0],
            class_info[class_names[class_idx]][1],
            confidences[i]
        ))

    # Print detailed report
    print("\n" + "=" * 50)
    print(f"🌴 Palm Oil Fruit Ripeness Analysis Report")
    print("=" * 50)
    print(f"📌 Top Prediction: {results[0][1]} {results[0][0]} ({results[0][2]*100:.2f}% confidence)")
    print(f"\n🔍 Explanation: {class_explanations[top_class_key]}\n")

    print("📊 Confidence Breakdown:")
    for name, emoji, confidence in results:
        print(f"  {emoji} {name.ljust(12)}: {confidence*100:.2f}%")

    print("\n" + "-" * 50)
    print("💡 Interpretation Guide:")
    print("- Empty Bunch: No fruit, dry or leftover bunch structure")
    print("- Unripe: Hard texture, deep green color")
    print("- Ripe: Fully orange-red, soft and ready for harvesting")
    print("=" * 50 + "\n")

    # Display the analyzed image
    print("🖼️ Displaying analyzed image...")
    image.show()

except FileNotFoundError:
    print("❌ Error: Image file not found. Please check the file path.")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")