import os
from PIL import Image

# Input & Output Directories
input_folder = "dataset/raw_images"
output_folder = "dataset/resized_images"
target_size = (224, 224)  # Change to 256x256 or 512x512 if needed

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process images
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img = img.resize(target_size, Image.ANTIALIAS)  # Resize with high quality
        img.save(os.path.join(output_folder, filename))

print("✅ All images resized successfully!")

#Go through all images in dataset/raw_images.
#Resize them to 224×224 pixels.
#Save them in dataset/resized_images.