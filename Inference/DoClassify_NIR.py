# DoClassify_NIR.py
import cv2
import torch
from PIL import Image
from Models.NIRModel import NIRModel
import numpy as np
from torchvision import transforms

class NIRClassifier:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = NIRModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def process_frame(self, frame):
        """Process NIR frame from sensor"""
        image = Image.fromarray(frame).convert("L")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy().flatten()
    
    def classify(self, frame):
        """Classify NIR frame with spectral analysis"""
        probs = self.process_frame(frame)
        class_idx = np.argmax(probs)
        return {
            "class": args.class_names[class_idx],
            "confidence": float(probs[class_idx]),
            "spectral_profile": probs.tolist()
        }
