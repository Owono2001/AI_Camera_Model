import os
import cv2
import torch
import shutil
import numpy as np
from torch import nn
from Models.RGBModel import RGBModel
from Models.NIRModel import NIRModel
from Models.FusionModel import FusionModel
from torchvision import transforms

# ✅ Set Random Seed for Reproducibility
def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.
    Args:
        seed (int): Random seed value.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ✅ Create Model Based on Arguments
def create_model(args):
    """
    Create the model based on the configuration in args.
    Args:
        args: Command-line arguments or configuration object.

    Returns:
        model (nn.Module): Initialized model.
    """
    if args.model == "RGBModel":
        model = RGBModel(args)
    elif args.model == "NIRModel":
        model = NIRModel(args)
    elif args.model == "FusionModel":
        model = FusionModel(args)
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    return model


# ✅ Enable Multi-GPU Support
def model_parallel(args, model):
    """
    Parallelize the model across multiple GPUs if available.
    Args:
        args: Configuration object with GPU settings.
        model (nn.Module): The model to parallelize.

    Returns:
        model (nn.DataParallel): Model parallelized across available GPUs.
    """
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)
    return model


# ✅ Remove and Recreate Directory
def remove_dir_and_create_dir(dir_name):
    """
    Remove an existing directory and create a new one.
    Args:
        dir_name (str): Directory path.

    Returns:
        None
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    print(f"{dir_name} created successfully.")


# ✅ Load NIR Data Matching RGB Filenames
def load_nir_data(nir_dir, file_names):
    """
    Load NIR data corresponding to RGB file names.
    Args:
        nir_dir (str): Directory containing NIR images.
        file_names (list): List of RGB file names to match.

    Returns:
        nir_data (list): List of NIR images as numpy arrays.
    """
    nir_data = []
    for file_name in file_names:
        nir_path = os.path.join(nir_dir, file_name)
        if os.path.exists(nir_path):
            nir_image = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            nir_data.append(nir_image)
        else:
            raise FileNotFoundError(f"NIR image not found for {file_name}")
    return nir_data


# ✅ Preprocess a Single NIR Image
def preprocess_nir_image(image):
    """
    Preprocess a single NIR image for model input.
    Args:
        image (numpy array): Input NIR image.

    Returns:
        torch.Tensor: Preprocessed NIR image tensor.
    """
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return image


# ✅ Save Model Weights
def save_model_weights(model, path):
    """
    Save model weights to a specified path.
    Args:
        model (nn.Module): Trained model.
        path (str): File path to save the weights.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}.")


# ✅ Load Model Weights
def load_model_weights(model, path, device):
    """
    Load model weights from a specified path.
    Args:
        model (nn.Module): The model to load weights into.
        path (str): Path to the weights file.
        device (torch.device): Device to map the weights.

    Returns:
        model (nn.Module): Model with loaded weights.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model weights loaded from {path}.")
    else:
        raise FileNotFoundError(f"Weights file not found at {path}.")
    return model


# ✅ Calculate Class Weights for Imbalanced Datasets
def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights to handle imbalanced datasets.
    Args:
        dataset: Dataset object with labels.
        num_classes (int): Number of classes.

    Returns:
        list: List of class weights.
    """
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    return class_weights.tolist()


# ✅ Preprocess RGB Frame for Inference
def preprocess_rgb_frame(frame):
    """
    Preprocess a single RGB frame for model inference.
    Args:
        frame: Input frame from the camera.

    Returns:
        torch.Tensor: Preprocessed RGB image tensor.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(frame)


# ✅ Preprocess NIR Frame for Inference
def preprocess_nir_frame(frame):
    """
    Preprocess a single NIR frame for model inference.
    Args:
        frame: Input frame from the NIR camera.

    Returns:
        torch.Tensor: Preprocessed NIR image tensor.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(frame)


# ✅ Fusion of RGB & NIR Predictions
def fuse_predictions(rgb_pred, nir_pred, weight_rgb=0.5, weight_nir=0.5):
    """
    Fuse predictions from RGB and NIR models.
    Args:
        rgb_pred (torch.Tensor): Softmax predictions from the RGB model.
        nir_pred (torch.Tensor): Softmax predictions from the NIR model.
        weight_rgb (float): Weight for RGB predictions.
        weight_nir (float): Weight for NIR predictions.

    Returns:
        torch.Tensor: Fused prediction vector.
    """
    return (weight_rgb * rgb_pred) + (weight_nir * nir_pred)
