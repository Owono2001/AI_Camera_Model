# Train_NIR.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import args
from DataLoader.NIRLoader import get_nir_dataloader
from Models.NIRModel import NIRModel
from utils import remove_dir_and_create_dir, set_seed, calculate_class_weights

def train_nir(args):
    """Train the NIR Vision Transformer model with enhanced spectral processing"""
    
    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Directory setup
    weights_dir = os.path.join(args.summary_dir, "NIR_Weights")
    log_dir = os.path.join(args.summary_dir, "NIR_Logs")
    remove_dir_and_create_dir(weights_dir)
    remove_dir_and_create_dir(log_dir)
    
    # Initialize components
    writer = SummaryWriter(log_dir=log_dir)
    set_seed(777)
    
    try:
        train_loader = get_nir_dataloader("Captured_Images/NIR", args.batch_size, 4, train=True)
        val_loader = get_nir_dataloader("Captured_Images/NIR", args.batch_size, 4, train=False)
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    # Model initialization
    model = NIRModel(args).to(device)
    
    # Enhanced spectral-aware loss function
    class_weights = calculate_class_weights(train_loader.dataset, args.num_classes)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device),
        label_smoothing=0.2  # Higher smoothing for spectral data
    )
    
    # Spectral-optimized optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.1,
        betas=(0.9, 0.999))
    
    # Spectral learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = [], 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Spectral gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss.append(loss.item())
            train_acc += (outputs.argmax(1) == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_acc = [], 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss.append(loss_fn(outputs, labels).item())
                val_acc += (outputs.argmax(1) == labels).sum().item()
        
        # Metrics calculation
        train_acc /= len(train_loader.dataset)
        val_acc /= len(val_loader.dataset)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{weights_dir}/best_nir_model.pth")
        
        # Update scheduler
        scheduler.step()
        
        # TensorBoard logging
        writer.add_scalars("Loss", {
            "train": np.mean(train_loss),
            "val": np.mean(val_loss)
        }, epoch)
        writer.add_scalars("Accuracy", {
            "train": train_acc,
            "val": val_acc
        }, epoch)
    
    writer.close()
    print(f"Best NIR Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_nir(args)