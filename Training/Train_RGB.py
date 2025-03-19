import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import args
from DataLoader.RGBLoader import get_rgb_dataloader
from Models.RGBModel import RGBModel
from utils import remove_dir_and_create_dir, set_seed, calculate_class_weights

def train_rgb(args):
    """ Train the RGB model using GPU acceleration and mixed precision with enhanced error handling """

    # 🔥 Force GPU Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance
    torch.backends.cudnn.deterministic = True  # Ensure stable training results

    # 🔥 Adjust Batch Size
    args.batch_size = 8  # Set batch size to 8 (reduce if OOM occurs)

    # 🔥 Create directories for saving model weights & logs
    weights_dir = os.path.join(args.summary_dir, "RGB_Weights")
    log_dir = os.path.join(args.summary_dir, "RGB_Logs")
    remove_dir_and_create_dir(weights_dir)
    remove_dir_and_create_dir(log_dir)

    # 🔥 Initialize TensorBoard logging
    writer = SummaryWriter(log_dir=log_dir)

    # 🔥 Set random seed for reproducibility
    set_seed(777)
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f"✅ Using {num_workers} dataloader workers.")

    try:
        # 🔥 Load RGB dataset
        train_loader = get_rgb_dataloader("Captured_Images/RGB", args.batch_size, num_workers, train=True, pin_memory=True)
        val_loader = get_rgb_dataloader("Captured_Images/RGB", args.batch_size, num_workers, train=False, pin_memory=True)
    except Exception as e:
        print("Error loading data: ", e)
        return

    train_num = len(train_loader.dataset)
    val_num = len(val_loader.dataset)
    print(f"✅ Training on {train_num} images, Validating on {val_num} images.")

    # 🔥 Create RGB Model and Move to GPU
    try:
        model = RGBModel(args).to(device)
    except Exception as e:
        print("Error creating or moving model to device: ", e)
        return

    # 🔥 Handle class imbalance by calculating class weights
    try:
        class_weights = calculate_class_weights(train_loader.dataset, args.num_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    except Exception as e:
        print("Error calculating class weights: ", e)
        return

    # 🔥 Define loss function with class weights and label smoothing (enhanced)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.1  # Reduces overconfidence
    )

    # 🔥 Enhanced optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-5, 
        weight_decay=0.05,  # Increased weight decay
        betas=(0.9, 0.999)
    )

    # 🔥 Use linear warmup + cosine schedule
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        [
            lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
        ],
        milestones=[5]
    )

    # 🔥 Enable Mixed Precision Training
    scaler = torch.amp.GradScaler()

    # 🔥 Gradient Accumulation for Large Batches
    accumulation_steps = 4  # Accumulate gradients over 4 steps

    best_acc = 0.0  # Track the best validation accuracy

    # 🔥 Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = [], 0
        train_bar = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, labels) in enumerate(train_bar):
            try:
                images, labels = images.to(device), labels.to(device)
            except Exception as e:
                print("Error moving data to device: ", e)
                continue

            optimizer.zero_grad()
            try:
                # 🔥 Use Mixed Precision Training
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)

                    # 🔥 Debug: Check for NaN in Model Output
                    if torch.isnan(outputs).any():
                        print("🔥 NaN detected in model outputs. Fixing with small epsilon...")
                        outputs = torch.nan_to_num(outputs, nan=0.0)  # Replace NaN values

                    loss = loss_fn(outputs, labels) / accumulation_steps  # Scale loss for accumulation
            except RuntimeError as e:
                print("GPU memory error during forward pass: ", e)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print("Error during forward pass: ", e)
                continue

            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print("GPU memory error during backward pass: ", e)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print("Error during backward pass: ", e)
                continue

            # 🔥 Gradient Clipping to Prevent Exploding Gradients
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except Exception as e:
                print("Error during gradient clipping: ", e)

            # 🔥 Apply optimizer step only after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                except RuntimeError as e:
                    print("GPU memory error during optimizer step: ", e)
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print("Error during optimizer step: ", e)
                    continue

            train_loss.append(loss.item() * accumulation_steps)  # Reverse scaling
            try:
                train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            except Exception as e:
                print("Error calculating training accuracy: ", e)

            # 🔥 Reduce logging overhead (log every 10 batches)
            if (batch_idx + 1) % 10 == 0:
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 🔥 Validation loop
        model.eval()
        val_loss, val_acc = [], 0
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # 🔥 Debug: Check for NaN in Model Output
                    if torch.isnan(outputs).any():
                        print("🔥 NaN detected in validation outputs. Fixing with small epsilon...")
                        outputs = torch.nan_to_num(outputs, nan=0.0)  # Replace NaN values

                    loss = loss_fn(outputs, labels)
                except RuntimeError as e:
                    print("GPU memory error during validation: ", e)
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print("Error during validation forward pass: ", e)
                    continue

                val_loss.append(loss.item())
                try:
                    val_acc += (outputs.argmax(dim=1) == labels).sum().item()
                except Exception as e:
                    print("Error calculating validation accuracy: ", e)

        # 🔥 Calculate epoch-level metrics
        try:
            train_acc /= train_num
            val_acc /= val_num
        except Exception as e:
            print("Error calculating epoch metrics: ", e)
            continue

        print(f"✅ Epoch {epoch+1}: Train Loss: {np.mean(train_loss):.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {np.mean(val_loss):.4f}, Val Acc: {val_acc:.4f}")

        # 🔥 Log metrics to TensorBoard
        writer.add_scalar("RGB/Train Loss", np.mean(train_loss), epoch)
        writer.add_scalar("RGB/Train Accuracy", train_acc, epoch)
        writer.add_scalar("RGB/Validation Loss", np.mean(val_loss), epoch)
        writer.add_scalar("RGB/Validation Accuracy", val_acc, epoch)
        writer.add_scalar("RGB/Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # 🔥 Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            try:
                torch.save(model.state_dict(), f"{weights_dir}/best_rgb_model.pth")
            except Exception as e:
                print("Error saving model weights: ", e)

        # 🔥 Step the scheduler
        try:
            scheduler.step()
        except Exception as e:
            print("Error stepping the scheduler: ", e)

    writer.close()
    print("✅ Training complete. Best Validation Accuracy: {:.4f}".format(best_acc))


if __name__ == '__main__':
    train_rgb(args)
