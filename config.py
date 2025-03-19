import argparse
import os

# Initialize argument parser
parser = argparse.ArgumentParser(description="Configuration file for palm oil fruit ripeness classification with RGB and NIR inputs")

# 🔥 Training Parameters
parser.add_argument('--num_classes', type=int, default=3, 
                    help='Number of classes for classification (e.g., empty_bunch, ripe, unripe)')
parser.add_argument('--epochs', type=int, default=100, 
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Batch size for training and validation')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='Initial learning rate')
parser.add_argument('--lrf', type=float, default=0.01, 
                    help='Learning rate factor for learning rate scheduler')

# 🔥 Dataset Paths (UPDATED)
parser.add_argument('--dataset_rgb_dir', type=str, default='Captured_Images/RGB', 
                    help='Path to the RGB dataset directory')
parser.add_argument('--dataset_nir_dir', type=str, default='Captured_Images/NIR', 
                    help='Path to the NIR dataset directory')

# 🔥 Model Selection & Pretrained Weights
parser.add_argument('--model', type=str, default='FusionModel', 
                    choices=['RGBModel', 'NIRModel', 'FusionModel'],
                    help='Choose the model to use (RGBModel, NIRModel, FusionModel)')
parser.add_argument('--weights', type=str, default='Pretrain_Weights/vit_base_patch16_224_in21k.pth', 
                    help='Path to pre-trained model weights (leave empty to disable)')
parser.add_argument('--freeze_layers', action='store_true', 
                    help='Freeze model layers during training (useful for transfer learning)')

# 🔥 Hardware Configurations
parser.add_argument('--gpu', type=str, default='0', 
                    help='Select the GPU to use (e.g., "0" for GPU 0 or "0,1" for multiple GPUs)')
parser.add_argument('--num_workers', type=int, default=4, 
                    help='Number of data loading workers')

# 🔥 Live Video Feed & Real-Time Inference
parser.add_argument('--use_nir', action='store_true', 
                    help='Enable or disable the use of NIR data')
parser.add_argument('--video_feed', action='store_true', 
                    help='Enable or disable live video feed for real-time classification')
parser.add_argument('--rgb_device', type=int, default=0, 
                    help='Device ID for the RGB camera (0 for primary camera)')
parser.add_argument('--nir_device', type=int, default=1, 
                    help='Device ID for the NIR sensor (e.g., 1 for secondary camera)')

# 🔥 Output & Logging
parser.add_argument('--summary_dir', type=str, default='Summary/FusionModel', 
                    help='Directory for saving model weights, logs, and TensorBoard summaries')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='Interval for logging training updates')

# 🔥 Class Labels
parser.add_argument('--label_name', type=list, default=[
    "empty_bunch", 
    "ripe",  
    "unripe"
], help='List of class names for classification')

# 🔥 Parse the arguments
args = parser.parse_args()

# 🔥 Set the GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 🔥 Optional: Print the Configuration for Debugging
if __name__ == "__main__":
    print("\n🔥 CONFIGURATION SETTINGS 🔥")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg.upper()}: {value}")
    print("=" * 50)
    