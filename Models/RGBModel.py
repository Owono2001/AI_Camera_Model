import torch
from Models.VisionTransformer import VisionTransformer  # ✅ Import VisionTransformer

def RGBModel(args):
    """Create an optimized RGB Vision Transformer Model."""
    return VisionTransformer(
        image_size=224, patch_size=16, in_c=3, num_classes=args.num_classes, embed_dim=768, depth=12, num_heads=12
    )
