# Models/NIRModel.py
from Models.VisionTransformer import VisionTransformer

def NIRModel(args):
    """Create a Vision Transformer optimized for NIR (grayscale) images"""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_c=1,  # Single channel for NIR
        num_classes=args.num_classes,
        embed_dim=384,  # Smaller embedding dimension for NIR
        depth=6,
        num_heads=6,
        mlp_ratio=3.0  # Adjusted ratio for NIR
    )