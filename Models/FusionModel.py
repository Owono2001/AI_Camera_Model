import torch
import torch.nn as nn
from Models.VisionTransformer import VisionTransformer


class FusionModel(nn.Module):
    """
    FusionModel for combining RGB and NIR inputs.
    """
    def __init__(self, args):
        super(FusionModel, self).__init__()

        # Define Vision Transformer for RGB input
        self.rgb_model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_c=3,
            num_classes=args.num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Define Vision Transformer for NIR input (single channel)
        self.nir_model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_c=1,  # NIR has only one channel
            num_classes=args.num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Fusion layer: concatenating RGB and NIR embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Classification head
        self.classifier = nn.Linear(768, args.num_classes)

    def forward(self, rgb_input, nir_input):
        # Extract features from RGB and NIR models
        rgb_features = self.rgb_model.forward_features(rgb_input)
        nir_features = self.nir_model.forward_features(nir_input)

        # Concatenate and pass through fusion layer
        fused_features = torch.cat((rgb_features, nir_features), dim=1)
        fused_features = self.fusion_layer(fused_features)

        # Classify fused features
        output = self.classifier(fused_features)
        return output
