from torchvision import transforms

# 🔥 Define transformations for RGB images
def get_rgb_transform(train=True):
    """
    Get the transformation pipeline for RGB images.

    Args:
        train (bool): If True, returns transformations for training; else for validation.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline for RGB images.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB images
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB images
        ])


# 🔥 Define transformations for NIR images (grayscale)
def get_nir_transform(train=True):
    """
    Get the transformation pipeline for NIR images.

    Args:
        train (bool): If True, returns transformations for training; else for validation.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline for NIR images.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize NIR images (grayscale)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize NIR images (grayscale)
        ])
