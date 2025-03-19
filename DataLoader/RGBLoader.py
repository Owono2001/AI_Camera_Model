import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class RGBDataset(Dataset):
    """Dataset class for loading RGB images and their labels."""

    def __init__(self, rgb_dir, train=True):
        self.rgb_dir = rgb_dir
        self.transform = self.get_transform(train)

        self.label_mapping = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(rgb_dir)))}
        self.rgb_images, self.labels = [], []

        for class_name, class_idx in self.label_mapping.items():
            class_folder = os.path.join(rgb_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for file_name in os.listdir(class_folder):
                if file_name.endswith(('.jpg', '.png')):
                    self.rgb_images.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_idx)

        if len(self.rgb_images) == 0:
            raise ValueError(f"No RGB images found in directory: {self.rgb_dir}")

    def get_transform(self, train):
        if train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, shear=0.1),
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        rgb_path = self.rgb_images[index]
        label = self.labels[index]
        rgb_image = Image.open(rgb_path).convert("RGB")
        rgb_image = self.transform(rgb_image)

        return rgb_image, label

    def __len__(self):
        return len(self.rgb_images)

def get_rgb_dataloader(rgb_dir, batch_size, num_workers, train=True, pin_memory=False):
    dataset = RGBDataset(rgb_dir, train)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory
    )

def get_rgb_transform(train=True):
    """Returns the transformation used for RGB images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])