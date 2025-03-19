import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NIRDataset(Dataset):
    """Dataset class for loading NIR (grayscale) images and their labels."""
    
    def __init__(self, nir_dir, train=True):
        self.nir_dir = nir_dir
        self.transform = self.get_transform(train)
        
        self.label_mapping = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(nir_dir)))}
        self.nir_images, self.labels = [], []

        for class_name, class_idx in self.label_mapping.items():
            class_folder = os.path.join(nir_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for file_name in os.listdir(class_folder):
                if file_name.endswith(('.jpg', '.png')):
                    self.nir_images.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_idx)

        if len(self.nir_images) == 0:
            raise ValueError(f"No NIR images found in directory: {self.nir_dir}")

    def get_transform(self, train):
        if train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, shear=0.1),
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __getitem__(self, index):
        nir_path = self.nir_images[index]
        label = self.labels[index]
        nir_image = Image.open(nir_path).convert("L")  # Force grayscale
        nir_image = self.transform(nir_image)
        return nir_image, label

    def __len__(self):
        return len(self.nir_images)

def get_nir_dataloader(nir_dir, batch_size, num_workers, train=True, pin_memory=False):
    dataset = NIRDataset(nir_dir, train)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory
    )