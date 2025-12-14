import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import random_split, Subset, Dataset, DataLoader, TensorDataset
from pathlib import Path


from PIL import Image
import glob
from loguru import logger

from matplotlib import pyplot as plt
from collections import Counter

from paths import DATA_DIR, RAW_DATA, SRC_DIR
from data.utils import (
   
    PROCESSED_DATA,
    TRANSFORMS_FOLDER,
    GOOGLENET_TRAIN_IMAGES_NPY,
    GOOGLENET_TRAIN_LABELS_NPY,

    GOOGLENET_TEST_IMAGES_NPY,
    GOOGLENET_TEST_LABELS_NPY,
    
    CLASSES_NAMES,
    CLASSES_INDEX,
    DATA_TRAIN_PATH,
    DATA_TESTA_PATH,

    show_images,
    save_npy,
    load_images_from_folders
)

def show_16augmented_data(dataset, transform, index=0):

    img_path, label = dataset.samples[index]
    original_img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(4, 4, figsize=(14, 6))
    axes = axes.flatten()

    axes[0].imshow(original_img)
    axes[0].set_title("Original", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    for i in range(1, 16):
        aug_tensor = transform(original_img)

        aug_np = aug_tensor.permute(2, 1, 0).numpy()
        aug_np = aug_np * 0.5 + 0.5 
        aug_np = np.clip(aug_np, 0, 1)

        axes[i].imshow(aug_np)
        axes[i].set_title(f"Augmentation {i}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Same Image with Different Augmentations", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("multiple_augmentations.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: multiple_augmentations.png")
    plt.show()


import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class ImageData:

    @classmethod
    def get_eval_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __init__(
        self,
        train_dir,
        test_dir=None,
        batch_size=32,
        num_workers=4,
        balance_classes=True,
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_classes = balance_classes


        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(p=0.55),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.eval_transform = ImageData.get_eval_transform()

       
        self._setup()


    def _setup(self):

        self.train_dataset = datasets.ImageFolder(
            self.train_dir, transform=self.train_transform
        )

        if self.balance_classes:
            class_counts = np.bincount(self.train_dataset.targets)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[t] for t in self.train_dataset.targets]
            sample_weights = torch.DoubleTensor(sample_weights)

            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            self.train_sampler = None

        if self.test_dir:
            self.full_test_dataset = datasets.ImageFolder(
                self.test_dir, transform=self.eval_transform
            )
            val_ratio = 0.2
            val_size = int(len(self.full_test_dataset) * val_ratio)
            test_size = len(self.full_test_dataset) - val_size

            self.val_dataset, self.test_dataset = random_split(
                self.full_test_dataset, [val_size, test_size], 
                generator=torch.Generator().manual_seed(42)
            )


        else:
            self.test_dataset = None

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_loader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )




if __name__ == "__main__":


    data = ImageData(
        train_dir = DATA_TRAIN_PATH,
        test_dir = DATA_TESTA_PATH,
        batch_size=32,
        num_workers=4,
        balance_classes=True
    )

    train_loader = data.train_loader()
    val_loader = data.val_loader()
    test_loader = data.test_loader()

    dataset = data.train_dataset

    def loader_class_distribution(loader):
        counter = Counter()
        for i, (_, labels) in enumerate(loader):
            counter.update(labels.tolist())
            if i == 100:
                break
        return dict(counter)

    print("train class distribution:", loader_class_distribution(train_loader))
    print("Validation class distribution:", loader_class_distribution(val_loader))
    print("Test class distribution:", loader_class_distribution(test_loader))

    show_16augmented_data(dataset, data.train_transform, index=13)

