import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
from collections import Counter
import gc
import faiss
from typing import Tuple, List, Dict, Any



class NonEmptyImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        valid_classes, class_to_idx = [], {}
        idx = 0
        for cls_name in classes:
            cls_dir = os.path.join(directory, cls_name)
            has_image = any(f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'))
                            for f in os.listdir(cls_dir))
            if has_image:
                valid_classes.append(cls_name)
                class_to_idx[cls_name] = idx
                idx += 1
            else:
                print(f"Skipping empty/invalid folder: {cls_name}")
        if not valid_classes:
            raise FileNotFoundError(f"No valid class folders with images in {directory}")
        return valid_classes, class_to_idx

class CleanDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.bad_files = []

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"[BAD FILE] idx={idx} | path={path} → {str(e)}")
            self.bad_files.append((idx, path, str(e)))
            dummy = torch.zeros(3, 112, 112, dtype=torch.float32)
            return dummy, torch.tensor(0, dtype=torch.long)
def load_and_filter_data(
    config: Dict,
    transform: transforms.Compose
) -> Tuple[Dataset, List[str], Dict[int, str], Dict[str, int]]:

    """
    Load dataset, filter classes with min samples, remap labels to 0..N-1.

    Returns:
        dataset
        new_class_names
        label_to_class
        class_to_label
    """

    print("Loading and filtering dataset...")

    raw_dataset = NonEmptyImageFolder(config['data_dir'], transform=None)

    all_paths = [s[0] for s in raw_dataset.samples]
    original_labels = np.array([s[1] for s in raw_dataset.samples])

    class_counts = Counter(original_labels)

    valid_old_classes = [
        c for c, cnt in class_counts.items()
        if cnt >= config['min_samples_per_class']
    ]

    print(f"Classes with >= {config['min_samples_per_class']} images: {len(valid_old_classes)}")

    valid_mask = np.isin(original_labels, valid_old_classes)

    filtered_paths = [
        p for p, keep in zip(all_paths, valid_mask)
        if keep
    ]

    filtered_old_labels = original_labels[valid_mask]

    sorted_valid_old = sorted(valid_old_classes)

    old_to_new_map = {
        old: new
        for new, old in enumerate(sorted_valid_old)
    }

    remapped_labels = np.array([
        old_to_new_map[lbl]
        for lbl in filtered_old_labels
    ])

    new_class_names = [
        raw_dataset.classes[old]
        for old in sorted_valid_old
    ]

    num_classes = len(new_class_names)

    print(f"Final dataset: {num_classes} classes, {len(remapped_labels)} samples")
    print(f"Label range: min={remapped_labels.min()}, max={remapped_labels.max()}")

    # -----------------------------
    # Create mapping dictionaries
    # -----------------------------
    label_to_class = {
        idx: name
        for idx, name in enumerate(new_class_names)
    }

    class_to_label = {
        name: idx
        for idx, name in enumerate(new_class_names)
    }

    dataset = CleanDataset(
        filtered_paths,
        remapped_labels,
        transform=transform
    )

    return dataset, new_class_names, label_to_class, class_to_label


# def load_and_filter_data(config: Dict, transform: transforms.Compose) -> Tuple[Dataset, List[str]]:
#     """
#     Load dataset, filter classes with min samples, remap labels to 0..N-1.
#     Returns: (dataset, list of new class names)
#     """
#     print("Loading and filtering dataset...")
#     raw_dataset = NonEmptyImageFolder(config['data_dir'], transform=None)

#     all_paths = [s[0] for s in raw_dataset.samples]
#     original_labels = np.array([s[1] for s in raw_dataset.samples])

#     class_counts = Counter(original_labels)
#     valid_old_classes = [c for c, cnt in class_counts.items() if cnt >= config['min_samples_per_class']]

#     print(f"Classes with >= {config['min_samples_per_class']} images: {len(valid_old_classes)}")

#     valid_mask = np.isin(original_labels, valid_old_classes)
#     filtered_paths = [p for p, keep in zip(all_paths, valid_mask) if keep]
#     filtered_old_labels = original_labels[valid_mask]

#     sorted_valid_old = sorted(valid_old_classes)
#     old_to_new_map = {old: new for new, old in enumerate(sorted_valid_old)}
#     remapped_labels = np.array([old_to_new_map[lbl] for lbl in filtered_old_labels])

#     new_class_names = [raw_dataset.classes[old] for old in sorted_valid_old]
#     num_classes = len(new_class_names)

#     print(f"Final dataset: {num_classes} classes, {len(remapped_labels)} samples")
#     print(f"Label range: min={remapped_labels.min()}, max={remapped_labels.max()}")

#     dataset = CleanDataset(filtered_paths, remapped_labels, transform=transform)
    return dataset, new_class_names
# def load_and_filter_data(config: Dict) -> Tuple[Dataset, List[str]]:
#     print("Loading and filtering dataset...")
#     raw_dataset = NonEmptyImageFolder(config['data_dir'], transform=None)

#     all_paths = [s[0] for s in raw_dataset.samples]
#     original_labels = np.array([s[1] for s in raw_dataset.samples])

#     class_counts = Counter(original_labels)
#     valid_old_classes = [c for c, cnt in class_counts.items() if cnt >= config['min_samples_per_class']]

#     print(f"Classes with >= {config['min_samples_per_class']} images: {len(valid_old_classes)}")

#     valid_mask = np.isin(original_labels, valid_old_classes)
#     filtered_paths = [p for p, keep in zip(all_paths, valid_mask) if keep]
#     filtered_old_labels = original_labels[valid_mask]

#     sorted_valid_old = sorted(valid_old_classes)
#     old_to_new_map = {old: new for new, old in enumerate(sorted_valid_old)}
#     remapped_labels = np.array([old_to_new_map[lbl] for lbl in filtered_old_labels])

#     new_class_names = [raw_dataset.classes[old] for old in sorted_valid_old]
#     num_classes = len(new_class_names)

#     print(f"Final dataset: {num_classes} classes, {len(remapped_labels)} samples")
#     print(f"Label range: min={remapped_labels.min()}, max={remapped_labels.max()}")

#     return CleanDataset(filtered_paths, remapped_labels, transform), new_class_names

def create_loaders(dataset: Dataset, config: Dict) -> Tuple[DataLoader, DataLoader]:
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config['test_size'],
        stratify=dataset.labels,
        random_state=42
    )

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds  = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader
