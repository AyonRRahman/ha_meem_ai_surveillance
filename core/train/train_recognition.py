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


# ──────────────────────────────────────────────
# CONFIG (all tunable parameters in one place)
# ──────────────────────────────────────────────
CONFIG = {
    'data_dir': '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/lfw_aligned',
    'pretrained_ckpt': 'models/face_recognition/adaface_ir50_webface4m.ckpt',
    'batch_size': 64,
    'test_size': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs_warmup': 50,           # head only
    'epochs_full': 300,            # full fine-tune
    'lr_head': 5e-4,
    'lr_backbone': 5e-6,
    'weight_decay': 1e-5,
    'min_samples_per_class': 30,
    'k_neighbors': 7,
    'contrast_lambda': 0.5,
    'embedding_size': 512,
    'adaface_repo_path': 'models/AdaFace',
}

import sys
import os

# Get absolute path of current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two directories: core/train → project_folder
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# Add to sys.path
sys.path.append(project_root)

from core.dataset.FR_Dataset import load_and_filter_data, create_loaders
from core.utils.utils import clear_gpu, load_model, set_seed, setup_optimizer_and_scheduler
from core.recognition.train_recognition_utils import train_full, train_warmup, collect_train_embeddings
from core.eval.eval_recognition import evaluate_faiss
# ──────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────
def main():
    clear_gpu()
    set_seed(42)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(112, scale=(0.8, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    criterion = nn.CrossEntropyLoss()

    # 1. Load & prepare data
    dataset, new_class_names = load_and_filter_data(CONFIG, transform)

    train_loader, test_loader = create_loaders(dataset, CONFIG)

    # 2. Load model
    backbone, head = load_model(CONFIG, num_classes=len(new_class_names))

    # 3. Setup optimizer & scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(backbone, head, CONFIG)

    # 4. Training phase 1: head warm-up
    train_warmup(head, backbone, train_loader, optimizer, criterion, scheduler, CONFIG)

    # 5. Training phase 2: full fine-tune
    train_full(backbone, head, train_loader, optimizer, criterion, scheduler, CONFIG)

    # 6. Collect embeddings & evaluate with FAISS
    train_emb, train_lbl = collect_train_embeddings(backbone, train_loader, CONFIG)
    evaluate_faiss(train_emb, train_lbl, test_loader, backbone, CONFIG)

    print("All done.")

if __name__=='__main__':
    main()