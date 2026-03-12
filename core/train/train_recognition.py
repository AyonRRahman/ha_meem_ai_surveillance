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


from clearml import Task
from clearml import Logger

from clearml import Task, Logger

task = Task.init(
    project_name="HaMeem-Entry-Recognition-PoC",
    task_name="AdaFace training",
    tags=["AdaFace", "FaceRecognition", "FineTuning"],
    reuse_last_task_id=False
)

logger = Logger.current_logger()

# Track git + packages automatically
task.set_base_docker("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime")


# ──────────────────────────────────────────────
# CONFIG (all tunable parameters in one place)
# ──────────────────────────────────────────────
CONFIG = {
    'data_dir': '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/all_pic_aligned',
    'pretrained_ckpt': 'models/face_recognition/adaface_ir50_webface4m.ckpt',
    'batch_size': 64,
    'test_size': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs_warmup': 50,           # head only
    'epochs_full': 300,            # full fine-tune
    'lr_head': 5e-4,
    'lr_backbone': 5e-6,
    'weight_decay': 1e-5,
    'min_samples_per_class': 25,
    'k_neighbors': 7,
    'contrast_lambda': 0.5,
    'embedding_size': 512,
    'adaface_repo_path': 'models/AdaFace',
}

task.connect(CONFIG)

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
from core.recognition.train_recognition_utils import train_full, train_warmup, collect_train_embeddings, train_one_epoch_full, train_one_epoch_head_warmup
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
    logger.report_text(
    f"Dataset path: {CONFIG['data_dir']}\n"
    f"Total classes: {len(new_class_names)}\n"
    )

    logger.report_scalar(
        title="dataset",
        series="num_classes",
        value=len(new_class_names),
        iteration=0
    )
    # class_counts = Counter([lbl for _, lbl in dataset.samples])
    # logger.report_text(str(class_counts))

    train_loader, test_loader = create_loaders(dataset, CONFIG)

    # 2. Load model
    backbone, head = load_model(CONFIG, num_classes=len(new_class_names))
    task.connect({
    "model": {
        "backbone": "AdaFace IR50",
        "embedding_size": CONFIG["embedding_size"],
        "num_classes": len(new_class_names)
        }
    })

    # 3. Setup optimizer & scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(backbone, head, CONFIG)

    # -----------------------------
    # Phase 1: Head warm-up
    # -----------------------------
    print("Phase 1: Training head only...")

    for epoch in range(CONFIG["epochs_warmup"]):

        stats = train_one_epoch_head_warmup(
            head=head,
            backbone=backbone,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=CONFIG
        )

        scheduler.step()

        avg_loss = stats["avg_loss"]

        print(
            f"[Warmup Ep {epoch+1}/{CONFIG['epochs_warmup']}] "
            f"Loss: {avg_loss:.4f}"
        )

        # ---- ClearML logging ----
        logger.report_scalar(
            title="train_loss",
            series="warmup",
            value=avg_loss,
            iteration=epoch
        )


    # -----------------------------
    # Phase 2: Full fine-tuning
    # -----------------------------
    print("\nPhase 2: Fine-tuning backbone + contrastive...")

    for param in backbone.parameters():
        param.requires_grad = True

    for epoch in range(CONFIG["epochs_full"]):

        stats = train_one_epoch_full(
            backbone=backbone,
            head=head,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=CONFIG
        )

        scheduler.step()

        avg_loss = stats["avg_loss"]

        print(
            f"[Full Ep {epoch+1}/{CONFIG['epochs_full']}] "
            f"Loss: {avg_loss:.4f}"
        )

        # ---- ClearML logging ----
        logger.report_scalar(
            title="train_loss",
            series="full",
            value=avg_loss,
            iteration=epoch
        )
    torch.save(backbone.state_dict(), "backbone_final.pth")
    torch.save(head.state_dict(), "head_final.pth")

    task.upload_artifact("backbone_model", "backbone_final.pth")
    task.upload_artifact("classification_head", "head_final.pth")
    # 6. Collect embeddings & evaluate with FAISS
    train_emb, train_lbl = collect_train_embeddings(backbone, train_loader, CONFIG)
    task.upload_artifact(
    name="train_embeddings",
    artifact_object={
        "embeddings": train_emb,
        "labels": train_lbl
        }
    )

    results = evaluate_faiss(train_emb, train_lbl, test_loader, backbone, CONFIG)
    logger.report_scalar(
        title="evaluation",
        series="accuracy",
        value=results["accuracy"],
        iteration=0
    )

    logger.report_scalar(
        title="evaluation",
        series="avg_similarity",
        value=results["avg_max_sim"],
        iteration=0
    )

    logger.report_scalar(
        title="evaluation",
        series="median_similarity",
        value=results["median_max_sim"],
        iteration=0
    )

    logger.report_scalar(
        title="evaluation",
        series="min_similarity",
        value=results["min_max_sim"],
        iteration=0
    )

    logger.report_scalar(
        title="evaluation",
        series="samples_above_threshold",
        value=results["samples_above_threshold"],
        iteration=0
    )

    logger.report_scalar(
        title="evaluation",
        series="threshold_ratio",
        value=results["threshold_ratio"],
        iteration=0
    )
    print("All done.")

if __name__=='__main__':
    main()