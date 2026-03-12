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



def train_one_epoch_head_warmup(head: nn.Module,
                         backbone: nn.Module,
                         train_loader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         criterion: nn.Module,
                         config: Dict):

    head.train()

    total_loss = 0.0
    batch_count = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(config['device'])
        labels = labels.to(config['device'])

        with torch.no_grad():
            embeddings, _ = backbone(images)

        embeddings = F.normalize(embeddings)
        norms = embeddings.norm(dim=1, keepdim=True)

        logits = head(embeddings, norms, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_count += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    stats = {
        "avg_loss": avg_loss,
        "total_samples": total_samples,
        "num_batches": batch_count
    }

    return stats

def train_warmup(head: nn.Module,
                 backbone: nn.Module,
                 train_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 scheduler: torch.optim.lr_scheduler,
                 config: Dict):

    print("Phase 1: Training head only...")

    for epoch in range(config['epochs_warmup']):

        stats = train_one_epoch_head_warmup(
            head=head,
            backbone=backbone,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config
        )

        scheduler.step()

        print(
            f"[Warm-up Ep {epoch+1}/{config['epochs_warmup']}] "
            f"Avg Loss: {stats['avg_loss']:.4f}"
        )
# def train_warmup(head: nn.Module, backbone: nn.Module, train_loader: DataLoader,
#                  optimizer: torch.optim.Optimizer, criterion: nn.Module,
#                  scheduler: torch.optim.lr_scheduler, config: Dict):
#     print("Phase 1: Training head only...")
#     for epoch in range(config['epochs_warmup']):
#         total_loss = 0.0
#         batch_count = 0
#         head.train()

#         for images, labels in train_loader:
#             images = images.to(config['device'])
#             labels = labels.to(config['device'])

#             with torch.no_grad():
#                 embeddings, _ = backbone(images)

#             embeddings = F.normalize(embeddings)
#             norms = embeddings.norm(dim=1, keepdim=True)

#             logits = head(embeddings, norms, labels)
#             loss = criterion(logits, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * images.size(0)
#             batch_count += 1

#         scheduler.step()
#         avg_loss = total_loss / (batch_count * config['batch_size']) if batch_count > 0 else 0
#         print(f"[Warm-up Ep {epoch+1}/{config['epochs_warmup']}] Avg Loss: {avg_loss:.4f}")


def train_one_epoch_full(backbone: nn.Module,
                         head: nn.Module,
                         train_loader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         criterion: nn.Module,
                         config: Dict):

    backbone.train()
    head.train()

    total_loss = 0.0
    total_samples = 0
    batch_count = 0

    for images, labels in train_loader:
        images = images.to(config['device'])
        labels = labels.to(config['device'])

        embeddings, _ = backbone(images)
        embeddings = F.normalize(embeddings)
        norms = embeddings.norm(dim=1, keepdim=True)

        logits = head(embeddings, norms, labels)
        ce_loss = criterion(logits, labels)

        # ---- Contrastive loss (SupCon style) ----
        sim_matrix = torch.mm(embeddings, embeddings.t())

        mask_same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask_diff = 1 - mask_same - torch.eye(len(labels), device=config['device'])

        pos_sim = (sim_matrix * mask_same).sum(1) / (mask_same.sum(1) + 1e-8)
        neg_sim = (sim_matrix * mask_diff).sum(1) / (mask_diff.sum(1) + 1e-8)

        contrast_loss = F.relu(0.4 - pos_sim + neg_sim).mean()

        loss = ce_loss + config['contrast_lambda'] * contrast_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_count += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    stats = {
        "avg_loss": avg_loss,
        "total_samples": total_samples,
        "num_batches": batch_count
    }

    return stats

def train_full(backbone: nn.Module,
               head: nn.Module,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               scheduler: torch.optim.lr_scheduler,
               config: Dict):

    print("\nPhase 2: Fine-tuning backbone + contrastive...")

    # unfreeze backbone
    for param in backbone.parameters():
        param.requires_grad = True

    for epoch in range(config['epochs_full']):

        stats = train_one_epoch_full(
            backbone=backbone,
            head=head,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config
        )

        scheduler.step()

        print(
            f"[Full Ep {epoch+1}/{config['epochs_full']}] "
            f"Avg Loss: {stats['avg_loss']:.4f}"
        )
# def train_full(backbone: nn.Module, head: nn.Module, train_loader: DataLoader,
#                optimizer: torch.optim.Optimizer, criterion: nn.Module,
#                scheduler: torch.optim.lr_scheduler, config: Dict):
#     print("\nPhase 2: Fine-tuning backbone + contrastive...")
#     for param in backbone.parameters():
#         param.requires_grad = True  # full unfreeze

#     for epoch in range(config['epochs_full']):
#         total_loss = 0.0
#         batch_count = 0
#         backbone.train()
#         head.train()

#         for images, labels in train_loader:
#             images = images.to(config['device'])
#             labels = labels.to(config['device'])

#             embeddings, _ = backbone(images)
#             embeddings = F.normalize(embeddings)
#             norms = embeddings.norm(dim=1, keepdim=True)

#             logits = head(embeddings, norms, labels)
#             ce_loss = criterion(logits, labels)

#             # Contrastive loss (SupCon style)
#             sim_matrix = torch.mm(embeddings, embeddings.t())
#             mask_same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
#             mask_diff = 1 - mask_same - torch.eye(len(labels), device=config['device'])
#             pos_sim = (sim_matrix * mask_same).sum(1) / (mask_same.sum(1) + 1e-8)
#             neg_sim = (sim_matrix * mask_diff).sum(1) / (mask_diff.sum(1) + 1e-8)
#             contrast_loss = F.relu(0.4 - pos_sim + neg_sim).mean()

#             loss = ce_loss + config['contrast_lambda'] * contrast_loss

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 list(backbone.parameters()) + list(head.parameters()), max_norm=1.0
#             )
#             optimizer.step()

#             total_loss += loss.item() * images.size(0)
#             batch_count += 1

#         scheduler.step()
#         avg_loss = total_loss / (batch_count * config['batch_size']) if batch_count > 0 else 0
#         print(f"[Full Ep {epoch+1}/{config['epochs_full']}] Avg Loss: {avg_loss:.4f}")

def collect_train_embeddings(backbone: nn.Module, train_loader: DataLoader, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    print("Collecting train embeddings...")
    train_emb = []
    train_lbl = []

    backbone.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(config['device'])
            emb, _ = backbone(images)
            emb = F.normalize(emb)
            train_emb.append(emb.cpu().numpy())
            train_lbl.append(labels.cpu().numpy())

    train_emb = np.concatenate(train_emb).astype(np.float32)
    train_lbl = np.concatenate(train_lbl)
    return train_emb, train_lbl