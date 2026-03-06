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



def evaluate_faiss(train_emb: np.ndarray, train_lbl: np.ndarray,
                   test_loader: DataLoader, backbone: nn.Module,
                   config: Dict) -> Dict[str, Any]:
    print("Building FAISS index...")
    d = train_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_emb)

    correct = 0
    total = 0
    max_sims = []

    print("Evaluating test set...")
    backbone.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config['device'])
            labels_np = labels.cpu().numpy()
            emb, _ = backbone(images)
            emb = F.normalize(emb).cpu().numpy().astype(np.float32)

            D, I = index.search(emb, k=config['k_neighbors'])

            for i in range(emb.shape[0]):
                neighbors = train_lbl[I[i]]
                pred = Counter(neighbors).most_common(1)[0][0]
                max_sim = D[i][0]
                max_sims.append(max_sim)

                if pred == labels_np[i]:
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0

    results = {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'avg_max_sim': np.mean(max_sims),
        'median_max_sim': np.median(max_sims),
        'min_max_sim': np.min(max_sims),
        'samples_above_threshold': sum(1 for s in max_sims if s >= 0.35),
        'threshold_ratio': sum(1 for s in max_sims if s >= 0.35) / total if total > 0 else 0
    }

    print(f"FAISS + k={config['k_neighbors']} voting accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Avg max sim: {results['avg_max_sim']:.4f}")
    print(f"Median max sim: {results['median_max_sim']:.4f}")
    print(f"Min max sim: {results['min_max_sim']:.4f}")
    print(f"Samples with sim >= 0.35: {results['samples_above_threshold']}/{results['total']} ({results['threshold_ratio']:.3f})")

    return results
