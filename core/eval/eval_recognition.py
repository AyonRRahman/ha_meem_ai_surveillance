from collections import Counter
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# Build FAISS index
# ------------------------------------------------------------
def build_faiss_index(train_emb: np.ndarray):

    d = train_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_emb)

    return index


# ------------------------------------------------------------
# Predict label from neighbors
# ------------------------------------------------------------
def predict_label(neighbor_labels: np.ndarray, strategy: str):

    if strategy == "nearest":
        return neighbor_labels[0]

    elif strategy == "majority":
        return Counter(neighbor_labels).most_common(1)[0][0]

    else:
        raise ValueError("Voting strategy must be 'nearest' or 'majority'")


# ------------------------------------------------------------
# Compute Top-K correctness
# ------------------------------------------------------------
def compute_topk(gt: int, neighbor_labels: np.ndarray, ks: List[int]):

    results = {}

    for k in ks:
        results[k] = int(gt in neighbor_labels[:k])

    return results


# ------------------------------------------------------------
# Compute classification metrics
# ------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred):

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return precision, recall, f1, cm, report


# ------------------------------------------------------------
# Compute class accuracy
# ------------------------------------------------------------
def compute_class_accuracy(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_accuracy = {}

    for cls in np.unique(y_true):

        idx = np.where(y_true == cls)[0]

        class_accuracy[int(cls)] = np.mean(
            y_pred[idx] == y_true[idx]
        )

    return class_accuracy


# ------------------------------------------------------------
# Print results nicely
# ------------------------------------------------------------
def print_results(results, label_to_class):

    print("\nEvaluation Results")
    print("-------------------")

    print(f"Accuracy: {results['accuracy']:.4f}")

    print(f"Top1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"Top3 Accuracy: {results['top3_accuracy']:.4f}")
    print(f"Top5 Accuracy: {results['top5_accuracy']:.4f}")

    print(f"Precision: {results['precision_macro']:.4f}")
    print(f"Recall: {results['recall_macro']:.4f}")
    print(f"F1: {results['f1_macro']:.4f}")

    print(f"Avg similarity: {results['avg_max_sim']:.4f}")
    print(f"Median similarity: {results['median_max_sim']:.4f}")

    print(
        f"Samples above threshold: "
        f"{results['samples_above_threshold']}/{results['total']}"
    )

    print("\nClass Accuracy:")

    for c, v in results["class_accuracy"].items():
        # print(f"Class {c}: {v:.3f}")
        print(f"{label_to_class[c]}: {v:.3f}")


# ------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------
def evaluate_faiss(
        train_emb: np.ndarray,
        train_lbl: np.ndarray,
        test_loader,
        backbone,
        label_to_class: Dict,
        config: Dict
    ) -> Dict[str, Any]:

    print("Building FAISS index...")

    index = build_faiss_index(train_emb)

    voting = config.get("voting", "majority")
    ks = [1, 3, 5]

    total = 0
    correct = 0

    topk_correct = {k: 0 for k in ks}

    max_sims = []

    y_true = []
    y_pred = []

    backbone.eval()

    print("Evaluating test set...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(config["device"])
            labels_np = labels.cpu().numpy()

            emb, _ = backbone(images)

            emb = F.normalize(emb).cpu().numpy().astype(np.float32)

            D, I = index.search(emb, k=config["k_neighbors"])

            for i in range(len(labels_np)):

                gt = labels_np[i]

                neighbors = train_lbl[I[i]]

                pred = predict_label(neighbors, voting)

                y_true.append(gt)
                y_pred.append(pred)

                if pred == gt:
                    correct += 1

                # --- Top-K metrics ---
                topk = compute_topk(gt, neighbors, ks)

                for k in ks:
                    topk_correct[k] += topk[k]

                # --- similarity ---
                max_sims.append(D[i][0])

                total += 1

    accuracy = correct / total

    topk_acc = {k: topk_correct[k] / total for k in ks}

    precision, recall, f1, cm, report = compute_classification_metrics(y_true, y_pred)

    class_accuracy = compute_class_accuracy(y_true, y_pred)

    results = {

        "accuracy": accuracy,

        "top1_accuracy": topk_acc[1],
        "top3_accuracy": topk_acc[3],
        "top5_accuracy": topk_acc[5],

        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,

        "class_accuracy": class_accuracy,
        "confusion_matrix": cm,
        "classification_report": report,

        "avg_max_sim": np.mean(max_sims),
        "median_max_sim": np.median(max_sims),
        "min_max_sim": np.min(max_sims),

        "samples_above_threshold": sum(s >= 0.35 for s in max_sims),
        "threshold_ratio": sum(s >= 0.35 for s in max_sims) / total,

        "total": total
    }

    print_results(results, label_to_class)

    return results

# import sys
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from PIL import Image
# import numpy as np
# from collections import Counter
# import gc
# import faiss
# from typing import Tuple, List, Dict, Any



# def evaluate_faiss(train_emb, train_lbl, test_loader, backbone, config):

#     print("Building FAISS index...")

#     d = train_emb.shape[1]
#     index = faiss.IndexFlatIP(d)
#     index.add(train_emb)

#     correct = 0
#     total = 0

#     top1_correct = 0
#     top3_correct = 0
#     top5_correct = 0

#     max_sims = []

#     y_true = []
#     y_pred = []

#     backbone.eval()

#     print("Evaluating test set...")

#     with torch.no_grad():

#         for images, labels in test_loader:

#             images = images.to(config['device'])
#             labels_np = labels.cpu().numpy()

#             emb, _ = backbone(images)
#             emb = F.normalize(emb).cpu().numpy().astype(np.float32)

#             D, I = index.search(emb, k=config['k_neighbors'])

#             for i in range(len(labels_np)):

#                 neighbors = train_lbl[I[i]]
#                 pred = Counter(neighbors).most_common(1)[0][0]

#                 gt = labels_np[i]

#                 y_true.append(gt)
#                 y_pred.append(pred)

#                 if pred == gt:
#                     correct += 1

#                 # -------- Top-K metrics --------

#                 top_labels = neighbors

#                 if gt in top_labels[:1]:
#                     top1_correct += 1

#                 if gt in top_labels[:3]:
#                     top3_correct += 1

#                 if gt in top_labels[:5]:
#                     top5_correct += 1

#                 # similarity statistics
#                 max_sim = D[i][0]
#                 max_sims.append(max_sim)

#                 total += 1

#     accuracy = correct / total

#     top1_acc = top1_correct / total
#     top3_acc = top3_correct / total
#     top5_acc = top5_correct / total

#     precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
#     recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

#     cm = confusion_matrix(y_true, y_pred)

#     report = classification_report(
#         y_true,
#         y_pred,
#         output_dict=True,
#         zero_division=0
#     )

#     # class accuracy
#     class_accuracy = {}

#     y_true_np = np.array(y_true)
#     y_pred_np = np.array(y_pred)

#     for cls in np.unique(y_true_np):

#         idx = np.where(y_true_np == cls)[0]

#         class_accuracy[int(cls)] = np.mean(
#             y_pred_np[idx] == y_true_np[idx]
#         )

#     results = {

#         "accuracy": accuracy,
#         "top1_accuracy": top1_acc,
#         "top3_accuracy": top3_acc,
#         "top5_accuracy": top5_acc,

#         "precision_macro": precision,
#         "recall_macro": recall,
#         "f1_macro": f1,

#         "class_accuracy": class_accuracy,
#         "confusion_matrix": cm,
#         "classification_report": report,

#         "avg_max_sim": np.mean(max_sims),
#         "median_max_sim": np.median(max_sims),
#         "min_max_sim": np.min(max_sims),

#         "samples_above_threshold": sum(s >= 0.35 for s in max_sims),
#         "threshold_ratio": sum(s >= 0.35 for s in max_sims) / total
#     }

#     print("\nEvaluation Results")
#     print("-------------------")

#     print(f"Accuracy: {accuracy:.4f}")

#     print(f"Top1 Accuracy: {top1_acc:.4f}")
#     print(f"Top3 Accuracy: {top3_acc:.4f}")
#     print(f"Top5 Accuracy: {top5_acc:.4f}")

#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1: {f1:.4f}")

#     print(f"Avg similarity: {results['avg_max_sim']:.4f}")
#     print(f"Median similarity: {results['median_max_sim']:.4f}")

#     print(f"Samples above threshold: {results['samples_above_threshold']}/{total}")


#     print("\nClass Accuracy:")
#     for c, v in class_accuracy.items():
#         print(f"Class {c}: {v:.3f}")

#     return results
