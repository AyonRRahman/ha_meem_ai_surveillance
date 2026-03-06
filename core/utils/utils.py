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

# Get absolute path of current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two directories: core/train → project_folder
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# Add to sys.path
sys.path.append(project_root)

from models.face_recognition.AdaFace.net import IR_50
from models.face_recognition.AdaFace.head import AdaFace

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared.")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(config: Dict, num_classes: int) -> Tuple[nn.Module, nn.Module]:
    print("\nLoading backbone...")
    backbone = IR_50(input_size=(112, 112))
    checkpoint = torch.load(config['pretrained_ckpt'], map_location='cpu', weights_only=True)
    backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    backbone = backbone.to(config['device'])

    # Freeze all initially
    for param in backbone.parameters():
        param.requires_grad = False

    head = AdaFace(embedding_size=config['embedding_size'], classnum=num_classes).to(config['device'])

    return backbone, head

def setup_optimizer_and_scheduler(backbone: nn.Module, head: nn.Module, config: Dict):
    optimizer = Adam([
        {'params': head.parameters(), 'lr': config['lr_head']},
        {'params': backbone.parameters(), 'lr': config['lr_backbone']}
    ], weight_decay=config['weight_decay'])

    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs_warmup'] + config['epochs_full'], eta_min=1e-7)

    return optimizer, scheduler
