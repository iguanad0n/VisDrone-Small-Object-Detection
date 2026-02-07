import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "num_classes": 9,
    "min_box_area": 4.0, 
    "img_size": 1024, 
    "max_img_size": 1333,
    
    "batch_size": 32,
    "num_epochs": 40,
    "learning_rate": 0.02,
    "weight_decay": 0.0005,
    "momentum": 0.9,
    "max_grad_norm": 10.0, 
    "warmup_epochs": 5,

    "anchor_sizes": (
        (11, 14, 18),
        (23, 29, 36),
        (45, 57, 72),
        (90, 113, 143),
        (180, 226, 286)
    ),
    "aspect_ratios": ((0.5, 1.0, 2.0),) * 5,

    "num_workers": 8, 
    "seed": 42,
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data",
    "checkpoint_dir": ROOT_DIR / "checkpoints",
    "logs_dir": ROOT_DIR / "logs",
    "train_images": ROOT_DIR / "train/images",
    "train_annotations": ROOT_DIR / "train/annotations",
    "val_images": ROOT_DIR / "validation/images",
    "val_annotations": ROOT_DIR / "validation/annotations",
}


CONFIG["checkpoint_dir"].mkdir(exist_ok=True, parents=True)
CONFIG["logs_dir"].mkdir(exist_ok=True, parents=True)