import os
import random
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    img = (img * std) + mean
    
    return np.clip(img, 0, 1)

def visualize_batch(loader, class_names: List[str], num_images: int = 4):
    cmap = plt.get_cmap('tab10')
    class_colors = {i: cmap(i / len(class_names)) for i in range(len(class_names))}

    images, targets = next(iter(loader))
    batch_size = len(images)
    samples_to_show = min(batch_size, num_images)

    cols = 2
    rows = math.ceil(samples_to_show / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10 * rows), constrained_layout=True)
    if samples_to_show == 1: 
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    print(f"\n[Visualizer] Визуализация {samples_to_show} примеров...")

    for idx in range(samples_to_show):
        ax = axes[idx]

        img_np = denormalize_image(images[idx])
        ax.imshow(img_np)

        boxes = targets[idx]['boxes'].cpu().numpy()
        labels = targets[idx]['labels'].cpu().numpy()

        for box, label_idx in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            if label_idx < len(class_names):
                color = class_colors.get(label_idx, (1, 0, 0))
                class_name = class_names[label_idx]
            else:
                color = (1, 0, 0)
                class_name = str(label_idx)

            rect = patches.Rectangle(
                (x1, y1), w, h, 
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.9
            )
            ax.add_patch(rect)

            ax.text(
                x1, y1 - 5, class_name, 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.8, pad=1)
            )

        ax.set_title(f"Sample {idx} | Objects: {len(boxes)}")
        ax.axis('off')

    if isinstance(axes, (list, np.ndarray)):
        for ax in axes[samples_to_show:]:
            ax.axis('off')

    plt.show()