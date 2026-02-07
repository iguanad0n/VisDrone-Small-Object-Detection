import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision.ops import box_iou
from typing import Dict, List, Any
from src.dataset import CLASS_NAMES

def collect_eda_stats(dataset, sample_ratio: float = 0.3) -> Dict[str, List]:
    print(f"[EDA] Сканирование датасета ({int(sample_ratio * 100)}%)...")

    stats = {
        "widths": [],
        "heights": [],
        "areas": [],
        "aspect_ratios": [],
        "centers_x": [],
        "centers_y": [],
        "labels": [],
        "objects_per_img": [],
        "all_boxes": []
    }

    indices = np.random.choice(len(dataset), int(len(dataset) * sample_ratio), replace=False)

    for idx in tqdm(indices):
        try:
            _, target = dataset[idx]
            boxes = target["boxes"].numpy()
            labels = target["labels"].numpy()

            if len(boxes) == 0:
                stats["objects_per_img"].append(0)
                continue

            stats["objects_per_img"].append(len(boxes))
            stats["labels"].extend(labels)
            stats["all_boxes"].append(torch.tensor(boxes))

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            w = np.abs(x2 - x1)
            h = np.abs(y2 - y1)

            area = w * h
            ar = w / (h + 1e-6)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            stats["widths"].extend(w)
            stats["heights"].extend(h)
            stats["areas"].extend(area)
            stats["aspect_ratios"].extend(ar)
            stats["centers_x"].extend(cx)
            stats["centers_y"].extend(cy)

        except Exception as e:
            continue

    return stats

def analyze_and_plot(stats: Dict[str, Any]):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    COLORMAP = "viridis"
    main_palette = sns.color_palette(COLORMAP, n_colors=8)

    COLOR_DISTRIBUTION = main_palette[3]
    COLOR_SAMPLES = main_palette[2]
    COLOR_PIE_CHART = [main_palette[2], main_palette[4], main_palette[6]]
    COLOR_ACCENT = "#dc1818"

    print(f"\n[Анализ] Обработка статистики для {len(stats['widths'])} объектов...")

    widths = np.array(stats["widths"])
    heights = np.array(stats["heights"])
    areas = np.array(stats["areas"])
    aspect_ratios = np.array(stats["aspect_ratios"])
    labels = np.array(stats["labels"])
    objects_per_img = np.array(stats["objects_per_img"])

    wh = np.column_stack((widths, heights))
    if len(wh) > 0:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(wh)
        anchors = kmeans.cluster_centers_
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    else:
        anchors = np.zeros((5, 2))

    n_small = np.sum(areas < 32**2)
    n_medium = np.sum((areas >= 32**2) & (areas < 96**2))
    n_large = np.sum(areas >= 96**2)

    iou_scores = []
    if "all_boxes" in stats:
        for boxes in stats["all_boxes"]:
            if len(boxes) > 1:
                iou_matrix = box_iou(boxes, boxes)
                iou_matrix.fill_diagonal_(0)
                max_iou, _ = iou_matrix.max(dim=1)
                iou_scores.extend(max_iou.numpy())
    iou_scores = np.array(iou_scores) if iou_scores else np.array([])

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(26, 13), constrained_layout=True)

    ax1 = axes[0, 0]
    x_lim = np.percentile(widths, 99.9) if len(widths) > 0 else 100
    y_lim = np.percentile(heights, 99.9) if len(heights) > 0 else 100

    ax1.scatter(widths, heights, alpha=0.2, s=10, color=COLOR_SAMPLES, label='Объекты', rasterized=True)
    ax1.scatter(anchors[:, 0], anchors[:, 1], color=COLOR_ACCENT, s=150, marker='x', linewidth=2, label='Якоря', zorder=10)

    median_w = int(np.median(widths)) if len(widths) > 0 else 0
    median_h = int(np.median(heights)) if len(heights) > 0 else 0
    
    ax1.set_title(f'1. Геометрия боксов\nМедиана: {median_w}x{median_h} px')
    ax1.set_xlabel('Ширина (px)')
    ax1.set_ylabel('Высота (px)')
    ax1.set_xlim(0, x_lim * 1.1)
    ax1.set_ylim(0, y_lim * 1.1)
    ax1.legend(loc='upper right')

    ax2 = axes[0, 1]
    sns.histplot(aspect_ratios, bins=50, binrange=(0, 4.0), ax=ax2, kde=True, element="step", alpha=0.6, color=COLOR_DISTRIBUTION)
    ax2.axvline(1.0, color=COLOR_ACCENT, linestyle='--', linewidth=1.5, label='Квадрат (1:1)')
    ax2.set_title('2. Соотношение сторон (W/H)')
    ax2.set_xlim(0, 4.0)
    ax2.legend(loc='upper right')

    ax3 = axes[0, 2]
    unique_labels, counts_val = np.unique(labels, return_counts=True)
    
    if len(counts_val) > 0:
        sorted_idx = np.argsort(counts_val)
        unique_labels = unique_labels[sorted_idx]
        counts_val = counts_val[sorted_idx]
        
        class_names_plot = []
        for k in unique_labels:
            idx = int(k)
            if idx < len(CLASS_NAMES):
                class_names_plot.append(CLASS_NAMES[idx])
            else:
                class_names_plot.append(str(idx))

        sns.barplot(x=counts_val, y=class_names_plot, ax=ax3, palette=COLORMAP, orient='h')
    
    ax3.set_title(f'3. Распределение классов (Всего: {len(unique_labels)})')
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="x", linestyle="--", alpha=0.5)

    ax4 = axes[0, 3]
    den_lim = np.percentile(objects_per_img, 99.9) if len(objects_per_img) > 0 else 10
    sns.histplot(objects_per_img, bins=40, ax=ax4, kde=True, element="step", alpha=0.6, color=COLOR_DISTRIBUTION)
    max_objs = np.max(objects_per_img) if len(objects_per_img) > 0 else 0
    ax4.set_title(f'4. Плотность сцены (Макс: {max_objs})')
    ax4.set_xlim(0, den_lim * 1.1)

    ax5 = axes[1, 0]
    max_w_data = np.max(stats["centers_x"]) if len(stats["centers_x"]) > 0 else 640
    max_h_data = np.max(stats["centers_y"]) if len(stats["centers_y"]) > 0 else 640
    
    if len(stats["centers_x"]) > 0:
        ax5.hist2d(stats["centers_x"], stats["centers_y"], bins=64, range=[[0, max_w_data], [0, max_h_data]], cmap=COLORMAP)
    ax5.invert_yaxis()
    ax5.set_title(f'5. Пространственное распределение')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    ax6 = axes[1, 1]
    if len(iou_scores) > 0:
        pos_iou_scores = iou_scores[iou_scores > 0.01]
        zero_pct = (len(iou_scores) - len(pos_iou_scores)) / len(iou_scores) * 100 if len(iou_scores) > 0 else 0
        sns.histplot(pos_iou_scores, bins=50, binrange=(0, 1), ax=ax6, kde=True, element="step", alpha=0.6, color=COLOR_DISTRIBUTION)
        ax6.set_title(f'6. IoU Перекрытия\n(Изолированные: {zero_pct:.1f}%)')
    else:
        ax6.set_title('6. IoU Перекрытия (Нет данных)')
    
    ax6.axvline(0.5, color=COLOR_ACCENT, linestyle='--', label='Порог 0.5')
    ax6.set_xlim(0, 1.0)
    ax6.legend(loc='upper right')

    ax7 = axes[1, 2]
    ax7.pie([n_small, n_medium, n_large], labels=['Мелкие', 'Средние', 'Крупные'],
            autopct='%1.1f%%', startangle=140, colors=COLOR_PIE_CHART, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1, 'width': 0.5})
    ax7.set_title('7. Группы масштабов (COCO Metrics)')

    ax8 = axes[1, 3]
    ax8.axis('off')

    if len(anchors) > 0:
        rec_anchors = np.round(anchors).astype(int).tolist()
        anchors_report = "\n".join([f"   • [{w}, {h}]" for w, h in rec_anchors])
    else:
        anchors_report = "N/A"

    if len(counts_val) > 0:
        imbalance = np.max(counts_val) / (np.min(counts_val) + 1e-6)
    else:
        imbalance = 0

    small_objects_ratio = n_small / len(areas) * 100 if len(areas) > 0 else 0
    median_aspect = np.median(aspect_ratios) if len(aspect_ratios) > 0 else 0
    
    class_areas = {}
    for lbl, area in zip(labels, areas):
        if lbl not in class_areas: class_areas[lbl] = []
        class_areas[lbl].append(area)

    min_area_val = float('inf')
    min_area_name = "N/A"

    for lbl, area_list in class_areas.items():
        median_area = np.median(area_list)
        if median_area < min_area_val:
            min_area_val = median_area
            idx = int(lbl)
            if idx < len(CLASS_NAMES):
                min_area_name = CLASS_NAMES[idx]
            else:
                min_area_name = str(idx)

    min_area_dim = int(np.sqrt(min_area_val)) if min_area_val != float('inf') else 0

    summary_text = (
        "Сводка по датасету VisDrone\n"
        "───────────────────────────\n\n"
        "1. Рекомендуемые якоря (K-Means):\n"
        f"{anchors_report}\n\n"
        "2. Геометрия объектов:\n"
        f"   • Медианный размер:    {median_w}x{median_h} px\n"
        f"   • Доля малых (<32²):   {small_objects_ratio:.1f}%\n"
        f"   • Медианное W/H:       {median_aspect:.2f}\n"
        f"   • Самый мелкий класс: {min_area_name} (~{min_area_dim}px)\n\n"
        "3. Характеристики изображений:\n"
        f"   • Разрешение (Макс):   {int(max_w_data)}x{int(max_h_data)}\n"
        f"   • Плотность (Медиана): {int(np.median(objects_per_img))} об/кадр\n"
        f"   • Коэффициент дисбаланса:   1 : {int(imbalance)}"
    )

    ax8.text(0.05, 0.5, summary_text, fontsize=11, family='monospace', va='center', ha='left',
             bbox=dict(facecolor='#f8f9fa', edgecolor='#dddddd', boxstyle='round,pad=1.2', linewidth=1))

    plt.show()

if __name__ == "__main__":
    from src.config import CONFIG
    from src.dataset import VisDroneDataset, get_transforms
    
    print("[Main] Загрузка датасета для анализа...")
    dataset = VisDroneDataset(
        images_dir=CONFIG['train_images'],
        annotations_dir=CONFIG['train_annotations'],
        split='train',
        transforms=get_transforms(train=False, img_size=CONFIG["img_size"]),
        filter_empty_samples=True
    )
    
    stats = collect_eda_stats(dataset, sample_ratio=1)
    analyze_and_plot(stats)