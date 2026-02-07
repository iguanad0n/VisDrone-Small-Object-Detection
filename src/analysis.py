import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict
from src.config import CONFIG
from src.dataset import VisDroneDataset, get_transforms
from src.models import get_model

from src.models.retinanet_ref import ReferenceRetinaNet
from src.models.retinanet_custom import CustomRetinaNet
from src.models.fasterrcnn_ref import ReferenceFasterRCNN
from src.models.fasterrcnn_custom import CustomFasterRCNN

class Analyzer:
    def __init__(self, model_configs: Dict[str, Path], output_dir: str = "analysis_results"):
        self.model_configs = model_configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logs_data = {}

        self._load_logs()

    def _load_logs(self):
        print(f"[Analyzer] Загрузка логов для {len(self.model_configs)} моделей...")

        for model_name, ckpt_dir in self.model_configs.items():
            log_path = ckpt_dir / "training_log.json"
            if not log_path.exists():
                print(f"Лог не найден: {log_path}")
                continue

            try:
                with open(log_path, 'r') as f:
                    self.logs_data[model_name] = json.load(f)
            except Exception as e:
                print(f"Ошибка чтения {log_path}: {e}")

    def plot_comparative_metrics(self):
        metrics_map = {
            "mAP (0.5:0.95)": "map",
            "mAP @ 0.50": "map_50",
            "mAP Small": "map_small",
            "mAP Medium": "map_medium",
            "mAP Large": "map_large",
            "Total Loss": "loss"
        }

        plt.rcParams['figure.figsize'] = (24, 14)
        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()

        for idx, (title, metric_key) in enumerate(metrics_map.items()):
            ax = axes[idx]

            for model_name, history in self.logs_data.items():
                key_to_plot = metric_key
                if key_to_plot not in history:
                    if metric_key == "loss" and "train_loss" in history:
                        key_to_plot = "train_loss"
                    elif metric_key == "map" and "val_map" in history:
                        key_to_plot = "val_map"
                    elif metric_key == "map_50" and "val_map_50" in history:
                        key_to_plot = "val_map_50"
                    elif metric_key == "map_small" and "val_map_s" in history:
                        key_to_plot = "val_map_s"
                    elif metric_key == "map_medium" and "val_map_m" in history:
                        key_to_plot = "val_map_m"
                    elif metric_key == "map_large" and "val_map_l" in history:
                        key_to_plot = "val_map_l"
                    else:
                        continue

                values = history[key_to_plot]
                epochs = range(1, len(values) + 1)

                if "loss" in key_to_plot:
                    ax.plot(epochs, values, label=model_name, linewidth=2, alpha=0.8)
                else:
                    ax.plot(epochs, values, marker='o', markersize=4, label=model_name, linewidth=2)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = self.output_dir / "comparative_metrics.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Analyzer] Графики сохранены в: {save_path}")
        plt.show()

    def create_summary_table(self):
        summary_rows = []

        for model_name, history in self.logs_data.items():
            map_key = "val_map" if "val_map" in history else "map"

            if map_key not in history or len(history[map_key]) == 0:
                continue

            best_epoch_idx = pd.Series(history[map_key]).idxmax()

            def get_val(key, alt_key, idx):
                k = key if key in history else alt_key
                return history[k][idx] if k in history else 0.0

            row = {
                "Model": model_name,
                "Best Epoch": best_epoch_idx + 1,
                "mAP": get_val("val_map", "map", best_epoch_idx),
                "mAP50": get_val("val_map_50", "map_50", best_epoch_idx),
                "mAP_S": get_val("val_map_s", "map_small", best_epoch_idx),
                "mAP_M": get_val("val_map_m", "map_medium", best_epoch_idx),
                "mAP_L": get_val("val_map_l", "map_large", best_epoch_idx),
                "Min Loss": min(history.get("train_loss", history.get("loss", [0])))
            }
            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['float']).columns
            df[numeric_cols] = df[numeric_cols].round(4)

            json_path = self.output_dir / "final_metrics_summary.json"
            df.to_json(json_path, orient="records", indent=4)
            df.to_csv(self.output_dir / "final_metrics_summary.csv", index=False)
            print(f"[Analyzer] Сводная таблица сохранена в: {json_path}")
            print(df)
        else:
            print("[Analyzer] Нет данных для таблицы.")
        return df

    def visualize_predictions(self, model, dataset, device, num_samples=3, threshold=0.3):
        model.eval()
        model.to(device)

        max_idx = len(dataset)
        indices = torch.randperm(max_idx)[:min(num_samples, max_idx)]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in indices:
            image, target = dataset[i]
            img_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img_tensor)[0]

            img_cpu = image.cpu()
            img_cpu = img_cpu * std + mean
            img_numpy = img_cpu.permute(1, 2, 0).numpy()
            img_numpy = np.clip(img_numpy, 0, 1)

            img_gt = (img_numpy * 255).astype(np.uint8).copy()
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
            img_pred = (img_numpy * 255).astype(np.uint8).copy()
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)

            for box in target['boxes']:
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)

            keep = prediction['scores'] > threshold
            pred_boxes = prediction['boxes'][keep]
            pred_scores = prediction['scores'][keep]

            for box, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_pred, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1, 2, figsize=(16, 8))

            ax[0].imshow(img_gt)
            ax[0].set_title("Ground Truth (Green)", fontsize=14)
            ax[0].axis('off')

            ax[1].imshow(img_pred)
            ax[1].set_title(f"Prediction (Red) | Thresh={threshold}", fontsize=14)
            ax[1].axis('off')

            plt.tight_layout()
            save_path = self.output_dir / f"prediction_sample_{i.item()}.png"
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

def run_analysis():
    experiment_paths = {
        "RetinaNet (Reference)":    CONFIG["checkpoint_dir"] / "retinanet_ref",
        "RetinaNet (Custom)":       CONFIG["checkpoint_dir"] / "retinanet_custom",
        "Faster R-CNN (Reference)": CONFIG["checkpoint_dir"] / "faster_rcnn_ref",
        "Faster R-CNN (Custom)":    CONFIG["checkpoint_dir"] / "faster_rcnn_custom",
    }

    print(f"Запуск анализа для {len(experiment_paths)} моделей...\n")

    analyzer = Analyzer(
        model_configs=experiment_paths,
        output_dir="visdrone_final_results"
    )

    print("Построение графиков метрик...")
    analyzer.plot_comparative_metrics()

    print("Генерация сводной таблицы...")
    summary_df = analyzer.create_summary_table()
    
    print("Визуализация работы лучшей модели...")

    if not summary_df.empty:
        best_row = summary_df.sort_values(by="mAP", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        print(f"Лидер: {best_model_name} (mAP: {best_row['mAP']})")

        name_map = {
            "RetinaNet (Reference)": "retinanet_ref",
            "RetinaNet (Custom)": "retinanet_custom",
            "Faster R-CNN (Reference)": "faster_rcnn_ref",
            "Faster R-CNN (Custom)": "faster_rcnn_custom"
        }
        
        internal_name = name_map.get(best_model_name)
        
        if internal_name:
             try:
                best_ckpt_path = experiment_paths[best_model_name] / "best_model.pth"
                if not best_ckpt_path.exists():
                     print(f"Checkpoint not found at {best_ckpt_path}, trying last_model.pth")
                     best_ckpt_path = experiment_paths[best_model_name] / "last_model.pth"

                if best_ckpt_path.exists():
                    print(f"Loading best model from {best_ckpt_path}")
                    
                    viz_model = get_model(internal_name, CONFIG)
                    viz_model.to(CONFIG["device"])

                    checkpoint = torch.load(best_ckpt_path, map_location=CONFIG["device"])
                    viz_model.load_state_dict(checkpoint["model_state_dict"])

                    val_dataset = VisDroneDataset(
                        images_dir=CONFIG['val_images'],
                        annotations_dir=CONFIG['val_annotations'],
                        split='val',
                        transforms=get_transforms(train=False, img_size=CONFIG["img_size"]),
                        filter_empty_samples=False,
                        min_box_area=CONFIG["min_box_area"]
                    )

                    analyzer.visualize_predictions(
                        model=viz_model,
                        dataset=val_dataset,
                        device=CONFIG["device"],
                        num_samples=4,
                        threshold=0.3
                    )
                else:
                    print(f"No checkpoint found for {best_model_name}")

             except Exception as e:
                print(f"Не удалось визуализировать: {e}")
        else:
            print(f"Unknown model name mapping for {best_model_name}")
    else:
        print("Нет данных для анализа. Убедитесь, что модели обучились и создали training_log.json")

if __name__ == "__main__":
    run_analysis()