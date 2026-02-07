import torch
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.model.to(self.device)

        self.current_epoch = 0
        self.best_metric = 0.0

        self.history = {
            "train_loss": [],
            "val_map": [],
            "val_map_50": [],
            "val_map_s": [],
            "val_map_m": [],
            "val_map_l": [],
            "lr": []
        }

        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "training_log.json"

    def load_checkpoint(self, filename: str = "last_model.pth") -> bool:
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"[Init] Чекпоинт {path} не найден. Начинаем с нуля.")
            return False

        print(f"[Init] Загрузка чекпоинта: {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.history = checkpoint.get("history", self.history)

        print(f"[Init] Успешно восстановлено! Эпоха: {self.current_epoch}, Best mAP: {self.best_metric:.4f}")
        return True

    def save_checkpoint(self, filename: str, is_best: bool = False):
        save_path = self.checkpoint_dir / filename
        tmp_path = self.checkpoint_dir / f"{filename}.tmp"

        state = {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "history": self.history,
            "config": self.config
        }

        torch.save(state, tmp_path)
        if tmp_path.exists():
            os.replace(tmp_path, save_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)

        try:
            with open(self.log_file, "w") as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            print(f"Ошибка сохранения JSON лога: {e}")

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]", leave=False)

        for i, (images, targets) in enumerate(pbar):
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [{k: v.to(self.device, non_blocking=True) for k,v in t.items()} for t in targets]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()

            if not np.isfinite(loss_value):
                print(f"\n[Warning] Infinite Loss ({loss_value}). Skip batch.")
                continue

            self.scaler.scale(losses).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss_value
            steps += 1

            pbar.set_postfix({"loss": f"{loss_value:.3f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.1e}"})

        return total_loss / max(1, steps)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        torch.cuda.empty_cache()

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]", leave=False)

        for images, targets in pbar:
            images = [img.to(self.device, non_blocking=True) for img in images]

            with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                outputs = self.model(images)

            preds = [{k: v.cpu() for k,v in t.items()} for t in outputs]
            targets_cpu = [{k: v.cpu() for k,v in t.items()} for t in targets]

            metric.update(preds, targets_cpu)

        try:
            result = metric.compute()
            return {
                "map": result["map"].item(),
                "map_50": result["map_50"].item(),
                "map_75": result["map_75"].item(),
                "map_small": result["map_small"].item(),
                "map_medium": result["map_medium"].item(),
                "map_large": result["map_large"].item()
            }
        except Exception as e:
            print(f"[Val Warning] Ошибка подсчета метрик: {e}")
            return {"map": 0.0, "map_50": 0.0, "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0}

    def run(self):
        print(f"Старт: Эпохи {self.current_epoch+1}-{self.config['num_epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Device: {self.device}")

        start_time = datetime.now()

        try:
            for epoch in range(self.current_epoch, self.config["num_epochs"]):

                train_loss = self.train_one_epoch()

                val_stats = self.validate()

                if self.scheduler:
                    self.scheduler.step()

                self.history["train_loss"].append(train_loss)
                self.history["val_map"].append(val_stats.get("map", 0))
                self.history["val_map_50"].append(val_stats.get("map_50", 0))
                self.history["val_map_s"].append(val_stats.get("map_small", 0))
                self.history["val_map_m"].append(val_stats.get("map_medium", 0))
                self.history["val_map_l"].append(val_stats.get("map_large", 0))
                self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

                print(
                    f"\n[Epoch {epoch+1}/{self.config['num_epochs']}] "
                    f"Loss: {train_loss:.4f} | "
                    f"mAP: {val_stats.get('map', 0):.4f} | "
                    f"mAP50: {val_stats.get('map_50', 0):.4f} | "
                    f"mAP_s: {val_stats.get('map_small', 0):.4f} | "
                    f"Time: {datetime.now() - start_time}"
                )

                current_map = val_stats.get("map", 0)
                is_best = current_map > self.best_metric
                if is_best:
                    self.best_metric = current_map
                    print(f"Новый рекорд! (mAP: {self.best_metric:.4f})")

                self.current_epoch += 1
                self.save_checkpoint("last_model.pth", is_best=is_best)

                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nОстановка пользователем. Сохраняем состояние...")
            self.save_checkpoint("emergency_stop.pth")

        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            self.save_checkpoint("error_stop.pth")
            raise e

        print(f"\nГотово. Общее время: {datetime.now() - start_time}. Лучший mAP: {self.best_metric:.4f}")