import argparse
import gc
import torch
import torch.nn as nn
import traceback
from typing import Dict, Any, Type, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.dataset import VisDroneDataset, get_transforms, collate_fn
from src.engine import Trainer
from src.utils import seed_everything
from src.models import get_model

def build_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1 or any(nd in name for nd in ["bias", "bn", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.SGD(
        optim_groups,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        nesterov=True
    )

    warmup_epochs = config["warmup_epochs"]
    main_epochs = config["num_epochs"] - warmup_epochs

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=main_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    return optimizer, scheduler

def run_experiment(model_name: str, train_loader, val_loader, args):
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    
    ckpt_dir = CONFIG["checkpoint_dir"] / safe_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_config = CONFIG.copy()
    current_config["checkpoint_dir"] = ckpt_dir
    if args.epochs:
        current_config["num_epochs"] = args.epochs

    print(f"Запуск эксперимента: {model_name}")
    print(f"Папка чекпоинтов: {ckpt_dir}")

    model = None
    optimizer = None
    scheduler = None
    trainer = None

    try:
        model = get_model(safe_name, current_config)
        model.to(current_config["device"])

        optimizer, scheduler = build_optimizer_and_scheduler(model, current_config)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=current_config,
            device=current_config["device"]
        )

        trainer.run()
        
        print(f"Эксперимент '{model_name}' успешно завершен.")

    except KeyboardInterrupt:
        print(f"\nОбучение прервано пользователем.")
    except Exception as e:
        print(f"\nКритическая ошибка в эксперименте '{model_name}': {e}")
        traceback.print_exc()
    finally:
        del model, optimizer, scheduler, trainer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Память очищена.\n")

def main():
    parser = argparse.ArgumentParser(description="VisDrone Object Detection Training")
    parser.add_argument(
        "--model", 
        type=str, 
        default="all", 
        choices=["all", "retinanet_ref", "retinanet_custom", "faster_rcnn_ref", "faster_rcnn_custom"],
        help="Choose model to train (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers")
    
    args = parser.parse_args()

    seed_everything(CONFIG["seed"])

    if args.batch_size: CONFIG["batch_size"] = args.batch_size
    if args.num_workers: CONFIG["num_workers"] = args.num_workers

    print(f"[Main] Device: {CONFIG['device']}")
    print(f"[Main] Batch Size: {CONFIG['batch_size']}")

    print("[Main] Loading Data...")
    
    train_dataset = VisDroneDataset(
        images_dir=CONFIG['train_images'],
        annotations_dir=CONFIG['train_annotations'],
        split='train',
        transforms=get_transforms(train=True, img_size=CONFIG["img_size"]),
        filter_empty_samples=True,
        min_box_area=CONFIG["min_box_area"]
    )

    val_dataset = VisDroneDataset(
        images_dir=CONFIG['val_images'],
        annotations_dir=CONFIG['val_annotations'],
        split='val',
        transforms=get_transforms(train=False, img_size=CONFIG["img_size"]),
        filter_empty_samples=False,
        min_box_area=CONFIG["min_box_area"]
    )

    use_cuda_pin = (CONFIG["device"].type == 'cuda')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=use_cuda_pin,
        persistent_workers=(CONFIG["num_workers"] > 0),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=use_cuda_pin,
        persistent_workers=(CONFIG["num_workers"] > 0),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None
    )
    
    print(f"[Main] Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if args.model == "all":
        models_to_run = [
            "retinanet_ref",
            "retinanet_custom",
            "faster_rcnn_ref",
            "faster_rcnn_custom"
        ]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        run_experiment(model_name, train_loader, val_loader, args)

if __name__ == "__main__":
    main()