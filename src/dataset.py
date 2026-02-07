import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

CLASS_MAPPING: Dict[int, int] = {
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8
}

CLASS_NAMES: List[str] = [
    "background", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]

class VisDroneDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        filter_empty_samples: bool = True,
        min_box_area: float = 4.0,
        verbose: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.split = split
        self.transforms = transforms
        self.min_box_area = float(min_box_area)
        self.verbose = bool(verbose)

        if self.split == "train" and self.transforms is None:
            raise ValueError("Тренировочный датасет должен быть инициализирован с аугментациями")

        self.should_filter_empty = (self.split == "train") and filter_empty_samples

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Директория изображений не найдена: {self.images_dir}")

        self.all_images_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not self.all_images_paths:
            raise RuntimeError(f"Изображения не найдены в {self.images_dir}")

        self.filename_to_id = {p.stem: i for i, p in enumerate(self.all_images_paths)}

        self.valid_files = self._build_index()

        if self.verbose:
            print(f"[Dataset] Сплит: {self.split} | Валидных сэмплов: {len(self.valid_files)}")

    def _is_box_valid(self, width: float, height: float, score: int, category_id: int) -> bool:
        if score == 0: 
            return False 
        if category_id not in CLASS_MAPPING: 
            return False
        if width <= 1 or height <= 1: 
            return False
        if (width * height) < self.min_box_area: 
            return False
        return True

    def _read_annotation_file(self, ann_path: Path) -> Tuple[List[List[float]], List[int]]:
        boxes, labels = [], []

        if not ann_path.exists():
            return boxes, labels

        try:
            with ann_path.open("r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8: continue
                    try:
                        x_min, y_min = float(parts[0]), float(parts[1])
                        w, h = float(parts[2]), float(parts[3])
                        score = int(parts[4])
                        category = int(parts[5])

                        if self._is_box_valid(w, h, score, category):
                            boxes.append([x_min, y_min, x_min + w, y_min + h])
                            labels.append(CLASS_MAPPING[category])

                    except ValueError:
                        continue
        except Exception:
            pass

        return boxes, labels

    def _build_index(self) -> List[Path]:
        if not self.should_filter_empty:
            return self.all_images_paths

        valid_paths = []
        for img_path in self.all_images_paths:
            ann_path = self.annotations_dir / f"{img_path.stem}.txt"
            boxes, _ = self._read_annotation_file(ann_path)

            if len(boxes) > 0:
                valid_paths.append(img_path)

        if not valid_paths:
            raise RuntimeError("Датасет пуст после фильтрации! Проверьте пути к аннотациям.")
        return valid_paths

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.valid_files[idx]
        ann_path = self.annotations_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            if self.verbose:
                print(f"[Внимание] Поврежденное изображение: {img_path}. Пропуск.")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes_list, labels_list = self._read_annotation_file(ann_path)

        boxes_np = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
        labels_np = np.asarray(labels_list, dtype=np.int64)

        if self.transforms is not None:
            try:
                transformed = self.transforms(image=image, bboxes=boxes_np, labels=labels_np)
                image_tensor = transformed['image']
                boxes_np = np.asarray(transformed['bboxes'], dtype=np.float32).reshape(-1, 4)
                labels_np = np.asarray(transformed['labels'], dtype=np.int64)
            except Exception as e:
                if self.verbose:
                    print(f"[Ошибка аугментации] {img_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))
        else:
            image_tensor = ToTensorV2()(image=image)["image"]
            image_tensor = image_tensor.float() / 255.0

        boxes_t = torch.as_tensor(boxes_np, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels_np, dtype=torch.int64)
        image_id = torch.tensor([self.filename_to_id[img_path.stem]], dtype=torch.int64)

        if boxes_t.numel() > 0:
            _, h_cur, w_cur = image_tensor.shape
            boxes_t[:, 0].clamp_(0, w_cur)
            boxes_t[:, 1].clamp_(0, h_cur)
            boxes_t[:, 2].clamp_(0, w_cur)
            boxes_t[:, 3].clamp_(0, h_cur)

            box_w = boxes_t[:, 2] - boxes_t[:, 0]
            box_h = boxes_t[:, 3] - boxes_t[:, 1]
            keep = (box_w > 1) & (box_h > 1) & ((box_w * box_h) >= self.min_box_area)

            boxes_t = boxes_t[keep]
            labels_t = labels_t[keep]

        if boxes_t.numel() > 0:
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
            "area": area_t,
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64)
        }

        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transforms(train: bool, img_size: int = 1024) -> A.Compose:
    transforms = []

    if train:
        transforms.extend([
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                position='center'
            ),
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.RandomBrightnessContrast(p=1.0),
            ], p=0.4),
        ])
    else:
        transforms.extend([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                position='center'
            ),
        ])

    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1,
        clip=True
    ))