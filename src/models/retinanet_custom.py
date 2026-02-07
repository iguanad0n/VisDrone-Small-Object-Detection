import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple
from torchvision.ops import sigmoid_focal_loss, box_iou, batched_nms

from src.models.common import CustomResNet50, CustomFeaturePyramidNetwork, CustomAnchorGenerator

class CustomLastLevelP6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

    def forward(self, results: List[torch.Tensor], x: List[torch.Tensor], names: List[str]):
        c5 = x[-1]
        p6 = self.p6(c5)
        results.append(p6)
        names.append("p6")
        return results, names

class CustomClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, prior_probability: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None: nn.init.constant_(layer.bias, 0)
        
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_cls_logits = self.cls_logits(self.conv(x))
        N, _, H, W = all_cls_logits.shape
        all_cls_logits = all_cls_logits.view(N, -1, self.num_classes, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, self.num_classes)
        return all_cls_logits

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs
        for i in range(len(targets)):
            target_labels = targets[i]["labels"]
            matched_idxs_per_image = matched_idxs[i]
            logits_per_image = cls_logits[i]
            
            foreground_idxs = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs.sum()
            
            gt_classes_target = torch.zeros_like(logits_per_image)
            matched_gt_labels = target_labels[matched_idxs_per_image[foreground_idxs]]
            
            matched_gt_labels = matched_gt_labels - 1 
            
            gt_classes_target[foreground_idxs, matched_gt_labels] = 1.0
            
            valid_idxs = matched_idxs_per_image != -2
            
            loss = sigmoid_focal_loss(logits_per_image[valid_idxs], gt_classes_target[valid_idxs], reduction="sum")
            losses.append(loss / max(1, num_foreground))
            
        return sum(losses) / max(1, len(targets))

class CustomRegressionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bbox_regression = self.bbox_pred(self.conv(x))
        N, _, H, W = bbox_regression.shape
        bbox_regression = bbox_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
        return bbox_regression

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        losses = []
        bbox_regression = head_outputs
        for i in range(len(targets)):
            target_boxes = targets[i]["boxes"]
            matched_idxs_per_image = matched_idxs[i]
            bbox_pred_per_image = bbox_regression[i]
            anchors_per_image = anchors[i]
            
            foreground_idxs = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs.numel()
            
            if num_foreground > 0:
                matched_gt_boxes = target_boxes[matched_idxs_per_image[foreground_idxs]]
                pos_anchors = anchors_per_image[foreground_idxs]
                pos_bbox_pred = bbox_pred_per_image[foreground_idxs]
                
                wx, wy, ww, wh = 1.0, 1.0, 1.0, 1.0
                ex_widths = pos_anchors[:, 2] - pos_anchors[:, 0]
                ex_heights = pos_anchors[:, 3] - pos_anchors[:, 1]
                ex_ctr_x = pos_anchors[:, 0] + 0.5 * ex_widths
                ex_ctr_y = pos_anchors[:, 1] + 0.5 * ex_heights
                
                gt_widths = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
                gt_heights = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
                gt_ctr_x = matched_gt_boxes[:, 0] + 0.5 * gt_widths
                gt_ctr_y = matched_gt_boxes[:, 1] + 0.5 * gt_heights
                
                targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
                targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
                targets_dw = ww * torch.log(gt_widths / ex_widths)
                targets_dh = wh * torch.log(gt_heights / ex_heights)
                
                target_deltas = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
                
                loss = F.smooth_l1_loss(pos_bbox_pred, target_deltas, beta=1.0, reduction="sum")
                losses.append(loss / max(1, num_foreground))
            else:
                losses.append(torch.tensor(0.0, device=bbox_regression.device))
                
        return sum(losses) / max(1, len(targets))

class CustomRetinaNetHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.classification_head = CustomClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = CustomRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_logits = []
        bbox_regression = []
        for feature in x:
            cls_logits.append(self.classification_head(feature))
            bbox_regression.append(self.regression_head(feature))
        return {
            "cls_logits": torch.cat(cls_logits, dim=1), 
            "bbox_regression": torch.cat(bbox_regression, dim=1)
        }

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs["cls_logits"], matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs["bbox_regression"], anchors, matched_idxs)
        }

class CustomRetinaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"[CustomRetinaNet] Инициализация Custom RetinaNet | Классов: {config['num_classes']}")

        num_classes = config["num_classes"]

        self.backbone = CustomResNet50(
            layers=[3, 4, 6, 3],
            norm_layer=nn.BatchNorm2d
        )

        self.neck = CustomFeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=CustomLastLevelP6(2048, 256)
        )

        self.anchor_generator = CustomAnchorGenerator(
            sizes=config["anchor_sizes"],
            aspect_ratios=config["aspect_ratios"]
        )

        num_anchors = len(config["anchor_sizes"][0]) * len(config["aspect_ratios"][0])
        self.head = CustomRetinaNetHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes - 1
        )

        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.detections_per_img = 500
        self.fg_iou_thresh = 0.5
        self.bg_iou_thresh = 0.4

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if isinstance(images, (list, tuple)):
            images_tensor = torch.stack(images).to(next(self.backbone.parameters()).device)
        else:
            images_tensor = images

        backbone_features = self.backbone(images_tensor)
        fpn_features = self.neck(backbone_features)
        fpn_features_list = list(fpn_features.values())

        anchors = self.anchor_generator(images_tensor, fpn_features_list)
        head_outputs = self.head(fpn_features_list)

        if self.training:
            if targets is None: raise ValueError("Targets required for training")
            matched_idxs = []
            
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device))
                    continue

                iou_matrix = box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_val, matches = iou_matrix.max(dim=0)
                match_indices = torch.full_like(matches, -1)

                positive_mask = matched_val >= self.fg_iou_thresh
                match_indices[positive_mask] = matches[positive_mask]

                ignore_mask = (matched_val > self.bg_iou_thresh) & (matched_val < self.fg_iou_thresh)
                match_indices[ignore_mask] = -2

                matched_idxs.append(match_indices)

            return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        
        else:
            return self.postprocess(head_outputs, anchors, images_tensor.shape[-2:])

    def postprocess(self, head_outputs, anchors, image_shape):
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        detections = []
        
        for i in range(len(anchors)):
            logits = cls_logits[i]
            bbox_deltas = bbox_regression[i]
            anchors_per_image = anchors[i]
            
            scores = torch.sigmoid(logits).flatten()
            num_anchors = logits.shape[0]
            num_classes = logits.shape[1]

            keep_idxs = scores > self.score_thresh
            scores = scores[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            if scores.numel() > 1000:
                scores, idxs = scores.sort(descending=True)
                scores = scores[:1000]
                topk_idxs = topk_idxs[idxs[:1000]]

            anchor_idxs = topk_idxs // num_classes
            labels = topk_idxs % num_classes

            cur_anchors = anchors_per_image[anchor_idxs]
            cur_deltas = bbox_deltas[anchor_idxs]

            widths = cur_anchors[:, 2] - cur_anchors[:, 0]
            heights = cur_anchors[:, 3] - cur_anchors[:, 1]
            ctr_x = cur_anchors[:, 0] + 0.5 * widths
            ctr_y = cur_anchors[:, 1] + 0.5 * heights

            dx = cur_deltas[:, 0]
            dy = cur_deltas[:, 1]
            dw = cur_deltas[:, 2]
            dh = cur_deltas[:, 3]

            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w = torch.exp(dw) * widths
            pred_h = torch.exp(dh) * heights

            pred_boxes = torch.stack([
                pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h
            ], dim=1)

            pred_boxes[:, 0::2].clamp_(min=0, max=image_shape[1])
            pred_boxes[:, 1::2].clamp_(min=0, max=image_shape[0])

            keep = batched_nms(pred_boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            
            detections.append({
                "boxes": pred_boxes[keep], 
                "scores": scores[keep], 
                "labels": labels[keep] + 1
            })
        return detections