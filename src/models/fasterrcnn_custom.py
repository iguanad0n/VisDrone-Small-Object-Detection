import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
from typing import List, Tuple, Dict, Optional

from src.models.common import CustomResNet50, CustomFeaturePyramidNetwork, CustomAnchorGenerator

class BoxCoder:
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.weights = weights

    def encode(self, reference_boxes, proposals):
        wx, wy, ww, wh = self.weights

        proposals_x1 = proposals[:, 0]
        proposals_y1 = proposals[:, 1]
        proposals_x2 = proposals[:, 2]
        proposals_y2 = proposals[:, 3]

        reference_boxes_x1 = reference_boxes[:, 0]
        reference_boxes_y1 = reference_boxes[:, 1]
        reference_boxes_x2 = reference_boxes[:, 2]
        reference_boxes_y2 = reference_boxes[:, 3]

        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    def decode(self, rel_codes, boxes):
        wx, wy, ww, wh = self.weights

        boxes = boxes.to(rel_codes.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes4 = pred_ctr_y + 0.5 * pred_h

        return torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=match_quality_matrix.device)

        matched_vals, matches = match_quality_matrix.max(dim=0)

        matches_idx = torch.full_like(matches, -1)

        positive = matched_vals >= self.high_threshold
        matches_idx[positive] = matches[positive]

        negative = matched_vals < self.low_threshold
        matches_idx[negative] = -2

        if self.allow_low_quality_matches:
            highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
            gt_pred_pairs_of_highest_quality = torch.where(
                match_quality_matrix == highest_quality_foreach_gt[:, None]
            )
            pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
            matches_idx[pred_inds_to_update] = matches[pred_inds_to_update]

        return matches_idx


class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 0)[0]
            negative = torch.where(matched_idxs_per_image == -2)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx.append(positive[perm1])
            neg_idx.append(negative[perm2])

        return pos_idx, neg_idx

class CustomLastLevelMaxPool(nn.Module):
    def forward(self, results: List[torch.Tensor], x: List[torch.Tensor], names: List[str]) -> Tuple[List[torch.Tensor], List[str]]:
        names.append("pool")
        last_inner = results[-1]
        results.append(F.max_pool2d(last_inner, 1, 2, 0))
        return results, names

class CustomRPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None: nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

class CustomRegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def forward(self, images, features, targets=None):
        anchors = self.anchor_generator(images, features)
        objectness, pred_bbox_deltas = self.head(features)

        num_images = len(anchors)
        objectness_flat = []
        pred_bbox_deltas_flat = []
        for i in range(len(objectness)):
            objectness_flat.append(objectness[i].permute(0, 2, 3, 1).flatten(1))
            pred_bbox_deltas_flat.append(pred_bbox_deltas[i].permute(0, 2, 3, 1).reshape(num_images, -1, 4))

        objectness = torch.cat(objectness_flat, dim=1)
        pred_bbox_deltas = torch.cat(pred_bbox_deltas_flat, dim=1)

        proposals = self._decode_proposals(anchors, pred_bbox_deltas, objectness, images.shape[-2:])
        
        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("Targets required for training")
            losses = self._compute_loss(objectness, pred_bbox_deltas, anchors, targets)
            
        return proposals, losses

    def _decode_proposals(self, anchors, pred_bbox_deltas, objectness, image_shape):
        proposals = []
        for i in range(len(anchors)):
            pre_nms_top_n = min(self.pre_nms_top_n, objectness.shape[1])
            scores, top_k_idx = objectness[i].topk(pre_nms_top_n)
            box_deltas = pred_bbox_deltas[i][top_k_idx]
            img_anchors = anchors[i][top_k_idx]
            
            decoded_boxes = self.box_coder.decode(box_deltas, img_anchors)
            decoded_boxes[:, 0::2].clamp_(min=0, max=image_shape[1])
            decoded_boxes[:, 1::2].clamp_(min=0, max=image_shape[0])
            
            keep = ops.remove_small_boxes(decoded_boxes, self.min_size)
            decoded_boxes, scores = decoded_boxes[keep], scores[keep]
            
            keep = ops.nms(decoded_boxes, scores, self.nms_thresh)
            keep = keep[:self.post_nms_top_n]
            proposals.append(decoded_boxes[keep])
            
        return proposals

    def _compute_loss(self, objectness, pred_bbox_deltas, anchors, targets):
        labels = []
        regression_targets = []
        for i in range(len(anchors)):
            gt_boxes = targets[i]["boxes"]
            if gt_boxes.numel() == 0:
                labels.append(torch.full((anchors[i].shape[0],), -2, dtype=torch.int64, device=anchors[i].device))
                regression_targets.append(torch.zeros_like(anchors[i]))
                continue
            
            match_quality_matrix = ops.box_iou(gt_boxes, anchors[i])
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            labels.append(matched_idxs)
            
            matched_gt_boxes = gt_boxes[matched_idxs.clamp(min=0)]
            regression_targets.append(self.box_coder.encode(matched_gt_boxes, anchors[i]))

        pos_idx_list, neg_idx_list = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.cat([torch.isin(torch.arange(l.numel(), device=l.device), p) for l, p in zip(labels, pos_idx_list)])
        sampled_neg_inds = torch.cat([torch.isin(torch.arange(l.numel(), device=l.device), n) for l, n in zip(labels, neg_idx_list)])
        sampled_inds = torch.where(sampled_pos_inds | sampled_neg_inds)[0]

        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pred_bbox_deltas = pred_bbox_deltas.reshape(-1, 4)
        
        final_labels = torch.zeros_like(labels, dtype=torch.float32)
        final_labels[labels >= 0] = 1.0

        loss_objectness = F.binary_cross_entropy_with_logits(objectness[sampled_inds], final_labels[sampled_inds])
        
        pos_inds_flattened = torch.where(sampled_pos_inds)[0]
        loss_box_reg = F.smooth_l1_loss(
            pred_bbox_deltas[pos_inds_flattened], 
            regression_targets[pos_inds_flattened], 
            beta=1.0 / 9, 
            reduction="sum"
        ) / (sampled_inds.numel() + 1e-6)
        
        return {"loss_rpn_objectness": loss_objectness, "loss_rpn_box_reg": loss_box_reg}

class CustomRoIHeads(nn.Module):
    def __init__(self, in_channels, num_classes, fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25, score_thresh=0.05, nms_thresh=0.5, detections_per_img=500):
        super().__init__()
        self.roi_pool = ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        resolution = self.roi_pool.output_size[0]
        representation_size = 1024
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels * resolution ** 2, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, std=0.01)
                if "bbox_pred" in name: nn.init.normal_(param, std=0.001)
            if "bias" in name: nn.init.constant_(param, 0)

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self._select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.roi_pool(features, proposals, image_shapes)
        box_features = self.head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)

        result = []
        losses = {}
        if self.training:
            losses = self._compute_loss(class_logits, box_regression, labels, regression_targets)
        else:
            result = self._postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        
        return result, losses

    def _select_training_samples(self, proposals, targets):
        proposals_with_gt = []
        for p, t in zip(proposals, targets):
            if t["boxes"].numel() > 0:
                proposals_with_gt.append(torch.cat([p, t["boxes"]], dim=0))
            else:
                proposals_with_gt.append(p)

        labels = []
        matched_idxs = []
        regression_targets = []
        
        for i, (props, target) in enumerate(zip(proposals_with_gt, targets)):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            
            if gt_boxes.numel() == 0:
                device = props.device
                labels.append(torch.zeros(props.shape[0], dtype=torch.int64, device=device))
                matched_idxs.append(torch.full((props.shape[0],), -1, dtype=torch.int64, device=device))
                regression_targets.append(torch.zeros_like(props))
                continue
                
            match_quality_matrix = ops.box_iou(gt_boxes, props)
            idxs = self.proposal_matcher(match_quality_matrix)
            
            labels_in_image = gt_labels[idxs.clamp(min=0)]
            labels_in_image[idxs < 0] = 0
            
            matched_gt_boxes = gt_boxes[idxs.clamp(min=0)]
            reg_targets = self.box_coder.encode(matched_gt_boxes, props)
            
            labels.append(labels_in_image)
            matched_idxs.append(idxs)
            regression_targets.append(reg_targets)

        pos_idx, neg_idx = self.fg_bg_sampler(matched_idxs)
        sampled_proposals = []
        sampled_labels = []
        sampled_targets = []
        
        for i in range(len(proposals_with_gt)):
            idx = torch.cat([pos_idx[i], neg_idx[i]])
            sampled_proposals.append(proposals_with_gt[i][idx])
            sampled_labels.append(labels[i][idx])
            sampled_targets.append(regression_targets[i][idx])
            
        return sampled_proposals, matched_idxs, sampled_labels, sampled_targets

    def _compute_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        loss_cls = F.cross_entropy(class_logits, labels)
        
        pos_inds = torch.where(labels > 0)[0]
        if pos_inds.numel() > 0:
            labels_pos = labels[pos_inds]
            box_regression = box_regression.reshape(box_regression.shape[0], -1, 4)
            box_regression_pos = box_regression[pos_inds, labels_pos]
            loss_box_reg = F.smooth_l1_loss(
                box_regression_pos, 
                regression_targets[pos_inds], 
                beta=1.0, 
                reduction="sum"
            ) / labels.numel()
        else:
            loss_box_reg = torch.tensor(0.0, device=class_logits.device)
            
        return {"loss_roi_classifier": loss_cls, "loss_roi_box_reg": loss_box_reg}

    def _postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        probs = F.softmax(class_logits, dim=-1)
        
        proposals_flat = torch.cat(proposals, dim=0)
        box_regression = box_regression.reshape(box_regression.shape[0], -1, 4)
        
        proposals_expanded = proposals_flat.unsqueeze(1).expand(-1, num_classes, -1).reshape(-1, 4)
        box_regression_flat = box_regression.reshape(-1, 4)
        
        decoded_boxes = self.box_coder.decode(box_regression_flat, proposals_expanded)
        decoded_boxes = decoded_boxes.reshape(-1, num_classes, 4)

        result = []
        boxes_start_idx = 0
        for i, shape in enumerate(image_shapes):
            num_boxes = proposals[i].shape[0]
            boxes = decoded_boxes[boxes_start_idx : boxes_start_idx + num_boxes]
            scores = probs[boxes_start_idx : boxes_start_idx + num_boxes]
            boxes_start_idx += num_boxes
            
            boxes[:, :, 0::2].clamp_(min=0, max=shape[1])
            boxes[:, :, 1::2].clamp_(min=0, max=shape[0])
            
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = torch.arange(1, num_classes, device=device).view(1, -1).expand(num_boxes, -1).reshape(-1)
            
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            
            keep = ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            
            result.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
            
        return result

class CustomFasterRCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"[CustomFasterRCNN] Инициализация Custom Faster R-CNN | Классов: {config['num_classes']}")

        self.backbone = CustomResNet50(layers=[3, 4, 6, 3], norm_layer=nn.BatchNorm2d)

        self.neck = CustomFeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=CustomLastLevelMaxPool()
        )

        self.anchor_generator = CustomAnchorGenerator(
            sizes=config["anchor_sizes"],
            aspect_ratios=config["aspect_ratios"]
        )

        num_anchors = len(config["anchor_sizes"][0]) * len(config["aspect_ratios"][0])

        rpn_head = CustomRPNHead(in_channels=256, num_anchors=num_anchors)
        self.rpn = CustomRegionProposalNetwork(
            self.anchor_generator, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=2000, post_nms_top_n=2000, nms_thresh=0.7
        )

        self.roi_heads = CustomRoIHeads(
            in_channels=256,
            num_classes=config["num_classes"],
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            score_thresh=0.05, nms_thresh=0.5,
            detections_per_img=400
        )

    def forward(self, images, targets=None):
        if isinstance(images, (list, tuple)):
            device = next(self.backbone.parameters()).device
            images = torch.stack(images).to(device)

        features = self.backbone(images)
        features = self.neck(features)

        features_list = list(features.values())
        proposals, rpn_losses = self.rpn(images, features_list, targets)

        image_shapes = [images.shape[-2:]] * images.shape[0]
        detections, roi_losses = self.roi_heads(features, proposals, image_shapes, targets)

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)

        return losses if self.training else detections