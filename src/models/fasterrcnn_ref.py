import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

class ReferenceFasterRCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"[ReferenceFasterRCNN] Инициализация Reference Faster R-CNN | Классов: {config['num_classes']}")

        norm_layer = nn.BatchNorm2d
        base_model = torchvision.models.resnet50(
            weights=None,
            norm_layer=norm_layer
        )

        backbone_body = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
        )

        return_layers = {'4': '0', '5': '1', '6': '2', '7': '3'}
        in_channels_list = [256, 512, 1024, 2048]

        self.backbone = BackboneWithFPN(
            backbone=backbone_body,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool()
        )

        anchor_generator = AnchorGenerator(
            sizes=config["anchor_sizes"],
            aspect_ratios=config["aspect_ratios"]
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=config["num_classes"],
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,

            min_size=config["img_size"],
            max_size=config["max_img_size"],

            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=2000,
            rpn_post_nms_top_n_test=2000,

            box_detections_per_img=400,
            box_score_thresh=0.05,
            box_nms_thresh=0.5
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)