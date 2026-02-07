import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

class ReferenceRetinaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"[ReferenceRetinaNet] Инициализация RetinaNet | Классов: {config['num_classes']}")

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
            extra_blocks=LastLevelP6P7(256, 256)
        )

        anchor_generator = AnchorGenerator(
            sizes=config["anchor_sizes"],
            aspect_ratios=config["aspect_ratios"]
        )

        head = RetinaNetHead(
            in_channels=self.backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            num_classes=config["num_classes"],
            norm_layer=norm_layer
        )

        self.model = RetinaNet(
            backbone=self.backbone,
            num_classes=config["num_classes"],
            anchor_generator=anchor_generator,
            head=head,
            min_size=config["img_size"],
            max_size=config["max_img_size"],
            
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=400,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.4
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)