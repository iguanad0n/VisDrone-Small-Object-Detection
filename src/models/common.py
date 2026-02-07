import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Optional

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CustomBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class CustomResNet50(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1; self.groups = 1; self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(CustomBottleneck, 64, layers[0])
        self.layer2 = self._make_layer(CustomBottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(CustomBottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(CustomBottleneck, 512, layers[3], stride=2)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer; downsample = None; previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3) 
        c5 = self.layer4(c4)

        return OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)])

class CustomFeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256, extra_blocks: Optional[nn.Module] = None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0: raise ValueError("in_channels=0 not supported")
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def forward(self, x: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        x_list = list(x.values())

        last_inner = self.inner_blocks[-1](x_list[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(x_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x_list[idx])
            feat_shape = inner_lateral.shape[-2:]
            
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            
            last_inner = inner_lateral + inner_top_down
            
            results.insert(0, self.layer_blocks[idx](last_inner))

        if self.extra_blocks is not None:
            names = list(x.keys())
            results, _ = self.extra_blocks(results, x_list, names)

        return OrderedDict([(str(i), v) for i, v in enumerate(results)])

class CustomAnchorGenerator(nn.Module):
    def __init__(self, sizes: Tuple[Tuple[int, ...], ...], aspect_ratios: Tuple[Tuple[float, ...], ...]):
        super().__init__()
        if len(sizes) != len(aspect_ratios):
            raise ValueError(f"Sizes ({len(sizes)}) and aspect_ratios ({len(aspect_ratios)}) mismatch.")

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [self._generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)]

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def _generate_anchors(self, scales, aspect_ratios):
        scales_t = torch.as_tensor(scales, dtype=torch.float32)
        aspect_ratios_t = torch.as_tensor(aspect_ratios, dtype=torch.float32)
        h_ratios = torch.sqrt(aspect_ratios_t)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales_t[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales_t[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def _grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            shifted_anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            anchors.append(shifted_anchors)
        return anchors

    def forward(self, images, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(images, (list, tuple)):
             pass 
        
        image_size = images.image_sizes[0] if hasattr(images, 'image_sizes') else images.shape[-2:]
        
        grid_sizes = [f.shape[-2:] for f in feature_maps]
        device = feature_maps[0].device
        dtype = feature_maps[0].dtype

        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        
        self.cell_anchors = [anchor.to(device=device, dtype=dtype) for anchor in self.cell_anchors]
        
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes, strides)
        
        anchors = []
        batch_size = images.tensors.shape[0] if hasattr(images, 'tensors') else 1
        
        anchors_in_image = torch.cat(anchors_over_all_feature_maps)
        
        for _ in range(batch_size):
            anchors.append(anchors_in_image)
            
        return anchors