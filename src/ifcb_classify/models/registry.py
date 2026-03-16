from dataclasses import dataclass
from typing import Any

import torchvision.models as tv


@dataclass(frozen=True)
class ModelSpec:
    constructor: Any
    head_path: str
    in_features: int
    bias: bool = True
    weights: str = "DEFAULT"


# head_path uses dot notation for attributes and [n] for Sequential indices.
# Examples: "fc" -> model.fc, "classifier[6]" -> model.classifier[6], "heads[0]" -> model.heads[0]

MODELS: dict[str, ModelSpec] = {
    # AlexNet
    "alexnet": ModelSpec(tv.alexnet, "classifier[6]", 4096),
    # ConvNeXt
    "convnext_tiny": ModelSpec(tv.convnext_tiny, "classifier[2]", 768),
    "convnext_small": ModelSpec(tv.convnext_small, "classifier[2]", 768),
    "convnext_base": ModelSpec(tv.convnext_base, "classifier[2]", 1024),
    "convnext_large": ModelSpec(tv.convnext_large, "classifier[2]", 1536),
    # DenseNet
    "densenet121": ModelSpec(tv.densenet121, "classifier", 1024),
    "densenet169": ModelSpec(tv.densenet169, "classifier", 1664),
    "densenet161": ModelSpec(tv.densenet161, "classifier", 2208),
    "densenet201": ModelSpec(tv.densenet201, "classifier", 1920),
    # EfficientNetV2
    "efficientnet_v2_s": ModelSpec(tv.efficientnet_v2_s, "classifier[1]", 1280),
    "efficientnet_v2_m": ModelSpec(tv.efficientnet_v2_m, "classifier[1]", 1280),
    "efficientnet_v2_l": ModelSpec(tv.efficientnet_v2_l, "classifier[1]", 1280),
    # GoogLeNet
    "googlenet": ModelSpec(tv.googlenet, "fc", 1024),
    # Inception
    "inception_v3": ModelSpec(tv.inception_v3, "fc", 2048),
    "inception_v3_untrained": ModelSpec(tv.inception_v3, "fc", 2048, weights=None),
    # MNASNet
    "mnasnet0_5": ModelSpec(tv.mnasnet0_5, "classifier[1]", 1280),
    "mnasnet0_75": ModelSpec(tv.mnasnet0_75, "classifier[1]", 1280),
    "mnasnet1_0": ModelSpec(tv.mnasnet1_0, "classifier[1]", 1280),
    "mnasnet1_3": ModelSpec(tv.mnasnet1_3, "classifier[1]", 1280),
    # MaxVit
    "maxvit_t": ModelSpec(tv.maxvit_t, "classifier[5]", 512, bias=False),
    # MobileNetV3
    "mobilenet_v3_large": ModelSpec(tv.mobilenet_v3_large, "classifier[3]", 1280),
    "mobilenet_v3_small": ModelSpec(tv.mobilenet_v3_small, "classifier[3]", 1024),
    # ResNet
    "resnet18": ModelSpec(tv.resnet18, "fc", 512),
    "resnet34": ModelSpec(tv.resnet34, "fc", 512),
    "resnet50": ModelSpec(tv.resnet50, "fc", 2048),
    "resnet101": ModelSpec(tv.resnet101, "fc", 2048),
    "resnet152": ModelSpec(tv.resnet152, "fc", 2048),
    # ResNeXt
    "resnext50_32x4d": ModelSpec(tv.resnext50_32x4d, "fc", 2048),
    "resnext101_32x8d": ModelSpec(tv.resnext101_32x8d, "fc", 2048),
    "resnext101_64x4d": ModelSpec(tv.resnext101_64x4d, "fc", 2048),
    # ShuffleNetV2
    "shufflenet_v2_x0_5": ModelSpec(tv.shufflenet_v2_x0_5, "fc", 1024),
    "shufflenet_v2_x1_0": ModelSpec(tv.shufflenet_v2_x1_0, "fc", 1024),
    "shufflenet_v2_x1_5": ModelSpec(tv.shufflenet_v2_x1_5, "fc", 1024),
    "shufflenet_v2_x2_0": ModelSpec(tv.shufflenet_v2_x2_0, "fc", 2048),
    # Swin Transformer V2
    "swin_v2_t": ModelSpec(tv.swin_v2_t, "head", 768),
    "swin_v2_s": ModelSpec(tv.swin_v2_s, "head", 768),
    "swin_v2_b": ModelSpec(tv.swin_v2_b, "head", 1024),
    # Vision Transformer
    "vit_b_16": ModelSpec(tv.vit_b_16, "heads[0]", 768),
    "vit_b_32": ModelSpec(tv.vit_b_32, "heads[0]", 768),
    "vit_l_16": ModelSpec(tv.vit_l_16, "heads[0]", 1024),
    "vit_l_32": ModelSpec(tv.vit_l_32, "heads[0]", 1024),
    "vit_h_14": ModelSpec(tv.vit_h_14, "heads[0]", 1280),
    # VGG
    "vgg11": ModelSpec(tv.vgg11, "classifier[6]", 4096),
    "vgg11_bn": ModelSpec(tv.vgg11_bn, "classifier[6]", 4096),
    "vgg13": ModelSpec(tv.vgg13, "classifier[6]", 4096),
    "vgg13_bn": ModelSpec(tv.vgg13_bn, "classifier[6]", 4096),
    "vgg16": ModelSpec(tv.vgg16, "classifier[6]", 4096),
    "vgg16_bn": ModelSpec(tv.vgg16_bn, "classifier[6]", 4096),
    "vgg19": ModelSpec(tv.vgg19, "classifier[6]", 4096),
    "vgg19_bn": ModelSpec(tv.vgg19_bn, "classifier[6]", 4096),
    # Wide ResNet
    "wide_resnet50_2": ModelSpec(tv.wide_resnet50_2, "fc", 2048),
    "wide_resnet101_2": ModelSpec(tv.wide_resnet101_2, "fc", 2048),
}
