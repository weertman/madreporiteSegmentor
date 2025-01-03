# resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PvtV2Model, PvtV2Config


class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in/out channels differ, we need a projection for the skip connection
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNetDecoder(nn.Module):
    """
    A decoder that uses a series of residual blocks to refine the features.
    """

    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(current_channels, current_channels))
        self.decoder = nn.Sequential(*blocks)

        # Final output conv
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.decoder(x)
        x = self.final_conv(x)
        return x


def initialize_decoder_weights(module, method="default"):
    """
    Applies a given weight initialization method to all conv/bn in the decoder.
    method: "default" (no change), "kaiming_normal"
    Additional methods can be added as needed.
    """
    if method == "default":
        return  # Do nothing

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if method == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class PVTv2ResNetSegmentationModel(nn.Module):
    def __init__(self,
                 model_name="OpenGVLab/pvt_v2_b0",
                 num_classes=1,
                 pretrained=True,
                 decoder_blocks=5,
                 freeze_backbone=False,
                 decoder_init_method="default"):
        """
        A segmentation model using PVTv2 as a backbone and a ResNet-like decoder.

        Args:
            model_name (str): Hugging Face model name or path of a PVTv2 variant.
            num_classes (int): Number of classes for segmentation.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            decoder_blocks (int): Number of residual blocks in the decoder.
            freeze_backbone (bool): If True, the backbone parameters are frozen (no grad).
            decoder_init_method (str): Initialization method for decoder weights,
                                       "default" or "kaiming_normal" are supported.
        """
        super().__init__()

        if pretrained:
            self.backbone = PvtV2Model.from_pretrained(model_name)
        else:
            config = PvtV2Config()
            self.backbone = PvtV2Model(config)

        encoder_channels = self.backbone.config.hidden_sizes[-1]

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.decoder = ResNetDecoder(in_channels=encoder_channels, out_channels=num_classes, num_blocks=decoder_blocks)

        # Initialize decoder weights if requested
        initialize_decoder_weights(self.decoder, method=decoder_init_method)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state  # [B, C, H', W']

        # Forward through the decoder
        logits = self.decoder(features)  # [B, num_classes, H', W']

        # Upsample to input resolution
        input_size = pixel_values.shape[-2:]  # (H, W)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits


if __name__ == "__main__":
    # Quick test with kaiming_normal initialization
    model = PVTv2ResNetSegmentationModel(
        model_name="OpenGVLab/pvt_v2_b0", #pvt_v2_b3 #https://huggingface.co/docs/transformers/en/model_doc/pvt_v2
        num_classes=1,
        pretrained=True,
        decoder_blocks=3,
        freeze_backbone=True,
        decoder_init_method="kaiming_normal"  # Apply kaiming initialization to decoder
    )
    dummy_input = torch.randn(1, 3, 480, 480)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expect (1, 1, 480, 480)
