import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PvtV2Model, PvtV2Config


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


class UNetDecoderBlock(nn.Module):
    """
    A single decoder block that:
    - Upsamples the input feature map
    - Concatenates with the corresponding encoder feature map
    - Applies convolutional layers to refine
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Upsample 'x' to match 'skip' spatial size
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ProgressiveUNetDecoder(nn.Module):
    """
    A U-Net-like decoder that progressively upsamples from the deepest feature map
    and fuses with higher-resolution encoder features at each stage.
    """

    def __init__(self, encoder_channels, decoder_channels, num_classes=1):
        """
        Args:
            encoder_channels (list[int]): Number of channels for each encoder feature map,
                                          ordered from shallowest (highest resolution) to
                                          deepest (lowest resolution).
                                          E.g., encoder_channels = [C1, C2, C3, C4] where C1
                                          is the earliest stage (high-res) and C4 is the deepest.
            decoder_channels (list[int]): Number of channels at each decoder stage.
                                          Should have length = len(encoder_channels)-1.
            num_classes (int): Number of output classes.
        """
        super().__init__()

        # We decode from deepest to shallowest:
        # features: [x1, x2, x3, x4] (x1 is shallowest, x4 is deepest)
        # We do x4 -> x3 -> x2 -> x1
        self.num_stages = len(encoder_channels) - 1
        self.blocks = nn.ModuleList()

        in_ch = encoder_channels[-1]  # Start from deepest (C4)
        for i in range(self.num_stages):
            skip_ch = encoder_channels[-(i + 2)]  # Fuse with next shallower encoder stage
            out_ch = decoder_channels[i]
            self.blocks.append(UNetDecoderBlock(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: list of encoder features [x1, x2, x3, x4] where
                      x1 is shallowest (high-res) and x4 is deepest (low-res).
        """
        x = features[-1]  # x4
        # Decode through stages:
        for i, block in enumerate(self.blocks):
            skip = features[-(i + 2)]
            x = block(x, skip)

        x = self.final_conv(x)
        return x


class PVTv2UNetSegmentationModel(nn.Module):
    def __init__(self,
                 model_name="OpenGVLab/pvt_v2_b0",
                 num_classes=1,
                 pretrained=True,
                 freeze_backbone=False,
                 decoder_init_method="default",
                 input_shape=None):
        """
        A segmentation model using PVTv2 as a backbone and a U-Net-like decoder.

        Args:
            model_name (str): Hugging Face model name or path of a PVTv2 variant.
            num_classes (int): Number of classes for segmentation.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            freeze_backbone (bool): If True, the backbone parameters are frozen (no grad).
            decoder_init_method (str): Initialization method for decoder weights,
                                       "default" or "kaiming_normal".
            input_shape (tuple or None): (H, W) shape. If provided, final output will match this shape.
                                         If None, output shape will match input size of pixel_values.
        """
        super().__init__()

        self.input_shape = input_shape

        if pretrained:
            config = PvtV2Config.from_pretrained(model_name)
            config.output_hidden_states = True
            self.backbone = PvtV2Model.from_pretrained(model_name, config=config)
        else:
            config = PvtV2Config()
            config.output_hidden_states = True
            self.backbone = PvtV2Model(config)

        # For PVTv2 b0: hidden_sizes = [64, 128, 320, 512]
        encoder_channels = self.backbone.config.hidden_sizes
        decoder_channels = [256, 128, 64]  # Example configuration

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.decoder = ProgressiveUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )

        # Initialize decoder weights if requested
        initialize_decoder_weights(self.decoder, method=decoder_init_method)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        # hidden_states: [x1, x2, x3, x4]
        features = outputs.hidden_states

        logits = self.decoder(features)

        # Determine final upsample size
        if self.input_shape is not None:
            final_size = self.input_shape
        else:
            final_size = pixel_values.shape[-2:]  # match input image size

        logits = F.interpolate(logits, size=final_size, mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    # Quick test: specify input_shape so the final output always matches (480,480)
    model = PVTv2UNetSegmentationModel(
        model_name="OpenGVLab/pvt_v2_b0",
        num_classes=1,
        pretrained=True,
        freeze_backbone=True,
        decoder_init_method="kaiming_normal",
        input_shape=(480, 480)
    )
    dummy_input = torch.randn(1, 3, 480, 480)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expect (1, 1, 480, 480)
