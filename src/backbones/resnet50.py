# src/backbones/resnet50.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 model architecture (Backbone), CNN base.
    Output: feature map from the last convolutional block (layer4).
    For use as a feature extractor in more complex models.
    """
    def __init__(self, num_input_channels: int = 3, pretrained: bool = True):
        """
        Initialization

        Args:
            num_input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            pretrained (bool): Whether to load ImageNet-pretrained weights.
        """
        super(ResNet50Backbone, self).__init__()

        # Load the standard ResNet50
        backbone = models.resnet50(pretrained=pretrained)

        # If input is not RGB, replace the first conv to accommodate different channels
        if num_input_channels != 3:
            # Keep the same conv1 settings except in_channels
            backbone.conv1 = nn.Conv2d(
                num_input_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=backbone.conv1.bias is not None
            )

        # Layer 1: Conv1 -> BatchNorm -> ReLU -> MaxPool
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool

        # Layer 2–5: Residual blocks
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Note: We omit the avgpool and fc layers from ResNet50.
        # These can be added later for classification heads if needed.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ResNet50 backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where N is batch size, C is num_input_channels,
                              H is height, and W is width.

        Returns:
            torch.Tensor: Output feature map from the last residual block.
                          The shape will be (N, 2048, H', W'), where H' and W'
                          depend on the input image size. For a 224x224 input,
                          it's typically (N, 2048, 7, 7).
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch_size, channels, height, width)
    # Common input size for ResNet50 is 224x224.
    dummy_input = torch.randn(1, 3, 224, 224)

    # Instantiate the backbone
    resnet_features = ResNet50Backbone(num_input_channels=3, pretrained=True)
    print("ResNet50 Backbone Architecture:")
    print(resnet_features)

    # Perform a forward pass
    print(f"\nInput shape: {dummy_input.shape}")
    output_features = resnet_features(dummy_input)
    print(f"Output feature map shape: {output_features.shape}")

    # Expected output shape for 224x224 input: (1, 2048, 7, 7)
    assert output_features.shape == (1, 2048, 7, 7), "Output shape mismatch!"
