# src/backbones/alexnet.py
import torch
import torch.nn as nn

class AlexNetBackbone(nn.Module):
    """
    AlexNet model architecture (Backbone), CNN base.
    Output: feature map from  last MaxPool2d.
    For use as a feature extractor in more complex models.
    """
    def __init__(self, num_input_channels: int = 3, use_batch_norm: bool = True):
        """
        Initialization

        Args:
            num_input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            use_batch_norm (bool): Whether to use Batch Normalization after convolutional layers.
                                   The original AlexNet used Local Response Normalization (LRN),
                                   but Batch Normalization is generally preferred in modern networks.
        """
        super(AlexNetBackbone, self).__init__()

        # Layer 1: Convolution -> ReLU -> (Optional BatchNorm) -> MaxPool
        self.conv1 = nn.Conv2d(num_input_channels, 96, kernel_size=11, stride=4, padding=2) # Adjusted padding for common output sizes
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(96) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2: Convolution -> ReLU -> (Optional BatchNorm) -> MaxPool
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3: Convolution -> ReLU -> (Optional BatchNorm)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(384) if use_batch_norm else nn.Identity()

        # Layer 4: Convolution -> ReLU -> (Optional BatchNorm)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(384) if use_batch_norm else nn.Identity()

        # Layer 5: Convolution -> ReLU -> (Optional BatchNorm) -> MaxPool
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # The original AlexNet had Local Response Normalization (LRN) layers.
        # LRN is less common now and often replaced by Batch Normalization or omitted.
        # If you need to strictly adhere to the original paper for LRN:
        # self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the AlexNet backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where N is batch size, C is num_input_channels,
                              H is height, and W is width.

        Returns:
            torch.Tensor: Output feature map from the last max-pooling layer.
                          The shape will be (N, 256, H', W'), where H' and W'
                          depend on the input image size. For a 224x224 input,
                          it's typically (N, 256, 6, 6).
        """
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        # If using LRN instead of BN: x = self.lrn(x) after ReLU

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        # If using LRN: x = self.lrn(x) after ReLU

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        # Layer 5
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.pool5(x)

        return x

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch_size, channels, height, width)
    # Common input size for AlexNet is 224x224 or 227x227.
    # Let's use 224x224.
    dummy_input = torch.randn(1, 3, 224, 224)

    # Instantiate the backbone
    alexnet_features = AlexNetBackbone(num_input_channels=3, use_batch_norm=True)
    print("AlexNet Backbone Architecture:")
    print(alexnet_features)

    # Perform a forward pass
    print(f"\nInput shape: {dummy_input.shape}")
    output_features = alexnet_features(dummy_input)
    print(f"Output feature map shape: {output_features.shape}")

    # Test without Batch Norm
    alexnet_features_no_bn = AlexNetBackbone(num_input_channels=3, use_batch_norm=False)
    print("\nAlexNet Backbone Architecture (no BatchNorm):")
    # print(alexnet_features_no_bn) # To keep output concise
    output_features_no_bn = alexnet_features_no_bn(dummy_input)
    print(f"Output feature map shape (no BatchNorm): {output_features_no_bn.shape}")

    # Expected output shape for 224x224 input: (N, 256, 6, 6)
    # Let's verify the output calculation:
    # Input: 224
    # Conv1 (k=11,s=4,p=2): floor((224 - 11 + 2*2) / 4) + 1 = floor(217/4) + 1 = 54 + 1 = 55
    # Pool1 (k=3,s=2): floor((55 - 3) / 2) + 1 = floor(52/2) + 1 = 26 + 1 = 27
    # Conv2 (k=5,s=1,p=2): floor((27 - 5 + 2*2) / 1) + 1 = floor(26/1) + 1 = 26 + 1 = 27 (stride is 1 for conv layers unless specified)
    # Pool2 (k=3,s=2): floor((27 - 3) / 2) + 1 = floor(24/2) + 1 = 12 + 1 = 13
    # Conv3 (k=3,s=1,p=1): floor((13 - 3 + 2*1) / 1) + 1 = floor(12/1) + 1 = 12 + 1 = 13
    # Conv4 (k=3,s=1,p=1): floor((13 - 3 + 2*1) / 1) + 1 = floor(12/1) + 1 = 12 + 1 = 13
    # Conv5 (k=3,s=1,p=1): floor((13 - 3 + 2*1) / 1) + 1 = floor(12/1) + 1 = 12 + 1 = 13
    # Pool5 (k=3,s=2): floor((13 - 3) / 2) + 1 = floor(10/2) + 1 = 5 + 1 = 6
    # So, the output spatial dimension should be 6x6.
    assert output_features.shape == (1, 256, 6, 6), "Output shape mismatch!"
