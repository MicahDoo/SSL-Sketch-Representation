# src/models/resnet50.py
import torch
import torch.nn as nn
import math  # For floor function

# Assuming your project structure is set up correctly and src is in PYTHONPATH
# or you are running scripts from the root of 'my_advanced_dl_project'
from src.backbones.resnet50 import ResNet50Backbone  # Assuming resnet50.py is in src/backbones/

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 model for image classification.
    Combines the ResNet50Backbone with a fully connected classifier head.
    The size of the first fully connected layer in the classifier is determined
    by calculating the expected output dimensions of the backbone based on the
    provided input image height and width.
    """
    def __init__(self,
                 num_classes: int,
                 input_image_height: int,
                 input_image_width: int,
                 num_input_channels: int = 3,
                 backbone_pretrained: bool = True,
                 dropout_prob: float = 0.5):
        """
        Initializes the ResNet-50 classifier.

        Args:
            num_classes (int): The number of classes for the classification task.
            input_image_height (int): The height of the input images.
            input_image_width (int): The width of the input images.
            num_input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
                                      Passed to the ResNet50Backbone.
            backbone_pretrained (bool): Whether to load ImageNet-pretrained weights.
            dropout_prob (float): Dropout probability for the fully connected layers
                                  in the classifier head.
        """
        super(ResNet50Classifier, self).__init__()

        self.input_image_height = input_image_height
        self.input_image_width = input_image_width

        self.backbone = ResNet50Backbone(
            num_input_channels=num_input_channels,
            pretrained=backbone_pretrained
        )

        # Calculate the expected spatial dimensions of the backbone's output
        backbone_out_h, backbone_out_w = self._calculate_backbone_output_dims(
            self.input_image_height,
            self.input_image_width
        )

        # The backbone is assumed to output 2048 channels.
        backbone_output_channels = 2048
        fc_input_features = backbone_output_channels * backbone_out_h * backbone_out_w

        if fc_input_features <= 0:
            raise ValueError(
                f"Calculated fc_input_features is {fc_input_features}. "
                f"This usually means the input_image_height ({self.input_image_height}) "
                f"and/or input_image_width ({self.input_image_width}) are too small "
                "for the ResNet-50 architecture."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(fc_input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _calculate_backbone_output_dims(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Calculates the output HxW dimensions of the ResNet50Backbone
        given input HxW dimensions.
        Follows the sequence: conv1 (7x7, s=2, p=3) -> maxpool (3x3, s=2, p=1)
        -> layer2 downsample (1x1, s=2) -> layer3 downsample -> layer4 downsample.
        """
        # conv1: k=7, s=2, p=3
        h = math.floor((h_in - 7 + 2 * 3) / 2) + 1
        w = math.floor((w_in - 7 + 2 * 3) / 2) + 1
        # maxpool: k=3, s=2, p=1
        h = math.floor((h - 3 + 2 * 1) / 2) + 1
        w = math.floor((w - 3 + 2 * 1) / 2) + 1
        # layer2 downsample: conv1 in first block, k=1, s=2, p=0
        h = math.floor((h - 1 + 0) / 2) + 1
        w = math.floor((w - 1 + 0) / 2) + 1
        # layer3 downsample
        h = math.floor((h - 1 + 0) / 2) + 1
        w = math.floor((w - 1 + 0) / 2) + 1
        # layer4 downsample
        h = math.floor((h - 1 + 0) / 2) + 1
        w = math.floor((w - 1 + 0) / 2) + 1
        return h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ResNet-50 classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where N is batch size, C is num_input_channels,
                              H is input_image_height, and W is input_image_width
                              (as specified during initialization).

        Returns:
            torch.Tensor: Output logits for each class, shape (N, num_classes).
        """
        # Extract features
        features = self.backbone(x)
        # Flatten the features before passing to the classifier
        x_flat = torch.flatten(features, 1)
        # Classifier head
        logits = self.classifier(x_flat)
        return logits

if __name__ == '__main__':
    num_example_classes = 1000
    batch_size = 4
    img_ch = 3

    # --- Standard 224x224 input ---
    h_std, w_std = 224, 224
    inp_std = torch.randn(batch_size, img_ch, h_std, w_std)
    model_std = ResNet50Classifier(
        num_classes=num_example_classes,
        input_image_height=h_std,
        input_image_width=w_std,
        num_input_channels=img_ch,
        backbone_pretrained=False,
        dropout_prob=0.5
    )
    print(f"Configured for {h_std}x{w_std} input; "
          f"calc HxW = {model_std._calculate_backbone_output_dims(h_std, w_std)}")
    out_std = model_std(inp_std)
    print(f"Output logits shape: {out_std.shape}")
    assert out_std.shape == (batch_size, num_example_classes)

    # --- Custom 128x128 input (expect 4x4 feature map) ---
    h_c, w_c = 128, 128
    inp_c = torch.randn(batch_size, img_ch, h_c, w_c)
    model_c = ResNet50Classifier(
        num_classes=num_example_classes,
        input_image_height=h_c,
        input_image_width=w_c
    )
    calc_h, calc_w = model_c._calculate_backbone_output_dims(h_c, w_c)
    print(f"\nConfigured for {h_c}x{w_c}; calc HxW = {calc_h}x{calc_w}")
    assert (calc_h, calc_w) == (4, 4)
    out_c = model_c(inp_c)
    print(f"Output logits shape: {out_c.shape}")
    assert out_c.shape == (batch_size, num_example_classes)

    # --- Tiny 64x64 input test (should still compute dims >=1) ---
    h_t, w_t = 64, 64
    print(f"\nTesting tiny {h_t}x{w_t} configuration...")
    try:
        _ = ResNet50Classifier(
            num_classes=num_example_classes,
            input_image_height=h_t,
            input_image_width=w_t
        )
        print("Tiny input configuration succeeded.")
    except ValueError as e:
        print(f"Caught expected error for tiny input: {e}")
