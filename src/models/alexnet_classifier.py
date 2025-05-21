# src/models/alexnet_classifier.py
import torch
import torch.nn as nn
import math # For floor function

# Assuming your project structure is set up correctly and src is in PYTHONPATH
# or you are running scripts from the root of 'my_advanced_dl_project'
from src.backbones.alexnet import AlexNetBackbone # Assuming alexnet.py is in src/backbones/

class AlexNetClassifier(nn.Module):
    """
    AlexNet model for image classification.
    Combines the AlexNetBackbone with a fully connected classifier head.
    The size of the first fully connected layer in the classifier is determined
    by calculating the expected output dimensions of the backbone based on the
    provided input image height and width.
    """
    def __init__(self, num_classes: int,
                 input_image_height: int,
                 input_image_width: int,
                 num_input_channels: int = 3,
                 backbone_use_batch_norm: bool = True,
                 dropout_prob: float = 0.5):
        """
        Initializes the AlexNet classifier.

        Args:
            num_classes (int): The number of classes for the classification task.
            input_image_height (int): The height of the input images.
            input_image_width (int): The width of the input images.
            num_input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
                                      Passed to the AlexNetBackbone.
            backbone_use_batch_norm (bool): Whether the backbone should use Batch Normalization.
                                            Passed to the AlexNetBackbone.
            dropout_prob (float): Dropout probability for the fully connected layers
                                  in the classifier head.
        """
        super(AlexNetClassifier, self).__init__()

        self.input_image_height = input_image_height
        self.input_image_width = input_image_width

        self.backbone = AlexNetBackbone(
            num_input_channels=num_input_channels,
            use_batch_norm=backbone_use_batch_norm
        )

        # Calculate the expected spatial dimensions of the backbone's output
        backbone_out_h, backbone_out_w = self._calculate_backbone_output_dims(
            self.input_image_height, self.input_image_width
        )

        # The backbone is assumed to output 256 channels.
        backbone_output_channels = 256
        fc_input_features = backbone_output_channels * backbone_out_h * backbone_out_w

        if fc_input_features <= 0:
            raise ValueError(
                f"Calculated fc_input_features is {fc_input_features}. "
                f"This usually means the input_image_height ({self.input_image_height}) "
                f"and/or input_image_width ({self.input_image_width}) are too small "
                "for the AlexNet architecture."
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
        Calculates the output HxW dimensions of the AlexNetBackbone
        given input HxW dimensions.
        This must match the layer configurations in AlexNetBackbone.
        """
        # Layer 1: Conv1 -> Pool1
        # Conv1: k=11, s=4, p=2
        h_out = math.floor((h_in - 11 + 2 * 2) / 4) + 1
        w_out = math.floor((w_in - 11 + 2 * 2) / 4) + 1
        # Pool1: k=3, s=2
        h_out = math.floor((h_out - 3) / 2) + 1
        w_out = math.floor((w_out - 3) / 2) + 1

        # Layer 2: Conv2 -> Pool2
        # Conv2: k=5, s=1, p=2
        h_out = math.floor((h_out - 5 + 2 * 2) / 1) + 1
        w_out = math.floor((w_out - 5 + 2 * 2) / 1) + 1
        # Pool2: k=3, s=2
        h_out = math.floor((h_out - 3) / 2) + 1
        w_out = math.floor((w_out - 3) / 2) + 1

        # Layer 3: Conv3
        # Conv3: k=3, s=1, p=1
        h_out = math.floor((h_out - 3 + 2 * 1) / 1) + 1
        w_out = math.floor((w_out - 3 + 2 * 1) / 1) + 1

        # Layer 4: Conv4
        # Conv4: k=3, s=1, p=1
        h_out = math.floor((h_out - 3 + 2 * 1) / 1) + 1
        w_out = math.floor((w_out - 3 + 2 * 1) / 1) + 1

        # Layer 5: Conv5 -> Pool5
        # Conv5: k=3, s=1, p=1
        h_out = math.floor((h_out - 3 + 2 * 1) / 1) + 1
        w_out = math.floor((w_out - 3 + 2 * 1) / 1) + 1
        # Pool5: k=3, s=2
        h_out = math.floor((h_out - 3) / 2) + 1
        w_out = math.floor((w_out - 3) / 2) + 1

        return h_out, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the AlexNet classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where N is batch size, C is num_input_channels,
                              H is input_image_height, and W is input_image_width
                              (as specified during initialization).

        Returns:
            torch.Tensor: Output logits for each class, shape (N, num_classes).
        """
        # assert x.shape[2] == self.input_image_height and x.shape[3] == self.input_image_width, \
        #     f"Warning: Input image HxW ({x.shape[2]}x{x.shape[3]}) does not match \
        #     model's configured HxW ({self.input_image_height}x{self.input_image_width}). \
        #     This may lead to unexpected behavior or errors if the calculated \
        #     feature map size for the classifier head is incorrect."
            # This warning is helpful during development/debugging.
            # In production, you might handle this differently or ensure inputs always match.
            # For robustness, one could re-calculate expected feature map size here if dynamic sizing
            # during forward pass was desired, but the current design is for fixed size at init.

        # Get features from the backbone
        features = self.backbone(x)

        # Flatten the features before passing to the classifier
        x_flattened = torch.flatten(features, 1) # Flatten all dimensions except batch

        # Pass through the classifier head
        output_logits = self.classifier(x_flattened)

        return output_logits

if __name__ == '__main__':
    num_example_classes = 1000
    dummy_input_batch_size = 4
    img_channels = 3

    # --- Test with standard 224x224 input ---
    img_height_std = 224
    img_width_std = 224
    dummy_input_std = torch.randn(dummy_input_batch_size, img_channels, img_height_std, img_width_std)

    alexnet_model_std = AlexNetClassifier(
        num_classes=num_example_classes,
        input_image_height=img_height_std,
        input_image_width=img_width_std,
        num_input_channels=img_channels,
        backbone_use_batch_norm=True,
        dropout_prob=0.5
    )
    print(f"AlexNet Classifier configured for {img_height_std}x{img_width_std} input images.")
    # Manually verify calculated backbone output for 224x224 (should be 6x6)
    calc_h, calc_w = alexnet_model_std._calculate_backbone_output_dims(img_height_std, img_width_std)
    print(f"Calculated backbone output HxW for {img_height_std}x{img_width_std} input: {calc_h}x{calc_w}")
    assert (calc_h, calc_w) == (6,6), "Calculation for 224x224 input is incorrect!"


    print(f"\n--- Testing with Standard Input ({img_height_std}x{img_width_std}) ---")
    print(f"Input shape: {dummy_input_std.shape}")
    output_std = alexnet_model_std(dummy_input_std)
    print(f"Output logits shape: {output_std.shape}")
    assert output_std.shape == (dummy_input_batch_size, num_example_classes)
    print("Successfully tested with standard input.")

    # --- Test with a different input size, e.g., 192x192 ---
    # For AlexNetBackbone, a 192x192 input should result in a 5x5 feature map.
    img_height_custom = 192
    img_width_custom = 192
    dummy_input_custom = torch.randn(dummy_input_batch_size, img_channels, img_height_custom, img_width_custom)

    alexnet_model_custom = AlexNetClassifier(
        num_classes=num_example_classes,
        input_image_height=img_height_custom,
        input_image_width=img_width_custom
    )
    print(f"\nAlexNet Classifier configured for {img_height_custom}x{img_width_custom} input images.")
    calc_h_custom, calc_w_custom = alexnet_model_custom._calculate_backbone_output_dims(img_height_custom, img_width_custom)
    print(f"Calculated backbone output HxW for {img_height_custom}x{img_width_custom} input: {calc_h_custom}x{calc_w_custom}")
    assert (calc_h_custom, calc_w_custom) == (5,5), "Calculation for 192x192 input is incorrect!"

    print(f"\n--- Testing with Custom Input ({img_height_custom}x{img_width_custom}) ---")
    print(f"Input shape: {dummy_input_custom.shape}")
    output_custom = alexnet_model_custom(dummy_input_custom)
    print(f"Output logits shape: {output_custom.shape}")
    assert output_custom.shape == (dummy_input_batch_size, num_example_classes)
    print("Successfully tested with custom input size.")

    # --- Test with a very small input size that might cause issues ---
    img_height_tiny = 64
    img_width_tiny = 64
    dummy_input_tiny = torch.randn(dummy_input_batch_size, img_channels, img_height_tiny, img_width_tiny)
    print(f"\n--- Testing with Tiny Input ({img_height_tiny}x{img_width_tiny}) ---")
    try:
        alexnet_model_tiny = AlexNetClassifier(
            num_classes=num_example_classes,
            input_image_height=img_height_tiny,
            input_image_width=img_width_tiny
        )
        # If the above doesn't raise an error, try a forward pass
        # output_tiny = alexnet_model_tiny(dummy_input_tiny)
        # print(f"Output logits shape for tiny input: {output_tiny.shape}")
    except ValueError as e:
        print(f"Caught expected ValueError for tiny input: {e}")

