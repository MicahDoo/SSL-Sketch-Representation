# src/models/raster_encoder.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50SketchEncoderBase(nn.Module):
    """
    A ResNet50-based feature extractor for sketch images.
    This module provides the ResNet50 backbone, with the first convolutional
    layer modified to accept a specified number of input channels (e.g., 1 for grayscale
    sketches or 3 if input images are already 3-channel).
    The final fully connected classification layer of the original ResNet50 is removed.
    The output of this module is the feature map from the ResNet backbone
    (typically after the average pooling layer).
    """
    def __init__(self, input_channels=1, use_pretrained=False, freeze_pretrained=False):
        """
        Args:
            input_channels (int): Number of input channels for the sketch images (1 for grayscale, 3 for RGB).
            use_pretrained (bool): If True, loads weights pre-trained on ImageNet.
                                   If False, initializes weights randomly.
            freeze_pretrained (bool): If True and use_pretrained is True, freezes the weights
                                      of the pre-trained backbone layers during training.
                                      Only applicable if use_pretrained is True.
        """
        super(ResNet50SketchEncoderBase, self).__init__()
        self.input_channels = input_channels
        self.use_pretrained = use_pretrained

        if self.use_pretrained:
            print(f"Loading PRETRAINED ResNet50 weights. Input channels: {self.input_channels}")
            resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            print(f"Initializing ResNet50 from SCRATCH. Input channels: {self.input_channels}")
            resnet50 = models.resnet50(weights=None) 

        # Modify the first convolutional layer if input_channels is not 3
        # or if we are training from scratch with a specific number of input channels.
        if self.input_channels != 3 or not self.use_pretrained:
            original_conv1_weights = None
            if self.use_pretrained and self.input_channels == 1 and resnet50.conv1.in_channels == 3:
                # Strategy for adapting pretrained 3-channel weights to 1-channel input:
                # Average the weights of the original 3 input channels.
                original_conv1_weights = resnet50.conv1.weight.data.mean(dim=1, keepdim=True)
                print("  Adapting first conv layer from 3-channel (pretrained) to 1-channel by averaging weights.")
            elif self.use_pretrained and self.input_channels != 1 and self.input_channels !=3:
                 print(f"  Warning: Pretrained weights are for 3 channels. Adapting to {self.input_channels} channels by averaging first conv layer's weights (if possible) or random init.")
                 # Fallback or more complex strategy needed if not 1 or 3 channels with pretraining
                 original_conv1_weights = resnet50.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1,self.input_channels,1,1)


            # Create new conv1 layer
            new_conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), 
                                  stride=(2, 2), padding=(3, 3), bias=False)
            
            if original_conv1_weights is not None:
                new_conv1.weight.data = original_conv1_weights
            # If not using pretrained weights, new_conv1 will be initialized randomly by PyTorch.
            
            resnet50.conv1 = new_conv1
            print(f"  Replaced first conv layer with a new one for {self.input_channels} input channel(s).")

        # Remove the final fully connected layer (the classifier)
        # The output of this module will be the features before this fc layer.
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.output_feature_dim = resnet50.fc.in_features # Typically 2048 for ResNet50

        if self.use_pretrained and freeze_pretrained:
            print("  Freezing weights of the pretrained backbone.")
            for param in self.features.parameters():
                param.requires_grad = False
            # Unfreeze the modified conv1 if it was adapted, so it can learn
            if self.input_channels != 3: # If conv1 was changed
                 for param in self.features[0].parameters(): # self.features[0] is the conv1 layer
                    param.requires_grad = True
                 print("  Unfroze the modified conv1 layer.")


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sketch images, shape [batch_size, input_channels, height, width].
        Returns:
            torch.Tensor: Feature representation from the ResNet backbone, 
                          shape [batch_size, self.output_feature_dim] after flattening.
        """
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten the output for use in subsequent layers
        return x

if __name__ == '__main__':
    # Example Usage:
    print("\n--- Testing ResNet50SketchEncoderBase ---")

    # Test 1: From scratch, 1 input channel (e.g., for grayscale sketches)
    print("\nTest 1: From scratch, 1 input channel")
    encoder_scratch_1channel = ResNet50SketchEncoderBase(input_channels=1, use_pretrained=False)
    dummy_input_gray = torch.randn(2, 1, 224, 224) # Batch of 2, 1 channel, 224x224
    features_gray = encoder_scratch_1channel(dummy_input_gray)
    print(f"Output feature shape (1-channel, scratch): {features_gray.shape}") # Expected: [2, 2048]
    assert features_gray.shape == (2, encoder_scratch_1channel.output_feature_dim)

    # Test 2: Using pretrained weights, adapting for 1 input channel
    print("\nTest 2: Pretrained, adapting to 1 input channel")
    encoder_pretrained_1channel = ResNet50SketchEncoderBase(input_channels=1, use_pretrained=True)
    features_pretrained_gray = encoder_pretrained_1channel(dummy_input_gray)
    print(f"Output feature shape (1-channel, pretrained): {features_pretrained_gray.shape}")
    assert features_pretrained_gray.shape == (2, encoder_pretrained_1channel.output_feature_dim)

    # Test 3: Using pretrained weights, with 3 input channels (e.g., if sketches are repeated to 3 channels)
    print("\nTest 3: Pretrained, 3 input channels")
    encoder_pretrained_3channel = ResNet50SketchEncoderBase(input_channels=3, use_pretrained=True)
    dummy_input_rgb = torch.randn(2, 3, 224, 224) # Batch of 2, 3 channels
    features_rgb = encoder_pretrained_3channel(dummy_input_rgb)
    print(f"Output feature shape (3-channel, pretrained): {features_rgb.shape}")
    assert features_rgb.shape == (2, encoder_pretrained_3channel.output_feature_dim)
    
    # Test 4: From scratch, 3 input channels
    print("\nTest 4: From scratch, 3 input channels")
    encoder_scratch_3channel = ResNet50SketchEncoderBase(input_channels=3, use_pretrained=False)
    features_scratch_rgb = encoder_scratch_3channel(dummy_input_rgb)
    print(f"Output feature shape (3-channel, scratch): {features_scratch_rgb.shape}")
    assert features_scratch_rgb.shape == (2, encoder_scratch_3channel.output_feature_dim)

    # Test 5: Pretrained, 1 input channel, freezing backbone
    print("\nTest 5: Pretrained, 1 input channel, freeze backbone")
    encoder_frozen = ResNet50SketchEncoderBase(input_channels=1, use_pretrained=True, freeze_pretrained=True)
    # Check if params are frozen (except potentially conv1)
    frozen_params = 0
    unfrozen_params = 0
    for name, param in encoder_frozen.named_parameters():
        if param.requires_grad:
            unfrozen_params +=1
            # print(f"  Unfrozen: {name}")
        else:
            frozen_params +=1
    print(f"  Frozen parameters: {frozen_params}, Unfrozen parameters: {unfrozen_params}")
    assert unfrozen_params > 0 # At least conv1 should be unfrozen
    features_frozen = encoder_frozen(dummy_input_gray)
    print(f"Output feature shape (1-channel, pretrained, frozen): {features_frozen.shape}")


    print("\nAll tests passed!")
