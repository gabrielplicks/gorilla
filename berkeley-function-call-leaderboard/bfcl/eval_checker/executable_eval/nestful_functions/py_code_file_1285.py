import torch

import torch.nn.functional as F



def forward_pass_conv_layer(x: torch.Tensor) -> torch.Tensor:

    """Performs the forward pass of a convolutional neural network (CNN) layer.



    Args:

        x: A 4D PyTorch tensor representing a mini-batch of images in the shape

           `[batch_size, channels, height, width]`.



    Returns:

        The output feature maps from the ReLU activation function.

    """

    conv_out = F.conv2d(x, weight, bias=bias)

    relu_out = F.relu(conv_out)

    return relu_out

