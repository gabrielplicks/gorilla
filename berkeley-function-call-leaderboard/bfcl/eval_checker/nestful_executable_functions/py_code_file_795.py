import torch



def filter_images_by_channel_and_threshold(image_tensor: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:

    """

    Filters images based on channel and threshold.



    Args:

        image_tensor: A 4D tensor with dimensions of (batch_size, height, width, channels).

        threshold: A 1D tensor of the same size as the channels dimension of the image_tensor.



    Returns:

        A 3D tensor of shape (batch_size, height, width) that contains the filtered image.

    """

    output = (image_tensor > threshold).float()  # Convert the boolean tensor to float tensor

    output = torch.sum(output, dim=-1)  # Sum across the channels

    output = torch.where(output > 0, torch.ones_like(output), torch.zeros_like(output))  # Check if any channel passes the threshold

    return output

