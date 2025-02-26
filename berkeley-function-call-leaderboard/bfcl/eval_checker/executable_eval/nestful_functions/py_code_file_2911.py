import torch



def divide_by_sum_and_preserve_shape(input_tensor: torch.Tensor) -> torch.Tensor:

    """Divides each element of an input tensor by the sum of all elements in the input tensor while preserving the same shape.



    Args:

        input_tensor: The input tensor.



    Returns:

        A tensor of the same shape as the input tensor, where each element is divided by the sum of all elements in the input tensor.

    """

    input_sum = input_tensor.sum()

    reshaped_input = input_tensor.reshape(-1)

    output_tensor = reshaped_input / input_sum

    return output_tensor.reshape(input_tensor.shape)

