import torch



def create_even_numbers_tensor() -> torch.Tensor:

    """Creates a PyTorch tensor that contains the first 100 positive integers in the form of a 10x10 matrix.

    Then creates a new tensor that only includes the even numbers from the original tensor.



    Returns:

        A PyTorch tensor containing the even numbers from the original 10x10 matrix.

    """

    matrix = torch.arange(1, 101).reshape(10, 10)

    even_numbers_tensor = matrix[matrix % 2 == 0]



    return even_numbers_tensor

