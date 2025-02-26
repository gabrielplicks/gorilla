import torch



def complex_exponential(z: torch.Tensor) -> torch.Tensor:

    """Calculates the complex exponential of each element in a complex tensor `z`.



    The complex exponential is defined as `e^(z*1j)`. Here `1j` denotes the imaginary part of the complex number.



    Args:

        z: A complex tensor.



    Returns:

        A tensor of the same shape as `z` containing the complex exponential of each element in `z`.

    """

    z_reshaped = z.view(-1, 2)

    complex_numbers = torch.complex(z_reshaped[:, 0], z_reshaped[:, 1])

    exponentials = torch.exp(complex_numbers * 1j)

    exponentials_reshaped = exponentials.view(z.shape)



    return exponentials_reshaped

