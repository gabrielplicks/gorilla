import torch



def power_spectrum(tensor: torch.Tensor) -> torch.Tensor:

    """Calculates the power spectrum of a 3D tensor.



    Args:

        tensor: A 3D tensor with dimensions (samples, frequency bins, channels).



    Returns:

        A 2D tensor representing the average power spectrum for each channel.

    """

    reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])

    dft = torch.fft.fft(reshaped_tensor)

    power = torch.abs(dft) ** 2

    average_power = torch.mean(power, dim=0)



    return average_power

