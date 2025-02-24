import torch

import torch.distributions as dists



def generate_ood_inputs(distribution: dists.Distribution, num_samples: int) -> torch.Tensor:

    """Generates out-of-distribution (OOD) test inputs from a given distribution's probability density function (PDF) and outputs them as a PyTorch tensor.



    Args:

        distribution: A torch.distribution object.

        num_samples: The desired number of samples to generate.



    Returns:

        A PyTorch tensor with the specified number of samples from the PDF.

    """

    tensor = torch.distributions.distribution.Distribution.sample(distribution, (num_samples,))

    tensor = tensor.type(torch.float32)



    return tensor

