import torch

import torch.nn as nn



def mask_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    """Computes the loss of a model in a masked fashion.

    Args:

        predicted: The model's predicted scores for each sample.

        target: The true labels for each sample.

    """

    mask = ~torch.isnan(target)

    predicted_masked = predicted[mask]

    target_masked = target[mask]

    loss_fn = nn.MSELoss()

    loss = loss_fn(predicted_masked, target_masked)

    return loss

