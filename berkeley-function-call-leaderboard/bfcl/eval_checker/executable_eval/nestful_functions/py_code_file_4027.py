import torch

from typing import Any



def has_parameter_with_substring(model: torch.nn.Module, substring: str) -> bool:

    """Recursively traverses a Pytorch model and identifies all parameters that have a name that contains a given substring.

    Args:

        model: The Pytorch model to traverse.

        substring: The substring to search for in parameter names.

    Returns:

        A boolean indicating whether any matching parameters were found.

    """

    def recursive_check(model: torch.nn.Module, substring: str) -> bool:

        return any(substring in name for name in model.state_dict())



    if isinstance(model, torch.nn.Module):

        for key, value in model.named_parameters():

            if substring in key:

                return True

        for module in model.children():

            if recursive_check(module, substring):

                return True

    return False

