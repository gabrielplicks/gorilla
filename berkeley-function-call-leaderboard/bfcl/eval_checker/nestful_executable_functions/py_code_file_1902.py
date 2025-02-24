from typing import TypeVar

import torch



ModelType = TypeVar("ModelType", bound=torch.nn.Module)



def freeze_model_parameters(model: ModelType) -> ModelType:

    """Freezes all the parameters of a pre-trained model in PyTorch.



    Parameter freezing means preventing the model from updating its parameters during the training process.

    This function accepts a PyTorch model as input and returns a new model with the same architecture but with all the parameters frozen.



    Args:

        model: The PyTorch model to freeze the parameters of.

    """

    for param in model.parameters():

        param.requires_grad = False

    return model

