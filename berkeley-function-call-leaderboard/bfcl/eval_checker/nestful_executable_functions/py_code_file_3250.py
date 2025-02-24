import torch



def serialize_torch_model(model: torch.nn.Module, filepath: str) -> torch.nn.Module:

    """Serializes a trained PyTorch model to a file.

    Args:

        model: The PyTorch model object.

        filepath: The target file path.

    """

    torch.save(model, filepath)

    return torch.load(filepath)



def load_torch_model(filepath: str) -> torch.nn.Module:

    """Loads a serialized PyTorch model from a file.

    Args:

        filepath: The file path of the serialized model.

    """

    return torch.load(filepath)

