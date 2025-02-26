from typing import Union



def validate_model(model: Union[float, int], threshold: float) -> bool:

    """Validates a machine learning model and determines whether it is accurate enough.

    Args:

        model: The accuracy of the machine learning model.

        threshold: The minimum accuracy required for the model to be considered valid.

    """

    return model >= threshold

