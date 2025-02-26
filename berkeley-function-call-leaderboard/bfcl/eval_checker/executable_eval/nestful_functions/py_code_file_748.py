import re

from typing import List



def extract_layer_numbers(model: List[str]) -> List[int]:

    """Extracts the layer numbers for all convolutional layers in the model.

    Utilizes string manipulation to extract the layer number from the layer's name.

    If the layer name does not contain a layer number, the function returns None.



    Args:

        model: A list of strings representing the layers in the model.



    Returns:

        A list of integers representing the layer numbers for all convolutional layers in the model.

    """

    layer_numbers = []

    for layer_name in model:

        match = re.search(r"layer(\d+)", layer_name)

        if match:

            layer_numbers.append(int(match.group(1)))

        else:

            layer_numbers.append(None)

    return layer_numbers

