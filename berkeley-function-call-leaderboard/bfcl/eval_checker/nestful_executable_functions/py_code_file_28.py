from typing import List, Dict



def get_layers(input_size: int, hidden_sizes: List[int], output_size: int) -> Dict[str, List[int]]:

    """Creates a dictionary of neural network layers.



    Args:

        input_size: The size of the input layer.

        hidden_sizes: A list of sizes for the hidden layers.

        output_size: The size of the output layer.



    Returns:

        A dictionary of neural network layers.

    """

    layers = dict()

    layers['input_layer'] = [input_size]

    for i, hidden_size in enumerate(hidden_sizes):

        layers[f'hidden_layer_{i + 1}'] = [hidden_size]

    layers['output_layer'] = [output_size]

    return layers

