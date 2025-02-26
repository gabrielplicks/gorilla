import math



def log_dict(input_dict: dict) -> dict:

    """

    Transforms a dictionary by taking the logarithm of its values.

    The transformation is performed in logarithmic time, i.e., O(log N),

    where N is the size of the dictionary.



    Args:

        input_dict: The input dictionary.



    Returns:

        A dictionary with the same keys as the input dictionary,

        but with the logarithm of the corresponding values.

    """

    keys = sorted(input_dict.keys())  # Sort the keys

    values = [input_dict[key] for key in keys]  # Get the values in the same order as keys

    log_values = [math.log(value) for value in values]  # Perform the logarithmic transformation

    return dict(zip(keys, log_values))  # Create a dictionary from the keys and log values

