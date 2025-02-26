from typing import Dict



def generate_iterative_dictionary(length: int) -> Dict[str, str]:

    """Generates a dictionary with `length` number of keys and their corresponding values.

    The values for each key are a concatenation of the key and its numeric value.



    Args:

        length: The number of keys to generate in the dictionary.



    Returns:

        A dictionary with the generated keys and their corresponding values.

    """

    my_dict = {}

    for i in range(1, length + 1):

        my_dict.update({str(i): str(i) + str(i)})

    return my_dict

