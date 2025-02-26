from typing import List, Tuple



def split_text_into_key_val_tuples(text: str) -> Tuple[List[Tuple[str, str]], str]:

    """Splits a string of key-value pairs into a list of tuples representing the split key-value pairs.



    Args:

        text: A string of key-value pairs.



    Returns:

        A tuple containing a list of tuples representing the split key-value pairs and the default value.

    """

    lines = text.splitlines()

    default_value = None

    output = []

    for line in lines:

        key, value = line.split(':', 1)

        if key == 'default':

            default_value = value

        else:

            output.append((key, value))

    return output, default_value

