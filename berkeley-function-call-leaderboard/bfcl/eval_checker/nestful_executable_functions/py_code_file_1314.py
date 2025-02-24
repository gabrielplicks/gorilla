from typing import List



def clean_values(values: List[str]) -> List[int]:

    """

    Converts a list of string values to integers and removes any non-numeric values.



    Args:

        values: A list of string values.



    Returns:

        A list of the remaining numeric values.

    """

    cleaned_values = []



    for value in values:

        try:

            cleaned_values.append(int(value))

        except ValueError:

            pass  # Ignore non-numeric values



    return cleaned_values

