from typing import Dict, Any



def extract_data(data: Dict[str, Any]) -> List[int]:

    """Extracts all integer values from a nested dictionary.



    Args:

        data: A dictionary with string keys and values of any type.



    Returns:

        A list of all integer values present in the input dictionary.

    """

    result = []



    def extract(data: Dict[str, Any]) -> None:

        """Recursively extracts integer values from a dictionary."""

        for key, value in data.items():

            if isinstance(value, int):

                result.append(value)

            elif isinstance(value, dict):

                extract(value)



    extract(data)

    return result

