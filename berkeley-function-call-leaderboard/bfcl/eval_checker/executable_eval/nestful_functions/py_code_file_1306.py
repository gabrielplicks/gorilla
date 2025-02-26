from typing import List



def validate_and_create_dict(data: List[str]) -> dict:

    """Creates a dictionary from a list of strings.



    Args:

        data: A list of strings.



    Raises:

        AssertionError: If the input is not a non-empty list of strings.

    """

    assert isinstance(data, list) and len(data) > 0 and all(isinstance(item, str) for item in data), "Invalid input"

    if len(data) % 2 == 0:

        return {data[i]: data[i + 1] for i in range(0, len(data), 2)}

    return {data[i]: None for i in range(0, len(data), 2)}

