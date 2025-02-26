from typing import Dict



def get_field02_of_item03(data: Dict[str, Dict[str, str]]) -> str:

    """Returns the value of field02 of item03 from the given data.



    Args:

        data: The data to parse.



    Returns:

        The value of field02 of item03.

    """

    return data["item03"]["field02"]

