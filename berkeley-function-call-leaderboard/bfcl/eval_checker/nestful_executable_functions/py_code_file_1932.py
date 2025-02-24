from typing import Dict



def was_successful(header: Dict[str, int]) -> bool:

    """Checks if the request was successful based on the status code in the header.



    Args:

        header: A dictionary containing the status code of the request.



    Returns:

        True if the request was successful, False otherwise.

    """

    return header['status_code'] == 200

