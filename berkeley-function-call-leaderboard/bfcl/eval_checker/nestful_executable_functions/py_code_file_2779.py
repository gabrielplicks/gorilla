from typing import Dict



def make_new_dict() -> Dict[int, str]:

    """Creates a dictionary that maps each number from 0 to 255 to its corresponding ASCII character.



    Returns:

        A dictionary that maps each number from 0 to 255 to its corresponding ASCII character.

    """

    return {i: chr(i) for i in range(256)}

