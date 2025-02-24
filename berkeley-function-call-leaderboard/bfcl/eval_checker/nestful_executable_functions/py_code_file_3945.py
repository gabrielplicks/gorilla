from typing import Dict



def format_dict_table(score_dict: Dict[str, int]) -> str:

    """Formats a dictionary into a string table with keys and values separated by a tab character.



    Args:

        score_dict: A dictionary containing the keys and values to be formatted.



    Returns:

        A string containing the formatted table.

    """

    if not isinstance(score_dict, dict):

        return ''



    return ''.join([f"{key}\t{value}\n" for key, value in score_dict.items()])

