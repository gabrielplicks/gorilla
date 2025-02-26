from typing import Dict



def replace_in_dict(d: Dict, old: str, new: str) -> None:

    """Recurses through a dictionary (including nested dictionaries) and replaces all instances of a given string with another string.

    Args:

        d: The dictionary to be modified.

        old: The string to be replaced.

        new: The replacement string.

    """

    def _replace_in_dict(d):

        for key, val in d.items():

            if isinstance(val, dict):

                _replace_in_dict(val)

            elif isinstance(val, str):

                d[key] = val.replace(old, new)



    _replace_in_dict(d)

