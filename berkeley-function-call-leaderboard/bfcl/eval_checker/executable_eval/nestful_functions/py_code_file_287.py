import re

from typing import Dict



def replace_brackets_with_ids(input_string: str) -> str:

    """Replaces text within brackets in an input string with a unique ID.



    Args:

        input_string: The input string to process.



    Returns:

        The processed string with bracketed text replaced by unique IDs.

    """

    ids: Dict[str, str] = {}



    def replace_match(match) -> str:

        text = match.group(1)

        if text not in ids:

            ids[text] = f'ID_{len(ids) + 1}'

        return f'[{ids[text]}]'



    output_string = re.sub(r'\[(.*?)\]', replace_match, input_string)

    return output_string

