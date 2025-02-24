from typing import List, Union



def convert_to_lower_case(tokens: List[Union[str, float, bool, int]]) -> List[str]:

    """Converts a list of strings into lower case.



    Args:

        tokens: A list of tokens.



    Returns:

        A list of lower case strings.

    """

    lower_case_tokens = []



    for token in tokens:

        try:

            if isinstance(token, str):

                lower_case_tokens.append(token.lower())

        except TypeError:

            pass



    return lower_case_tokens

