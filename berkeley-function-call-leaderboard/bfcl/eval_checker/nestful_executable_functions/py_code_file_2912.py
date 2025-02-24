from typing import List



def output_string_formatted(text_list: List[str]) -> str:

    """Formats a list of strings such that they are output in the form:



    [

      'FirstString',

      'SecondString',

      'ThirdString'

    ]



    The function accepts a `text_list` argument and returns the formatted string. If the list is empty, the output is `[]`.



    Args:

        text_list: A list of strings to be formatted.

    """

    if len(text_list) == 0:

        return "[]"

    else:

        output = "["

        for string in text_list:

            output += f'\n  "{string}"'

        output += "\n]"

        return output

