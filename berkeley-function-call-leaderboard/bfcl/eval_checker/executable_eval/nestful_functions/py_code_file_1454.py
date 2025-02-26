from typing import List



def get_string_with_list_contents(input_list: List[str]) -> str:

    """Concatenates all the strings in the input list, separated by commas.



    Args:

        input_list: A list of strings.



    Returns:

        A string with all the strings in the input list separated by commas.

    """

    if not input_list:

        return ""



    if len(input_list) == 1:

        return input_list[0]



    output = input_list[0]

    for item in input_list[1:]:

        output += ", " + item



    return output

