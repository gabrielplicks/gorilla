import re



def merge_odd_position_elements(string: str) -> str:

    """Merges the elements in odd positions of a string, separated by space, and removes all the characters that match the regular expression '\w{3}'.



    Args:

        string: The input string.



    """

    string = re.sub(r'\w{3}', '', string)

    elements = string.split()

    merged_string = ' '.join(elements[1::2])



    return merged_string

