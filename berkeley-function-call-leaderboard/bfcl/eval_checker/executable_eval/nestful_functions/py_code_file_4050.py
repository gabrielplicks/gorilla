from typing import AnyStr



def count_consecutive_characters(s: AnyStr) -> AnyStr:

    """

    Returns a string containing the count of consecutive characters in the input string.

    The output string is constructed by appending the count of consecutive occurrences of each character in the input string.



    Args:

        s: The input string.



    Returns:

        A string containing the count of consecutive characters in the input string.

    """

    output = ""

    prev_char = ""

    count = 0



    for char in s:

        if char == prev_char:

            count += 1

        else:

            if count > 0:

                output += prev_char + str(count)

            prev_char = char

            count = 1



    if count > 0:

        output += prev_char + str(count)



    return output

