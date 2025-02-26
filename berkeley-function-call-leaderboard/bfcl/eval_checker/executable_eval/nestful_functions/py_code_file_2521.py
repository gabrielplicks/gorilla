from typing import Union



def replace_japanese_numbers(input_str: Union[str, int]) -> Union[str, int]:

    """Replaces all occurrences of Japanese numbers with the corresponding English numbers.

    Args:

        input_str: The input string or integer containing Japanese numbers.

    """

    japanese_to_english = {

        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',

        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'

    }

    is_string = isinstance(input_str, str)

    output = '' if is_string else 0

    for char in input_str:

        if char in japanese_to_english:

            output += japanese_to_english[char]

        else:

            output += char

    return output if is_string else int(output)

