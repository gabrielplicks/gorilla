from typing import List



def parse_range_description(input_str: str) -> List[int]:

    """Parses a range description string and returns a list of integers.



    Args:

        input_str: A string representing a range description as a comma-separated list of ranges.



    Raises:

        ValueError: If the input string is invalid.

    """

    result = []

    values = input_str.split(",")



    for value in values:

        parts = value.split("-")



        if len(parts) == 1:

            result.append(int(parts[0]))

        elif len(parts) == 2:

            start, end = map(int, parts)

            result.extend(range(start, end + 1))

        else:

            raise ValueError("Invalid value: " + value)



    return result

