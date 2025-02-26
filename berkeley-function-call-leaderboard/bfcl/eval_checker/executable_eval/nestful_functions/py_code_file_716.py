import re



def parse_region_list(input_string: str) -> list[str]:

    """Parses an input string that may contain complex region names in parentheses, such as "US (New York)", "US (California)" and "US (Arizona)", and returns a list of the regions as strings, excluding the parentheses.



    Args:

        input_string: The input string to parse.



    Returns:

        A list of the region names as strings.

    """

    region_regex = r"US \(.*?\)"

    matches = re.findall(region_regex, input_string)



    return matches

