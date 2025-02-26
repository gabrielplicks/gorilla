import re

from typing import Tuple, Optional



def parse_operator(operator_str: str) -> Optional[Tuple[str, str, str]]:

    """Parses an operator string and returns a tuple containing the operator name, operator symbol, and version range.



    Args:

        operator_str: The operator string to parse.



    Returns:

        A tuple containing the operator name, operator symbol, and version range, or `None` if the input string does not match the expected format.

    """

    pattern = r'(\w+)(==|>=|<|<=|!=)(.+)'

    match = re.match(pattern, operator_str)

    if match:

        operator_name, operator_symbol, version_range = match.groups()

        return (operator_name, operator_symbol, version_range)

    else:

        return None

