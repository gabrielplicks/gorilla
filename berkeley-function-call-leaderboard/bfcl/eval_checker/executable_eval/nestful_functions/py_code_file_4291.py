import re



def parse_weight_error(weight_error: str) -> float:

    """Parses a weight error string and returns a numeric value in kilograms.



    Args:

        weight_error: A string representing a weight error.



    Returns:

        A numeric value in kilograms.

    """

    assert isinstance(weight_error, str), "Input must be a string"



    weight_error_pattern = r"^([+-]?[0-9]*\.?[0-9]+)\s*kg$"

    match = re.match(weight_error_pattern, weight_error)



    if match:

        weight = float(match.group(1))

        return weight

    else:

        return 0.0

