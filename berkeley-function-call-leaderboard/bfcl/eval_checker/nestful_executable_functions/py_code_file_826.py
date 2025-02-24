from typing import Dict



def convert_to_bytes(value: int, unit: str) -> str:

    """Converts a numeric value to its corresponding value in bytes.



    Args:

        value: The numeric value to convert.

        unit: The unit of measurement for the value.



    Raises:

        ValueError: If the input is not valid.

    """

    if not isinstance(value, int):

        raise ValueError("Invalid input: value is not a number")

    if unit not in ('mb', 'gb', 'tb'):

        raise ValueError(f"Invalid input: invalid unit '{unit}'")

    bytes_per_unit: Dict[str, int] = {

        'mb': 1024 ** 2,

        'gb': 1024 ** 3,

        'tb': 1024 ** 4,

    }

    bytes_value = value * bytes_per_unit[unit]

    return f"{bytes_value} bytes"

