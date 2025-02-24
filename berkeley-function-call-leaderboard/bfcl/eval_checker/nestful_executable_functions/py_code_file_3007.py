from typing import Union



def format_with_units(value: Union[float, str], unit: str) -> str:

    """Formats a numeric value with units.



    Args:

        value: The numeric value to be formatted.

        unit: The unit of the value.



    Raises:

        TypeError: If the value is not a float or an integer.

        ValueError: If the unit is invalid.

    """

    if not isinstance(value, float) and not isinstance(value, int):

        try:

            value = float(value)

        except ValueError:

            raise TypeError("Value must be a float or an integer.")

    valid_units = {'cm': 'cm', 'km': 'km', 'm': 'm', 'in': 'in', 'ft': 'ft', 'yd': 'yd'}

    if unit not in valid_units:

        raise ValueError(f"Invalid unit: {unit}")

    return f"{value} {valid_units[unit]}"

