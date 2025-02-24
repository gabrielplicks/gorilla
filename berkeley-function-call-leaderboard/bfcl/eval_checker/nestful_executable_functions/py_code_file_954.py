from typing import Union



def format_measurement_interval(measurement_value: Union[int, float], unit: str) -> str:

    """Formats a measurement interval using named mapping.



    Args:

        measurement_value: The measurement value.

        unit: The unit of measurement.

    """

    format_string = "The measurement interval is {measurement_value} {unit}."

    formatted_string = format_string.format(measurement_value=measurement_value, unit=unit)

    return formatted_string

