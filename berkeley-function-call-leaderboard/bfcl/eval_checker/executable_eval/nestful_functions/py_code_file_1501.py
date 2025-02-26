from typing import List



def format_percentiles(percentiles: List[float], precision: int = 2) -> str:

    """Formats a list of percentiles as a string with the specified precision.



    Args:

        percentiles: A list of percentiles.

        precision: The number of decimal places to use in the formatted percentages.

    """

    formatted_percentiles = ["{:.{}f}%".format(percentile * 100, precision) for percentile in percentiles]

    return "[" + ", ".join(formatted_percentiles) + "]"

