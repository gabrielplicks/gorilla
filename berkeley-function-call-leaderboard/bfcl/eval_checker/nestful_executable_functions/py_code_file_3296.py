from typing import Dict, Union



def chart_type_value_to_name(chart_type_value: int) -> Union[str, None]:

    """

    Converts a chart type value to a human-readable name.

    Args:

        chart_type_value: A positive integer representing a chart type.

    Returns:

        The human-readable name for the chart type, or None if the chart type is not recognized.

    """

    chart_type_map: Dict[int, str] = {

        1: "Line",

        2: "Bar",

        3: "Area",

        4: "Scatter",

        5: "Pie",

        6: "Bubble",

        7: "Donut",

        8: "Table",

        9: "Guage"

    }

    return chart_type_map.get(chart_type_value, None)

