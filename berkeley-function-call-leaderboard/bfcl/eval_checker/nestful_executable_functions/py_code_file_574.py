from typing import Dict



def process_temperatures(temps: Dict[str, float]) -> str:

    """Processes a dictionary of temperatures and returns a string containing the temperatures in Fahrenheit.

    Args:

        temps: A dictionary where the keys are city names and the values are temperatures in Celsius.

    Returns:

        A string containing the temperatures in Fahrenheit, formatted as "City1: Temp1°F, City2: Temp2°F, ...".

    """

    output = []

    for city, temp in temps.items():

        fahrenheit = temp * 9 / 5 + 32

        output.append(f"{city}: {fahrenheit:.1f}°F")

    return ", ".join(output)

