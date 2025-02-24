import json

from typing import List, Dict



def format_sensor_data(data: str) -> str:

    """Formats sensor data from a JSON-formatted string into a human-readable string.



    Args:

        data: A JSON-formatted string of sensor data.



    Returns:

        A human-readable string of formatted sensor data.

    """

    sensor_list: List[Dict[str, str]] = json.loads(data)

    formatted_sensors: List[str] = []

    for sensor in sensor_list:

        if "value" in sensor:

            formatted_sensor: str = f"{sensor['name']}: {sensor['value']} {sensor['unit']}"

        else:

            formatted_sensor: str = f"{sensor['name']}: N/A"

        formatted_sensors.append(formatted_sensor)

    return '\n'.join(formatted_sensors)

