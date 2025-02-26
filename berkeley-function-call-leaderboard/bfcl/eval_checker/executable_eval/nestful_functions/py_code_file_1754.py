import urllib.request

import json

from typing import Dict



def get_weather_data(url: str) -> Dict[str, float]:

    """Retrieves weather data from a given URL.



    Args:

        url: The URL to retrieve the data from.



    Returns:

        A dictionary containing the temperature, humidity, wind_speed, and rainfall data.

    """

    response = urllib.request.urlopen(url)

    weather_data = json.loads(response.read())

    return {

        'temperature': weather_data['temperature'],

        'humidity': weather_data['humidity'],

        'wind_speed': weather_data['wind_speed'],

        'rainfall': weather_data['rainfall'],

    }

