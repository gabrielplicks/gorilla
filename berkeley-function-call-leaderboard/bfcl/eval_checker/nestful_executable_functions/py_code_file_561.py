import pandas as pd

import typing



def get_atemp_from_weather_df(weather_df: pd.DataFrame) -> typing.List[float]:

    """Extracts the `atemp` column from a Pandas DataFrame `weather_df` and returns the values as a list of floats.



    Args:

        weather_df: The input DataFrame containing the `atemp` column.



    Returns:

        A list of floats representing the `atemp` column values.

    """

    return weather_df['atemp'].tolist()

