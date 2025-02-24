import pandas as pd



def longest_ride(df: pd.DataFrame) -> float:

    """Finds the length of the longest ride by duration in a given pandas dataframe.



    Args:

        df: The input dataframe with 'bike_id' and 'trip_duration' columns.



    Returns:

        The length of the longest ride by duration.



    Raises:

        IndexError: If there are no rides in the dataframe.

    """

    if df.empty:

        raise IndexError("No rides in the dataframe.")



    grouped = df.groupby('bike_id')['trip_duration'].max()

    return grouped.max()

