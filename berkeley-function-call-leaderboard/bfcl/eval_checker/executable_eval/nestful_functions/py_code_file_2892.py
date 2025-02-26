import pandas as pd

from typing import Union



def add_duration(data: Union[str, pd.DataFrame]) -> pd.DataFrame:

    """

    Adds a duration column to the input data, which contains the number of days that elapsed between the end_date and start_date for each row.



    Args:

        data: The input data in CSV format or a Pandas DataFrame.



    Returns:

        A Pandas DataFrame with a duration column added.

    """

    if isinstance(data, str):

        df = pd.read_csv(data, parse_dates=['start_date', 'end_date'])

    else:

        df = data

    df['duration'] = (df['end_date'] - df['start_date']).dt.days

    df['duration'].fillna((df['end_date'].max() - df['start_date'].min()).days, inplace=True)

    return df

