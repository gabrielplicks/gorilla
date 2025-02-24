import pandas as pd

from typing import List, Tuple, Union



def process_dataframe(df: pd.DataFrame) -> List[Union[Tuple[int, str], None]]:

    """Processes a pandas DataFrame and returns a list of tuples containing the values of the "Year" and "City" columns for rows where the "Type" column is equal to "Gold".



    Args:

        df: The input DataFrame.



    Returns:

        A list of tuples containing the values of the "Year" and "City" columns for rows where the "Type" column is equal to "Gold", or None if the "Type" column is not equal to "Gold".

    """

    result = []

    for _, row in df.iterrows():

        if row["Type"] == "Gold":

            result.append((row["Year"], row["City"]))

        else:

            result.append(None)

    return result

