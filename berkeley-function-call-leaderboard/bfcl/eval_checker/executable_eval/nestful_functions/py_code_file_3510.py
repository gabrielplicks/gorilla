import pandas as pd



def dict_to_df(data: dict) -> pd.DataFrame:

    """Converts a dictionary of lists into a Pandas data frame.



    The dictionary keys will be the column names, and the list values will be the column values.



    Args:

        data: A dictionary where keys are column names and values are lists of column values.

    """

    return pd.DataFrame(data)

