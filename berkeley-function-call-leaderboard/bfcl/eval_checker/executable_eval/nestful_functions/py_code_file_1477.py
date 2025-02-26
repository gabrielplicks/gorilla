import pandas as pd



def transform_dataframe(dataframe: pd.DataFrame, column_check: str = None) -> pd.DataFrame:

    """Processes a Pandas DataFrame in a specific way.



    If `column_check` is `None`, the function does not do anything and simply returns the input DataFrame.

    If it is a string, then the function checks if the string is present as the name of a column in the DataFrame.

    If so, it returns a new DataFrame with a column appended, containing the same data in the specified column,

    but with all values multiplied by 2.



    Args:

        dataframe: The input DataFrame.

        column_check: The name of the column to check and process. Default is `None`.

    """

    if column_check is None:

        return dataframe

    if column_check in dataframe.columns:

        return dataframe.assign(new_column=dataframe[column_check] * 2)

    else:

        return dataframe

