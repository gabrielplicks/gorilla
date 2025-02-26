import pandas as pd



def sum_of_column(dataframe: pd.DataFrame, column_name: str) -> float:

    """Calculates the sum of a specific column from a pandas dataframe.



    Args:

        dataframe: The pandas dataframe.

        column_name: The name of the column to sum.

    """

    assert column_name in dataframe.columns, "Column does not exist in the dataframe"

    assert dataframe[column_name].dtype == 'int64' or dataframe[column_name].dtype == 'float64', "Column values must be numeric"

    return dataframe[column_name].sum()

