import pandas as pd



def partition_by_category(df: pd.DataFrame) -> dict:

    """Splits a pandas DataFrame into a dictionary of DataFrames based on the unique values of a column.



    Args:

        df: The input DataFrame.



    Returns:

        A dictionary where the keys are the unique values of the `category` column and the values are the corresponding DataFrames.

    """

    groups = df.groupby("category")

    return {category: group.drop("category", axis=1) for category, group in groups}

