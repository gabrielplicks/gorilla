import pandas as pd



def print_datatypes(df: pd.DataFrame) -> None:

    """Prints the data type of each column in a pandas dataframe.



    Args:

        df: The input dataframe.

    """

    for column in df.columns:

        print(f"Column {column} has type: {df[column].dtypes}")

