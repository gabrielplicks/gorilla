import pandas as pd



def partition_data_frame(df: pd.DataFrame) -> list[pd.DataFrame]:

    """Partitions a pandas data frame into three data frames based on the values in the "label" column.



    Args:

        df: The input data frame.



    Returns:

        A list of three data frames.

    """

    groups = df.groupby('label')

    return [group.copy() for _, group in groups]

