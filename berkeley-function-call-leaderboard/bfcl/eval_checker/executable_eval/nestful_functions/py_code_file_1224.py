import pandas as pd

from typing import Dict



def group_and_aggregate_data(df: pd.DataFrame, grouping_column: str) -> Dict[str, float]:

    """Groups and aggregates data from a Pandas dataframe.



    Args:

        df: The input dataframe.

        grouping_column: The name of the column used for grouping.



    Returns:

        A dictionary where the keys are the unique values of the grouping column and the values are the sum of the values in the 'median_income' column for the corresponding group.

    """

    try:

        grouped_data = df.groupby(grouping_column)['median_income'].sum()

        return grouped_data.to_dict()

    except Exception as e:

        print(f"An error occurred while grouping and aggregating data: {e}")

        return {}

