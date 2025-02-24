import pandas as pd

from typing import Dict



def check_data_types(df: pd.DataFrame, expected_types: Dict[str, type]) -> bool:

    """Checks if the data types of a pandas data frame are as expected.



    Args:

        df: The pandas data frame to check.

        expected_types: A dictionary mapping column names to expected data types.



    Returns:

        A boolean value indicating whether the data types are valid or not.

    """

    for col in df.columns:

        if df[col].dtypes != expected_types[col]:

            return False

    return True

