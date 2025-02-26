import pandas as pd

from typing import List



def convert_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """Converts a DataFrame with a `labels` column of lists to a DataFrame with a `label` column of strings.



    Args:

        df: The input DataFrame.



    Returns:

        A new DataFrame with the desired structure.

    """

    new_rows = []

    for index, row in df.iterrows():

        category = row['category']

        description = row['description']

        for label in row['labels']:

            new_row = {'category': category, 'label': label, 'description': description}

            new_rows.append(new_row)



    new_df = pd.DataFrame(new_rows)

    return new_df

