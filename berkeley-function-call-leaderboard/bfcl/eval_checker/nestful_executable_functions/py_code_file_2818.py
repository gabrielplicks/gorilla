import pandas as pd

from typing import List



def extract_languages_from_df(df: pd.DataFrame) -> List[str]:

    """Extracts a list of unique languages from a Pandas dataframe.



    The dataframe contains two columns: 'id' and 'languages'. The 'languages' column contains strings of the form 'lang1, lang2, lang3' with up to three languages in the string.



    Args:

        df: The input Pandas dataframe.



    Returns:

        A list of unique languages found in the dataframe.

    """

    languages_list = df['languages'].str.split(', ').explode('languages').unique().tolist()

    return languages_list

