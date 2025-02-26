import pandas as pd

from typing import Callable



def create_dataframe(N: int) -> pd.DataFrame:

    """Creates a pandas DataFrame of length N with the following information:



    * The `ID` column should be a sequence of integers from `1` to `N`.

    * The `Odd/Even` column should indicate whether the ID is odd or even.

    * The `Sum` column should contain the sum of the ID and the previous ID.



    Args:

        N: The desired length of the DataFrame.



    Returns:

        The resulting DataFrame.

    """

    df = pd.DataFrame()

    df['ID'] = range(1, N + 1)

    is_odd: Callable[[int], str] = lambda x: 'odd' if x % 2 else 'even'

    df['Odd/Even'] = df['ID'].apply(is_odd)

    df['Sum'] = df['ID'].cumsum()



    return df

