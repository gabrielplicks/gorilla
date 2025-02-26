import pandas as pd



def shuffle_rows(df: pd.DataFrame) -> pd.DataFrame:

    """Shuffles the rows of a Pandas DataFrame and returns it.

    Args:

        df: The DataFrame to be shuffled.

    """

    shuffled_df = df.sample(frac=1)  # Shuffle the DataFrame by passing frac=1

    return shuffled_df

