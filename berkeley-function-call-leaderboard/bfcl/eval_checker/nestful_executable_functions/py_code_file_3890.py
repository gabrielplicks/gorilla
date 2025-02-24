import pandas as pd



def add_possession_stats(df: pd.DataFrame) -> pd.DataFrame:

    """Adds possession statistics to a DataFrame.



    Args:

        df: The input DataFrame with columns 'team_a', 'team_b', and 'score_a'.



    Returns:

        The modified DataFrame with new columns 'possession_a' and 'possession_b'.

    """

    assert {'team_a', 'team_b', 'score_a'}.issubset(df.columns), 'Missing required columns'

    df = df.assign(

        possession_a=lambda d: d['score_a'] / (d['score_a'] + d['score_b']),

        possession_b=lambda d: d['score_b'] / (d['score_a'] + d['score_b']),

    )

    return df

