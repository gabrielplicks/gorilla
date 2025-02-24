import pandas as pd



def get_average_of_three_factors(df: pd.DataFrame) -> float:

    """Calculates the arithmetic average of three factors in a pandas dataframe.

    Args:

        df: The pandas dataframe containing the factors.

    """

    average = df[['factor_one', 'factor_two', 'factor_three']].mean().mean()

    return float(average)

