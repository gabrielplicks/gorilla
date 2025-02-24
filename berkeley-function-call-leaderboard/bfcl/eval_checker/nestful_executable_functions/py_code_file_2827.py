import pandas as pd

import numpy as np



def add_color_code(df: pd.DataFrame) -> pd.DataFrame:

    """Adds a new column named "color_code" to the data frame based on the deal amount.



    Args:

        df: The input data frame containing columns for acquirer, target, deal amount, and deal date.



    Returns:

        The input data frame with an additional column "color_code".

    """

    bins = [0, 10000000, 100000000, np.inf]

    labels = ["green", "yellow", "red"]

    df["color_code"] = np.where(df["deal_amount"] < 10000000, "green", np.where(df["deal_amount"] < 100000000, "yellow", "red"))



    return df

