import numpy as np

import pandas as pd



def extract_data_from_dataframe(df: pd.DataFrame) -> tuple:

    """Extracts preprocessed data from a Pandas dataframe.



    Args:

        df: The input dataframe.



    Returns:

        A tuple of two elements, `(X, y)`. `X` is a numpy array containing the input features,

        while `y` is a numpy array containing the corresponding labels.

    """

    X = df['tokenized_text'].tolist()

    y = df['label_encoded'].tolist()

    X = np.array(X)

    y = np.array(y)



    return X, y

