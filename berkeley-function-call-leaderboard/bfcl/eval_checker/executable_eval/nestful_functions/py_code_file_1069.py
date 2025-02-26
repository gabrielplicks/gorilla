import pandas as pd



def flatten_dict_to_dataframe(d: dict) -> pd.DataFrame:

    """Creates a pandas DataFrame from a nested dictionary.



    Args:

        d: The nested dictionary to flatten.



    Returns:

        A pandas DataFrame containing the flattened data.

    """

    df = pd.DataFrame(d)

    return df

