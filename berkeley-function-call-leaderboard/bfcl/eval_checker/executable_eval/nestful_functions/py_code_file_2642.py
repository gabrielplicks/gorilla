import pandas as pd



def convert_type_to_int(csv_file: str) -> pd.DataFrame:

    """

    Converts the 'Type' column in a csv file to integers, where 'A' = 1 and 'B' = 2.



    Args:

        csv_file: The path to the csv file.



    Returns:

        The modified DataFrame.

    """

    df = pd.read_csv(csv_file)

    df['Type'] = df['Type'].replace({'A': 1, 'B': 2})

    return df

