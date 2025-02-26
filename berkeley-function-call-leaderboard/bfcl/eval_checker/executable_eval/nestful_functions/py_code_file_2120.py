import re

import pandas as pd



def find_all_image_files(df: pd.DataFrame) -> list:

    """Finds all rows in a pandas dataframe that have a file_path value that ends with ".png" or ".jpg".



    Args:

        df: The pandas dataframe containing a column 'file_path'.



    Returns:

        A list of rows in the dataframe that have a 'file_path' value that ends with ".png" or ".jpg".

    """

    image_files = []



    for index, row in df.iterrows():

        file_path = row['file_path']

        if re.search(r'(\.png|\.jpg)$', file_path):

            image_files.append(row)



    return image_files

