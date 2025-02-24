import pandas as pd



def calculate_gender_percentage(df: pd.DataFrame) -> dict:

    """Calculates the percentage of male and female passengers in a given data frame.



    Args:

        df: The input data frame.



    Returns:

        A dictionary containing the percentage of male and female passengers.

    """

    if not isinstance(df, pd.DataFrame):

        return "Invalid data frame"



    male_passengers = df['Sex'].value_counts()['male']

    female_passengers = df['Sex'].value_counts()['female']

    total_passengers = len(df)



    male_percentage = male_passengers / total_passengers

    female_percentage = female_passengers / total_passengers



    return {'Male': male_percentage, 'Female': female_percentage}

