import pandas as pd



def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    """Imputes missing values in a Pandas DataFrame with the mean value of the corresponding column.



    Args:

        df: The input DataFrame.



    Returns:

        A new DataFrame with the imputed values.

    """

    df_imputed = df.copy()

    for column in df.columns:

        if df_imputed[column].isnull().any():

            mean_value = df_imputed[column].mean()

            df_imputed[column] = df_imputed[column].fillna(mean_value)

    return df_imputed

