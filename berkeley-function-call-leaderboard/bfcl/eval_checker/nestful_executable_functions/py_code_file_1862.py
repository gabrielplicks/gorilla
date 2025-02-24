from typing import Union

import pandas as pd



def insert_df_into_table(df: pd.DataFrame, table_name: str, conn: Union[str, pd.io.sql.SQLDatabase]) -> None:

    """Inserts each row of a pandas dataframe into a database table.



    Args:

        df: A pandas dataframe to insert into the table.

        table_name: The name of the database table to insert into.

        conn: A connection to the database.

    """

    df.to_sql(table_name, conn, if_exists='append', index=False)

