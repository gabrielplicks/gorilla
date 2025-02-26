import datetime



def day_count(date_str: str) -> int:

    """Calculates the day count since 1970-01-01 for a given date in the format of `YYYY_MM_DD`.



    Args:

        date_str: The date string in the format of `YYYY_MM_DD`.



    """

    date_format = "%Y_%m_%d"  # input date format

    date_obj = datetime.datetime.strptime(date_str, date_format)

    date_1970 = datetime.datetime(1970, 1, 1)  # 1970-01-01

    diff_days = (date_obj - date_1970).days

    return diff_days

