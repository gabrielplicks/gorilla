from typing import Union



def compute_time_bucket(timestamp: int, bucket_size: int) -> Union[int, float]:

    """Calculates the time bucket for a given timestamp and bucket size.



    Args:

        timestamp: The timestamp to calculate the time bucket for.

        bucket_size: The size of the time bucket.



    Returns:

        The time bucket for the given timestamp and bucket size.

    """

    quotient = timestamp / bucket_size



    rounded_quotient = round(quotient)



    if rounded_quotient < 0:

        rounded_quotient = int(rounded_quotient)



    return rounded_quotient

