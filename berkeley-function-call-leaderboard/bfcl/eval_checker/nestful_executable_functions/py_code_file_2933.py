import time



def timestamp_difference(timestamp: float) -> int:

    """Calculates the difference between the current timestamp and a given timestamp in seconds, rounded to the nearest integer.



    Args:

        timestamp: The earlier timestamp.

    """

    current_timestamp = time.time()

    difference = current_timestamp - timestamp

    return round(difference)

