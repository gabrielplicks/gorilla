import sys



def get_object_size_in_mb(obj: object) -> str:

    """Calculates the size of a given object in megabytes (MB) and returns a string with the rounded size in MB followed by the unit "MB".



    Args:

        obj: The object to calculate the size of.



    Returns:

        A string with the rounded size in MB followed by the unit "MB".

    """

    byte_size = sys.getsizeof(obj)

    mb_size = byte_size / (1024 * 1024)

    return f'{round(mb_size, 2)} MB'

