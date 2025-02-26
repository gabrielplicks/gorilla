from typing import List, Dict



def convert_2d_array_to_1d_array(arr_2d: List[List[int]], lookup_table: Dict[int, int]) -> List[int]:

    """Converts a 2D array of binary values into a 1D array of binary values, where each binary value is determined by a lookup table.

    Args:

        arr_2d: A 2D array of shape `(M, N)` where `M` is the number of rows and `N` is the number of columns.

        lookup_table: A dictionary consisting of keys from 0 to 2<sup>N</sup> and values of 0 or 1.

    """

    arr_1d = [item for sublist in arr_2d for item in sublist]

    binary_values = [lookup_table[entry] for entry in arr_1d]



    return binary_values

