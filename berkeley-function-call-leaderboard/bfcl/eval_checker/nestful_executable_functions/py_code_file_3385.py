from typing import Tuple



def is_one_of_modes(is_read_mode: bool, is_write_mode: bool, is_append_mode: bool, is_update_mode: bool, is_binary_mode: bool) -> bool:

    """Checks if at least one of the mode flags is set.



    Args:

        is_read_mode: A boolean indicating if the read mode is set.

        is_write_mode: A boolean indicating if the write mode is set.

        is_append_mode: A boolean indicating if the append mode is set.

        is_update_mode: A boolean indicating if the update mode is set.

        is_binary_mode: A boolean indicating if the binary mode is set.



    Returns:

        A boolean indicating if at least one of the flags is set to True.

    """

    return is_read_mode or is_write_mode or is_append_mode or is_update_mode or is_binary_mode

