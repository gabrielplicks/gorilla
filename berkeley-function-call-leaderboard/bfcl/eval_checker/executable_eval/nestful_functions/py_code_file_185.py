from typing import Optional

from pathlib import Path



def join_paths(path1: Optional[Path], path2: Optional[Path]) -> Path:

    """Concatenates two file paths using the Path type from the pathlib module.



    Args:

        path1: The first file path.

        path2: The second file path.



    Raises:

        ValueError: If either path is None.

    """

    if path1 is None or path2 is None:

        raise ValueError("neither path can be None")



    joined_path = Path(path1) / Path(path2)

    return joined_path

