from typing import Union



def virtual_size(file_size: Union[int, float]) -> int:

    """Calculates the virtual size of a file given its physical size and the number of clusters.

    Args:

        file_size: The physical size of the file.

    """

    cluster_size = 4096

    num_clusters = file_size // cluster_size

    virtual_size = num_clusters * cluster_size

    if file_size % cluster_size != 0:

        virtual_size += cluster_size



    return virtual_size

