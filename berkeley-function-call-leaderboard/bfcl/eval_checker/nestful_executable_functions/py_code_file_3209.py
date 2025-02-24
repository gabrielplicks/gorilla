import numpy as np



def select_objects(points: np.ndarray, anchor_index: int, context_frame_indices: List[int]) -> np.ndarray:

    """Selects objects from a numpy array based on their indices.



    Args:

        points: A numpy array of shape `(N, 3)`, representing 3D coordinates for `N` objects.

        anchor_index: The index of the anchor frame.

        context_frame_indices: A list of indices of the contextual frames.



    Returns:

        A new numpy array containing only the objects with the specified indices.

    """

    mask = np.isin(points[:, 0], [anchor_index] + context_frame_indices)

    return points[mask]

