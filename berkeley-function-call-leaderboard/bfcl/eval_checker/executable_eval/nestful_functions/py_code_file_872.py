import numpy as np



def transform_end_effector_position(T: np.ndarray, end_effector_position: np.ndarray) -> np.ndarray:

    """Transforms the position of an end effector from its local frame to the world frame.



    Args:

        T: A 4x4 homogeneous transformation matrix representing the transformation from the local frame to the world frame.

        end_effector_position: A 1x4 vector representing the position of the end effector in its local frame.



    Returns:

        The position of the end effector in the world frame as a 1x3 vector.

    """

    transformed_position = np.matmul(T, end_effector_position)

    return transformed_position[:3]

