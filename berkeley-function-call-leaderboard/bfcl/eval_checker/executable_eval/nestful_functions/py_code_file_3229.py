import torch



def batch_matmul_dot_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

    """Computes the dot product between two batches of matrices by batch matrix multiplication.

    Args:

        A: A batch of matrices with dimensions (batch_size, num_rows_A, num_cols_A).

        B: A batch of matrices with dimensions (batch_size, num_rows_B, num_cols_B).

    """

    return torch.matmul(A, B)

