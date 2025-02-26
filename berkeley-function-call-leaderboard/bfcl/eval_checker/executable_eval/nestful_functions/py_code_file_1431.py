import torch



def combine_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:

    """Combines two masks into a single mask.



    Args:

        mask1: The first mask.

        mask2: The second mask.



    Returns:

        The combined mask.

    """

    if mask2.shape == mask1.shape:

        return mask1 & mask2

    else:

        ndim = len(mask1.shape)

        target_mask_idx = 0 if ndim == 2 else 1

        target_mask = mask2[target_mask_idx]

        return mask1 & target_mask

