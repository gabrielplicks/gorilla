import torch



def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:

    """Generates an attention mask matrix for a transformer model given input IDs and a pad token ID.



    The attention mask is a binary matrix where 1 indicates that the corresponding token should be attended to and 0 indicates that the token should be ignored.



    Args:

        input_ids: The input IDs tensor.

        pad_token_id: The pad token ID.



    Returns:

        The attention mask tensor.

    """

    attention_mask = torch.ones_like(input_ids)

    attention_mask = attention_mask.masked_fill(input_ids == pad_token_id, 0)



    return attention_mask

