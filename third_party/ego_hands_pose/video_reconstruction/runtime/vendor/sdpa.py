from torch.nn import functional as F


def attention(q, k, v):
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
    )
    return result.transpose(1, 2).contiguous()
