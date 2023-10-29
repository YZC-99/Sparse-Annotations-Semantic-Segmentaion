import numpy as np
import torch
import torch.nn.functional as F
def clean_mask(mask, cls_label, softmax=True):
    if softmax:
        mask = F.softmax(mask, dim=1)
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)

mask = torch.rand(2,3,256,256)
# cls_label = torch.tensor([[1,1,1],[1,1,1]])
# out = clean_mask(mask,cls_label)
mask = mask.view(2, 1, 256, 256)
print(mask.shape)