import torch
def one_hot_2d(label, nclass):
    h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, h*w)

    mask = torch.zeros(nclass+1, h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()
    return mask[:-1, :, :]

mask = torch.ones(2,2)
mask[0,0] = 0
mask[1,1] = 2
preds = torch.ones(2,2)
print(mask)
onehot_mask = one_hot_2d(mask, 3)
print(onehot_mask)