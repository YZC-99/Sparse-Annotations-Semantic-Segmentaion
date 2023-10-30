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
mask[1,1] = 3
preds = torch.ones(2,2)
print(mask)
onehot_mask = one_hot_2d(mask, 3)
print(onehot_mask)
new_mask = onehot_mask.view(3, -1).max(-1)[0]
print(new_mask)
print(new_mask.view(3, 1, 1))