import torch
import torch.nn.functional as F
import torch.nn as nn
def statistic_pseudo_labels(masks):
    '''
    b,c,h,w = masks.shape
    函数主要是计算传入进来标签之间的重叠部分，重叠的部分就被认为是一个可供参考的伪标签
    masks的值有0，1，2。伪标签的计算法则如下：
    1、将所有的mask大小调整为masks[0]的一样
    2、创建一个新的全0的mask，名为cover_all_mask
    3、cover_all_mask中某个像素点在b个mask中的值都是1，则cover_all_mask也赋值为1，
    4、cover_all_mask中某个像素点在b个mask中的值都是2，则cover_all_mask也赋值为2，
    '''
    b,c,h,w = masks.shape
    for i in range(b):
        masks[i] = F.interpolate(masks[i].unsqueeze(0),size=masks[0].size()[-2:], mode='bilinear').squeeze(0)
    # 创建与masks形状相同的全0张量
    cover_all_masks = torch.zeros_like(masks)

    # 为每个mask in batch计算cover_all_mask
    for i in range(b):
        # 对于每个像素位置，检查所有mask是否都是1或2
        cover_all_masks[i][(masks == 1).all(dim=0)] = 1
        cover_all_masks[i][(masks == 2).all(dim=0)] = 2
    return cover_all_masks

def one_hot_2d(label, nclass):
    h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, h*w)

    mask = torch.zeros(nclass+1, h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()
    return mask[:-1, :, :]


class PrototypeManager(nn.Module):
    def __init__(self, num_classes, num_channels, dtype, device):
        super(PrototypeManager, self).__init__()
        self.prototypes = torch.zeros((num_classes, num_channels),
                                      dtype=dtype, device=device)
        # self.register_buffer('prototypes', self.prototypes)
        self.initialized = False
        self.num_classes = num_classes

    def compute_prototype(self,feats,masks,nclass):
        feats = F.interpolate(feats, size=masks.size()[-2:], mode='bilinear')
        b, c, h, w = feats.size()
        prototypes = torch.zeros((nclass, c),
                                 dtype=feats.dtype,
                                 device=feats.device)
        for i in range(b):
            cur_mask = masks[i]
            cur_mask_onehot = one_hot_2d(cur_mask, nclass)

            cur_feat = feats[i]
            cur_prototype = torch.zeros((nclass, c),
                                        dtype=feats.dtype,
                                        device=feats.device)

            cur_set = list(torch.unique(cur_mask))
            if nclass in cur_set:
                cur_set.remove(nclass)
            if 255 in cur_set:
                cur_set.remove(255)

            for cls in cur_set:  # cur_set:0,1,2
                # 获取mask中当前类别的像素数量
                m = cur_mask_onehot[cls].view(1, h, w)
                sum = m.sum()
                m = m.expand(c, h, w).view(c, -1)
                cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1) / (sum + 1e-6)
                cur_prototype[cls, :] = cls_feat
            prototypes += cur_prototype
        return prototypes / b

    def initialize(self, feats, masks, num_classes):
        # 您的初始化逻辑，例如：
        new_prototypes = self.compute_prototype(feats, masks, num_classes)
        self.prototypes = new_prototypes
        self.initialized = True

    def update_prototypes(self, feats, masks, momentum=0.99):
        if not self.initialized:
            self.initialize(feats, masks, self.num_classes)
            return
        # 使用EMA更新
        new_prototypes = self.compute_prototype(feats,masks,self.num_classes)  # compute prototypes based on feats and masks
        # new_prototypes = new_prototypes.mean(dim=0)
        self.prototypes = (1.0 - momentum) * self.prototypes + momentum * new_prototypes

    def forward(self, feats, masks):
        self.update_prototypes(feats, masks)
        return self.prototypes

class Correct_PrototypeManager(nn.Module):
    def __init__(self,num_classes, num_channels, dtype, device):
        super(Correct_PrototypeManager, self).__init__()
        self.prototypes = torch.zeros((num_classes, num_channels),
                                      dtype=dtype, device=device)
        # self.register_buffer('prototypes', self.prototypes)
        self.initialized = False
        self.num_classes = num_classes

    def compute_prototype(self,feats,preds,masks,nclass):
        feats = F.interpolate(feats, size=masks.size()[-2:], mode='bilinear')
        b, c, h, w = feats.size()
        prototypes = torch.zeros((nclass, c),
                                 dtype=feats.dtype,
                                 device=feats.device)
        for i in range(b):
            cur_mask = masks[i]
            cur_pred = preds[i]
            cur_mask_onehot = one_hot_2d(cur_mask, nclass)
            cur_pred_onehot = one_hot_2d(cur_pred, nclass)
            cur_mask_onehot = cur_mask_onehot * cur_pred_onehot

            cur_feat = feats[i]
            cur_prototype = torch.zeros((nclass, c),
                                        dtype=feats.dtype,
                                        device=feats.device)

            cur_set = list(torch.unique(cur_mask))
            if nclass in cur_set:
                cur_set.remove(nclass)
            if 255 in cur_set:
                cur_set.remove(255)

            for cls in cur_set:  # cur_set:0,1,2
                # 获取mask中当前类别的像素数量
                m = cur_mask_onehot[cls].view(1, h, w)
                sum = m.sum()
                m = m.expand(c, h, w).view(c, -1)
                cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1) / (sum + 1e-6)
                cur_prototype[cls, :] = cls_feat
            prototypes += cur_prototype
        return prototypes / b

    def initialize(self, feats,preds, masks, num_classes):
        # 您的初始化逻辑，例如：
        new_prototypes = self.compute_prototype(feats,preds, masks, num_classes)
        self.prototypes = new_prototypes
        self.initialized = True

    def update_prototypes(self, feats,preds, masks, momentum=0.99):
        if not self.initialized:
            self.initialize(feats,preds, masks, self.num_classes)
            return
        # 使用EMA更新
        new_prototypes = self.compute_prototype(feats,preds,masks,self.num_classes)  # compute prototypes based on feats and masks
        # new_prototypes = new_prototypes.mean(dim=0)
        self.prototypes = (1.0 - momentum) * self.prototypes + momentum * new_prototypes

    def forward(self, feats,preds,masks):
        preds = torch.argmax(preds, dim=1)
        self.update_prototypes(feats,preds, masks)
        return self.prototypes



def pseudo_from_prototype(prototypes,feats,threshold=0.0):
    """
    我想让feats里面的每个像素与prototypes中的每个类别进行余弦相似度计算，距离最近的那个就是当前feats的伪标签
    最后返回一个mask
    prototypes:(num_class,c)
    feats:(b,c,h,w)
    """
    # 步骤1: 对prototypes和feats进行标准化
    prototypes_norm = prototypes / (prototypes.norm(p=2, dim=1, keepdim=True) + 1e-6)
    feats_norm = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-6)

    # 步骤2: 计算余弦相似度
    similarity = torch.einsum('nc,bchw->bnhw', prototypes_norm, feats_norm)  # b x num_class x h x w

    # 步骤3: 获取最相似的原型作为伪标签
    '''
    原来我的代码思想是，返回值最大的那一层的索引回去用作伪标签的值，但现在我希望
    1、对similarity在dim=1上进行softmax
    2、设置一个阈值，如果在dim上最大的那个值与另外通道的的差值没有超过这个阈值，则使用3为当前点的标签，而不是使用当前最大值通道值为标签
    3、如果在dim上最大的那个值与另外通道的的差值超过这个阈值，则当前最大值通道值为标签
    '''
    probabilities = F.softmax(similarity,dim=1)
    # 2. 找到最大概率及其索引
    max_values, max_indices = torch.max(probabilities, dim=1)
    # 3. 找到第二大的概率
    sorted_probs, _ = torch.sort(probabilities, descending=True, dim=1)
    second_max_values = sorted_probs[:, 1, :, :]
    # 4. 计算最大概率与第二大概率之间的差值
    diff = max_values - second_max_values

    # 6. 根据差值与阈值的比较，得到伪标签
    pseudo_labels = torch.where(diff > threshold, max_indices, torch.tensor(3, dtype=torch.long).to(similarity.device))

    # pseudo_labels = torch.argmax(similarity, dim=1)  # b x numclass_ x h x w

    return pseudo_labels



if __name__ == '__main__':
    input = torch.ones(2,1,256,256)
    out = statistic_pseudo_labels(input)
    print(out.shape)