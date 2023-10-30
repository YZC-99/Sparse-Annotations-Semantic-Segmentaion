import argparse
from itertools import cycle
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD



from torch.utils.data import DataLoader
import yaml
from model.pseudo_tools import *
from dataset.semi_sass import *
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, evaluate
from util.dist_helper import setup_distributed

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["LOCAL_RANK"] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'
# sh tools/train_city.sh 3 28890

parser = argparse.ArgumentParser(description='Sparsely-annotated Semantic Segmentation')
parser.add_argument('--config', type=str, default='configs/semi-drishti.yaml')
parser.add_argument('--save-path', type=str, default='exp/semi-drishti')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

#nih

def main():
    args = parser.parse_args()

    # cfg = yaml.load(open(os.path.join('/root/autodl-tmp/Saprse-Annotations-Semantic-Segmentaion',args.config), "r"), Loader=yaml.Loader)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    logger.info('{}\n'.format(pprint.pformat(cfg)))

    os.makedirs(args.save_path, exist_ok=True)

    # cudnn.enabled = True
    # cudnn.benchmark = True

    model = DeepLabV3Plus(cfg, aux=cfg['aux'])

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(local_rank)

    ohem = False if cfg['criterion']['name'] == 'CELoss' else True
    use_weight = False

    trainset = Semi_DrishtiDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                           cfg['crop_size'], cfg['aug'])
    valset = Semi_DrishtiDataset(cfg['dataset'], cfg['data_root'], 'val', None)

#     trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True)
#     valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0

    #==========初始化prototype=========
    # prototype_manager = PrototypeManager(cfg['nclass'],256,torch.float32,'cuda:0')
    prototype_manager = Correct_PrototypeManager(cfg['nclass'],256,torch.float32,'cuda:0')
    #===================
    for epoch in range(cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        loss_m = AverageMeter()
        seg_m = AverageMeter()
        gmm_m = AverageMeter()

#         trainsampler.set_epoch(epoch)

        for i, (labeled_img,labeled_mask,labeled_id,unlabeled_img,unlabeled_mask,unlabeled_id) in enumerate(trainloader):
            labeled_img, labeled_mask, unlabeled_img, unlabeled_mask = labeled_img.cuda(), labeled_mask.cuda(), unlabeled_img.cuda(), unlabeled_mask.cuda()

            input_img = torch.cat([labeled_img,unlabeled_img],dim=0)

            feat, pred = model(input_img)
            labeled_feat, labeled_pred = feat[:cfg['batch_size'],...] ,pred[:cfg['batch_size'],...]
            unlabeled_feat, unlabeled_pred = feat[cfg['batch_size']:,...] , pred[cfg['batch_size']:,...]

            seg_loss = loss_calc(labeled_pred, labeled_mask,
                                 ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)

            # prototypes = prototype_manager(labeled_feat.detach(), labeled_mask)
            prototypes = prototype_manager(labeled_feat.detach(),labeled_pred.detach(), labeled_mask)
            if epoch < 20:
                threshold = 0.0
            else:
                threshold = 0.1
            pseudo_mask_1 = pseudo_from_prototype(prototypes,unlabeled_feat,threshold)
            pseudo_mask_1 = F.interpolate(pseudo_mask_1.unsqueeze(1).float(), size=unlabeled_pred.size()[-2:],
                                      mode='bilinear').squeeze().long()

            # Gaussian
            cur_cls_label = build_cur_cls_label(pseudo_mask_1, cfg['nclass'])
            pred_cl = F.softmax(unlabeled_pred, dim=1)

            #proto_loss：L_con
            vecs, unlabled_proto_loss = cal_protypes(unlabeled_feat,pseudo_mask_1, cfg['nclass'])
            _, labled_proto_loss = cal_protypes(labeled_feat,labeled_mask, cfg['nclass'])

            # res应该是代表公式(6)的平方(b,num_class,h,w)
            res = GMM(unlabeled_feat, vecs, pred_cl, pseudo_mask_1, cur_cls_label)
            gmm_loss = cal_gmm_loss(unlabeled_pred.softmax(1), res, cur_cls_label, pseudo_mask_1) + unlabled_proto_loss

            # total loss
            loss = seg_loss + gmm_loss + labled_proto_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), labeled_img.size()[0])
            seg_m.update(seg_loss.item(), labeled_img.size()[0])
            gmm_m.update(gmm_loss.item(), labeled_img.size()[0])

            iters += 1
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters:{:}, loss:{:.3f}, seg_loss:{:.3f}, '
                            'gmm_loss:{:.3f}'.format
                            (i, loss_m.avg, seg_m.avg, gmm_m.avg))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU, iou_class = semi_evaluate(model, valloader, eval_mode, cfg)

        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()
