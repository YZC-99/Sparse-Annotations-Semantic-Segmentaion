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

from dataset.sass import *
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
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

#nih

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(os.path.join('/root/autodl-tmp/Saprse-Annotations-Semantic-Segmentaion',args.config), "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    logger.info('{}\n'.format(pprint.pformat(cfg)))

    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg, aux=cfg['aux'])

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(local_rank)

    ohem = False if cfg['criterion']['name'] == 'CELoss' else True
    use_weight = False

    trainset = DrishtiDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                           cfg['crop_size'], cfg['aug'])
    valset = DrishtiDataset(cfg['dataset'], cfg['data_root'], 'val', None)

#     trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True)
#     valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0

    for epoch in range(cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        loss_m = AverageMeter()
        seg_m = AverageMeter()
        gmm_m = AverageMeter()

#         trainsampler.set_epoch(epoch)

        for i, (img, mask, cls_label, id) in enumerate(trainloader):
            img, mask, cls_label = img.cuda(), mask.cuda(), cls_label.cuda()
            feat, pred = model(img)

            loss = loss_calc(pred, mask,
                                 ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters:{:}, loss:{:.3f}'.format
                            (i, loss_m.avg))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()
