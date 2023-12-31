from dataset.sass import *
from dataset.transform import normalize_back
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import *
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MODE = None


def parse_args():
    name = 'semi-refuge400'
    mode = 'semi'

    parser = argparse.ArgumentParser(description='SASS Framework')
    parser.add_argument('--resume_model', type=str,
                        default='./exp/%s/resnet50_86.14.pth' % (name))
    parser.add_argument('--config', type=str, default='./configs/%s' % (name + '.yaml'))
    parser.add_argument('--save-mask-path', type=str, default='exp/%s/predicts' % (name + '/' + mode))
    args = parser.parse_args()
    return args


def get_dataset(cfg):
    if cfg['dataset'] == 'REFUGE/400semi':
        valset = DrishtiDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'cityscapes':
        valset = CityDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    else:
        valset = None

    return valset


def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = DeepLabV3Plus(cfg, aux=False)
    checkpoint = torch.load(args.resume_model)
    model.load_state_dict(checkpoint, strict=False)
    print('\nParams: %.1fM' % count_params(model))
    model = model.cuda()
    model.eval()

    if not os.path.exists(args.save_mask_path):
        os.makedirs(args.save_mask_path)

    dataset = get_dataset(cfg)
    valloader = DataLoader(dataset, batch_size=1,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    tbar = tqdm(valloader)
    background_binary_jaccard = JaccardIndex(num_classes=2, task='binary', average='micro').to('cuda:0')
    od_binary_jaccard = JaccardIndex(num_classes=2, task='binary', average='micro').to('cuda:0')
    oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary', average='micro').to('cuda:0')
    cmap = color_map(cfg['dataset'])

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            mask = mask.cuda()
            if cfg['dataset'] == 'cityscapes':
                pred = pre_slide(model, img, num_classes=cfg['nclass'],
                                 tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)
            else:
                pred = ms_test(model, img)

            pred = torch.argmax(pred, dim=1)

            background_preds = copy.deepcopy(pred)
            background_y = copy.deepcopy(mask)
            background_preds[background_preds == 0] = 3
            background_y[background_y == 0] = 3
            background_preds[background_preds != 3] = 0
            background_y[background_y != 3] = 0
            background_preds[background_preds == 3] = 1
            background_y[background_y == 3] = 1



            od_preds = copy.deepcopy(pred)
            od_y = copy.deepcopy(mask)
            od_preds[od_preds != 1] = 0
            od_y[od_y != 1] = 0

            oc_preds = copy.deepcopy(pred)
            oc_y = copy.deepcopy(mask)
            oc_preds[oc_preds != 2] = 0
            oc_preds[oc_preds != 0] = 1
            oc_y[oc_y != 2] = 0
            oc_y[oc_y != 0] = 1

            # 计算 od_cover_oc
            od_cover_gt = od_y + oc_y
            od_cover_gt[od_cover_gt > 0] = 1
            od_cover_preds = od_preds + oc_preds
            od_cover_preds[od_cover_preds > 0] = 1

            od_binary_jaccard.update(od_cover_preds,od_cover_gt)
            background_binary_jaccard.update(background_preds,background_y)
#             od_binary_jaccard.update(od_preds,od_y)
            oc_binary_jaccard.update(oc_preds,oc_y)
            background_IOU = background_binary_jaccard.compute()
            od_IOU = od_binary_jaccard.compute()
            oc_IOU = oc_binary_jaccard.compute()

            mIOU = (od_IOU + oc_IOU + background_IOU) / 3
            mIOU = mIOU

            tbar.set_description('OD_IOU: %.2f --OC_IOU:%.2f--mIOU: %.2f'% (od_IOU * 100.0,oc_IOU * 100.0,mIOU * 100.0))

            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred = Image.fromarray(pred, mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1])))
    print("OD_IOU: %.2f --OC_IOU:%.2f"% (od_IOU * 100.0,oc_IOU * 100.0))
    mIOU *= 100.0


if __name__ == '__main__':
    args = parse_args()

    print()
    print(args)

    main(args)
