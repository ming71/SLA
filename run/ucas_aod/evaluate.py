import sys

sys.path.append('.')

import os
import tqdm
import torch
import argparse
import cv2
import numpy as np

from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import UCAS_AOD

from model.net import Net
from model.backbone import resnet

from utils.box.bbox_np import xy42xywha, xywha2xy4
from utils.box.metric import get_det_aps
from utils.parallel import CustomDetDataParallel


@torch.no_grad()
def main():
    global checkpoint
    if checkpoint is None:
        dir_weight = os.path.join(dir_save, 'ucas_weight')
        indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
        current_step = max(indexes)
        checkpoint = os.path.join(dir_weight, '%d.pth' % current_step)

    image_size = 768
    batch_size = 8
    num_workers = 4

    aug = ops.Resize(image_size)
    
    dataset = UCAS_AOD(data_dir, 'test', aug)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }
    conf_thresh = 0.05
    nms_thresh = 0.2
    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
        'conf_thresh': conf_thresh,
        'nms_thresh': nms_thresh,
    }

    model = Net(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model.restore(checkpoint)
    if len(device_ids) > 1:
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    model.eval()

    count = 0
    gt_list, det_list = [], []
    for images, targets, infos in tqdm.tqdm(loader):
        images = images.cuda() / 255
        dets = model(images)
        for target, det, info in zip(targets, dets, infos):
            if target:
                bboxes = np.stack([xy42xywha(bbox) for bbox in info['objs']['bboxes']])
                labels = info['objs']['labels']
                gt_list.extend([count, bbox, 1, label] for bbox, label in zip(bboxes, labels))
            if det:
                ih, iw = info['shape'][:2]
                bboxes, scores, labels = list(map(lambda x: x.cpu().numpy(), det))
                bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                bboxes_ = bboxes * [iw / image_size, ih / image_size]
                # bboxes = np.stack([xy42xywha(bbox) for bbox in bboxes_])
                bboxes = []
                for bbox in bboxes_.astype(np.float32):
                    (x, y), (w, h), a = cv2.minAreaRect(bbox)
                    bboxes.append([x, y, w, h, a])
                bboxes = np.array(bboxes)
                det_list.extend([count, bbox, score, label] for bbox, score, label in zip(bboxes, scores, labels))
            count += 1
    APs = get_det_aps(det_list, gt_list, num_classes, use_07_metric=use_07_metric)
    mAP = sum(APs) / len(APs)
    print('AP')
    for label in range(num_classes):
        print(f'{dataset.label2name[label]}: {APs[label]}')
    print(f'mAP: {mAP}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation for UCAS-AOD dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    parser.add_argument('--ckpt', help='checkpoint')
    parser.add_argument('--use_voc07', help='voc07 or voc10 metric')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet50

    data_dir = 'data/UCAS_AOD'
    dir_save = 'weights/ucas_weight'
    

    checkpoint = os.path.join(dir_save, args.ckpt)
    use_07_metric = args.use_voc07


    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(eval(args.gpus))]

    main()

