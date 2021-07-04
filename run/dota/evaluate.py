import sys

sys.path.append('.')

import os
import tqdm
import torch
import shutil  
import argparse
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader

from data.aug import ops
from data.aug.compose import Compose
from data.dataset import DOTA

from model.net import Net
from model.backbone import resnet

from utils.box.bbox_np import xywha2xy4, xy42xywha
from utils.box.rbbox_np import rbbox_batched_nms
from utils.parallel import CustomDetDataParallel


@torch.no_grad()
def main():
    batch_size = 8
    num_workers = 4

    image_size = 768
    aug = Compose([ops.PadSquare(), ops.Resize(image_size)])
    dataset = DOTA(data_dir, image_set, aug)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }
    conf_thresh = 0.01
    nms_thresh = 0.45
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

    ret_raw = defaultdict(list)
    for images, targets, infos in tqdm.tqdm(loader):
        images = images.cuda() / 255
        dets = model(images)
        for (det, info) in zip(dets, infos):
            if det:
                bboxes, scores, labels = det
                bboxes = bboxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                fname, x, y, w, h = os.path.splitext(os.path.basename(info['img_path']))[0].split('-')[:5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                long_edge = max(w, h)
                pad_x, pad_y = (long_edge - w) // 2, (long_edge - h) // 2
                bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                bboxes *= long_edge / image_size
                bboxes -= [pad_x, pad_y]
                bboxes += [x, y]
                bboxes = np.stack([xy42xywha(bbox) for bbox in bboxes])
                ret_raw[fname].append([bboxes, scores, labels])
    import ipdb;ipdb.set_trace()

    print('merging results...')
    ret = []

    for fname, dets in ret_raw.items():
        bboxes, scores, labels = zip(*dets)
        bboxes = np.concatenate(list(bboxes))
        scores = np.concatenate(list(scores))
        labels = np.concatenate(list(labels))
        keeps = rbbox_batched_nms(bboxes, scores, labels, nms_thresh)
        ret.append([fname, [bboxes[keeps], scores[keeps], labels[keeps]]])

    print('converting to submission format...')
    ret_save = defaultdict(list)
    for fname, (bboxes, scores, labels) in ret:
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = xywha2xy4(bbox).ravel()
            line = '%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f' % (fname, score, *bbox)
            ret_save[dataset.label2name[label]].append(line)

    print('saving...')
    os.makedirs('submission', exist_ok=True)
    for name, dets in ret_save.items():
        with open(os.path.join('submission', 'Task%d_%s.txt' % (1, name)), 'wt') as f:
            f.write('\n'.join(dets))

    print('creating submission...')
    if os.path.exists('Task1.zip'):
        os.remove('Task1.zip')
    os.system('zip -j Task1.zip {}'.format('submission/*'))
    shutil.rmtree('submission')  
    print('finished')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation for HRSC2016 dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    parser.add_argument('--ckpt', help='checkpoint')
    parser.add_argument('--use_voc07', help='voc07 or voc10 metric')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet101

    data_dir = 'data/DOTA'
    dir_save = 'weights/dota_weight'

    checkpoint = os.path.join(dir_save, args.ckpt)

    image_set = 'test' 
    # image_set = 'test-768' 

    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(eval(args.gpus))]

    main()

