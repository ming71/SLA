import sys

sys.path.append('.')

import os
import tqdm
import torch
import random
import shutil  
import argparse
import numpy as np
import multiprocessing

from collections import defaultdict
from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import DOTA
from data.aug.compose import Compose
from data.dataset.dota import NAMES

from model.net import Net
from model.backbone import resnet

from utils.utils import hyp_parse
from utils.box.bbox_np import xywha2xy4, xy42xywha
from utils.box.rbbox_np import rbbox_batched_nms
from utils.parallel import CustomDetDataParallel

from torch import distributed as dist
from torch.nn import SyncBatchNorm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def main(args, rank, world_size, res):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    set_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet101
    batch_size = 8
    num_workers = 4

    image_size = 768
    data_dir = 'data/DOTA'
    dir_save = 'weights/dota_weight'
    image_set = 'test' 

    checkpoint = os.path.join(dir_save, args.ckpt)

    aug = Compose([ops.PadSquare(), ops.Resize(image_size)])
    dataset = DOTA(data_dir, image_set, aug)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size,
     rank)
    batch_sampler = torch.utils.data.BatchSampler(test_sampler, batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,\
             num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }
    conf_thresh = 0.01
    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
        'conf_thresh': conf_thresh,
    }

    device = torch.device(f'cuda:{rank}')
    model = Net(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.module.load_state_dict(torch.load(checkpoint, map_location=device))
    
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
    res.update(ret_raw)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Distributed evaluation for DOTA dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    parser.add_argument('--ckpt', help='checkpoint')
    parser.add_argument('--use_voc07', help='voc07 or voc10 metric')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    nms_thresh = 0.45

    multiprocessing.set_start_method('spawn')


    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(int(eval(args.gpus)))]


    res = multiprocessing.Manager().dict()

    processes = []
    for device_id in device_ids:
        p = multiprocessing.Process(target=main, args=(args, device_id, len(device_ids), res))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    print('merging results...')
    ret = []
    for fname, dets in res.items():
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
            ret_save[NAMES[label]].append(line)

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
