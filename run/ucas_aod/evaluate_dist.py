import sys
sys.path.append('.')

import os
import cv2
import tqdm
import torch
import random
import tempfile
import argparse
import numpy as np
import multiprocessing

from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import UCAS_AOD
from data.dataset.ucas_aod import NAMES

from model.net import Net
from model.backbone import resnet

from utils.utils import hyp_parse
from utils.box.bbox_np import xy42xywha, xywha2xy4
from utils.box.metric import get_det_aps
from utils.parallel import CustomDetDataParallel


from torch import distributed as dist
from torch.nn import SyncBatchNorm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()


def main(args, rank, world_size, res, gts):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    set_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet50

    image_size = 768
    batch_size = 4
    num_workers = 4
    data_dir = 'data/UCAS_AOD'
    dir_save = 'weights/ucas_weight'

    checkpoint = os.path.join(dir_save, args.ckpt)

    aug = ops.Resize(image_size)
    dataset = UCAS_AOD(data_dir, 'test', aug)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size,
     rank)
    batch_sampler = torch.utils.data.BatchSampler(test_sampler, batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,\
             num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2]] * 5,
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

    device = torch.device(f'cuda:{rank}')
    model = Net(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.module.load_state_dict(torch.load(checkpoint, map_location=device))
    
    model.eval()

    count = 0
    gt_list, det_list = [], []
    tqdm_loader = tqdm.tqdm(loader) if rank == 0 else loader
    for images, targets, infos in tqdm_loader:
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
    

    res.extend(det_list)
    gts.extend(gt_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Distributed evaluation for UCAS_AOD dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    parser.add_argument('--ckpt', help='checkpoint')
    parser.add_argument('--use_voc07', help='voc07 or voc10 metric')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn')

    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(int(eval(args.gpus)))]


    res = multiprocessing.Manager().list()
    gts = multiprocessing.Manager().list()

    processes = []
    for device_id in device_ids:
        p = multiprocessing.Process(target=main, args=(args, device_id, len(device_ids), res, gts))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    num_classes = len(NAMES)
    APs = get_det_aps(res, gts, num_classes, use_07_metric=args.use_voc07)
    mAP = sum(APs) / len(APs)
    print('AP')
    for name, ap in zip(NAMES, APs):
        print(f'{name}: {ap}')
    print(f'mAP: {mAP}')

