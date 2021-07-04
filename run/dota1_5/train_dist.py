import sys
sys.path.append('.')

import os
import tqdm
import torch
import random
import tempfile
import argparse
import numpy as np
import multiprocessing

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.aug.compose import Compose
from data.aug import ops
from data.dataset import DOTA1_5

from model.net import Net
from model.backbone import resnet

from utils.utils import hyp_parse
from utils.adjust_lr import adjust_lr_multi_step

from torch import distributed as dist
from torch.nn import SyncBatchNorm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(rank, world_size):

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    set_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet101

    data_dir = 'data/DOTA1_5'
    dir_save = 'weights'
    hyp = 'run/dota1_5/hyp.py'

    hyps = hyp_parse(hyp)

    dir_weight = os.path.join(dir_save, 'dota1_5_weight')
    dir_log = os.path.join(dir_save, 'dota1_5_log')
    os.makedirs(dir_weight, exist_ok=True)
    if rank == 0:
        writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    lr = hyps['lr']
    image_size = int(hyps['image_size'])
    max_step = int(hyps['max_step'])
    lr_cfg = [[0.7 * max_step, lr], [0.9 * max_step, lr / 10], [max_step, lr / 50]]
    warm_up = [1000, lr / 50, lr]
    save_interval = hyps['save_interval']

    aug = Compose([
        ops.ToFloat(),
        ops.PhotometricDistort(),
        ops.RandomHFlip(),
        ops.RandomVFlip(),
        ops.RandomRotate90(),
        ops.ResizeJitter([0.8, 1.2]),
        ops.PadSquare(),
        ops.Resize(image_size),
        ops.BBoxFilter(5 * 5 * 0.4)
    ])

    dataset = DOTA1_5(data_dir, ['train', 'val'], aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size,
     rank)
    batch_sampler = torch.utils.data.BatchSampler(train_sampler, int(hyps['batch_size']), drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,\
        num_workers=int(hyps['num_workers']), pin_memory=False, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }


    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
    }
    device = torch.device(f'cuda:{rank}')
    model = Net(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if current_step:
        model.module.load_state_dict(torch.load(os.path.join(dir_weight, '%d.pth' % current_step), map_location=device))
    else:
        checkpoint = os.path.join(tempfile.gettempdir(), "initial-weights.pth")
        if rank == 0:
            model.module.init()
            torch.save(model.module.state_dict(), checkpoint)
        dist.barrier()
        if rank > 0:
            model.module.load_state_dict(torch.load(checkpoint, map_location=device))
        dist.barrier()
        if rank == 0:
            os.remove(checkpoint)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=5e-4)
    

    training = True
    while training and current_step < max_step:
        tqdm_loader = tqdm.tqdm(loader) if rank == 0 else loader
        for images, targets, infos in tqdm_loader:
            current_step += 1
            adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            process = current_step / max_step
            images = images.cuda() / 255
            losses = model(images, targets, process=process)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                for key, val in list(losses.items()):
                    losses[key] = val.item()
                    writer.add_scalar(key, val, global_step=current_step)
                writer.flush()
                tqdm_loader.set_postfix(losses)
                tqdm_loader.set_description(f'<{current_step}/{max_step}>')

                if current_step % save_interval == 0:
                    save_path = os.path.join(dir_weight, '%d.pth' % current_step)
                    state_dict = model.module.state_dict()
                    torch.save(state_dict, save_path)
                    cache_file = os.path.join(dir_weight, '%d.pth' % (current_step - save_interval))

            if current_step >= max_step:
                training = False
                if rank == 0:
                    writer.close()
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed training for DOTA1_5 dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()


    multiprocessing.set_start_method('spawn')

    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(int(eval(args.gpus)))]

    processes = []
    for device_id in device_ids:
        p = multiprocessing.Process(target=main, args=(device_id, len(device_ids)))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
