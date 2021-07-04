import sys
sys.path.append('.')

import os
import tqdm
import torch
import argparse

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.aug.compose import Compose
from data.aug import ops
from data.dataset import HRSC2016

from model.net import Net
from model.backbone import resnet

from utils.utils import hyp_parse
from utils.adjust_lr import adjust_lr_multi_step
from utils.parallel import convert_model, CustomDetDataParallel


def main():
    dir_weight = os.path.join(dir_save, 'hrsc_weight')
    dir_log = os.path.join(dir_save, 'hrsc_log')
    os.makedirs(dir_weight, exist_ok=True)
    writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    lr = hyps['lr']
    image_size = int(hyps['image_size'])
    max_step = int(hyps['max_step'])
    lr_cfg = [[0.8 * max_step, lr], [max_step, lr / 10]]
    warm_up = [500, lr / 50, lr]
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
    ])
    dataset = HRSC2016(data_dir, ['trainval'], aug)
    loader = DataLoader(dataset, int(hyps['batch_size']), shuffle=True, num_workers=int(hyps['num_workers']), pin_memory=True, drop_last=True,
                        collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1.5, 3, 5, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }

    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
    }

    model = Net(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    if current_step:
        model.restore(os.path.join(dir_weight, '%d.pth' % current_step))
    else:
        model.init()
    if len(device_ids) > 1:
        model = convert_model(model)
        model = CustomDetDataParallel(model, device_ids)

    model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=5e-4)

    training = True
    while training and current_step < max_step:
        tqdm_loader = tqdm.tqdm(loader)
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

            for key, val in list(losses.items()):
                losses[key] = val.item()
                writer.add_scalar(key, val, global_step=current_step)
            writer.flush()
            tqdm_loader.set_postfix(losses)
            tqdm_loader.set_description(f'<{current_step}/{max_step}>')

            if current_step % save_interval == 0:
                save_path = os.path.join(dir_weight, '%d.pth' % current_step)
                state_dict = model.state_dict() if len(device_ids) == 1 else model.module.state_dict()
                torch.save(state_dict, save_path)

            if current_step >= max_step:
                training = False
                writer.close()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for HRSC2016 dataset...')
    parser.add_argument('--gpus', help='num of gpus')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    backbone = resnet.resnet101

    data_dir = 'data/HRSC2016'
    dir_save = 'weights'
    hyp = 'run/hrsc2016/hyp.py'

    hyps = hyp_parse(hyp)


    if ',' in args.gpus:
        device_ids = [eval(x) for x in args.gpus.split(',') if len(x)!=0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in device_ids])
        device_ids = list(range(len(device_ids)))
    else:
        device_ids = [x for x in range(int(eval(args.gpus)))]

    main()
