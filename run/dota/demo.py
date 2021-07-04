import sys

sys.path.append('.')

import os
import math
import tqdm
import torch
import numpy as np
import cv2
import argparse

from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import DOTA

from model.net import Net
from model.backbone import resnet
from utils.parallel import CustomDetDataParallel


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
            + cal_line_length(combinate[i][1], dst_coordinate[1]) \
            + cal_line_length(combinate[i][2], dst_coordinate[2]) \
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def draw_caption(image, box, caption, color, class_):

    label = str(class_) + str('%.2f' % caption)
    fontScale = 0.7
    font = cv2.FONT_HERSHEY_COMPLEX
    thickness = 1
    t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
    c1 = tuple(box[:2].astype('int'))
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
    cv2.rectangle(image, c1, c2, color, -1)  # filled
    cv2.putText(image, label, (c1[0], c1[1] - 5), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)


def rbox2rect(polys):
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1)

def get_ratios(dataset, img, base, scale=1.0):
    ratios = np.array([img.shape[1] / base, img.shape[0] / base])
    return ratios

@torch.no_grad()
def main():

    image_size = 768
    batch_size = 1
    num_workers = 1

    aug = ops.Resize(image_size)
    dataset = DOTA(data_dir, 'test', aug)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }

    conf_thresh = 0.5
    nms_thresh = 0.1
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
    for images, _, info in tqdm.tqdm(loader):
        images = images.cuda() / 255
        dets = model(images)
        for _, det in zip(_, dets):
            if det:
                bboxes, scores, labels = list(map(lambda x: x.cpu().numpy(), det))
                det_list.extend([bbox, score, label] for bbox, score, label in zip(bboxes, scores, labels))
            count += 1
        det = np.array(det_list, dtype=object)
        img_path = str(info[0]['img_path'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if len(det):
            class_label = det[:, 2]
            score_label = det[:, 1]
            bbox_label = det[:, 0]
        else:
            continue
        
        # ratios
        ratios = get_ratios(data_dir, img, image_size)

        for cls, score, bbox in zip(class_label, score_label, bbox_label):
            # color index
            bbox_color = colormap[cls]
            bbox_class = classmap[cls]

            # draw bbox
            bbox = np.array(xywha2xy4(bbox) * ratios, dtype=np.float32)  # rbox2quad
            bbox = cv2.boxPoints(cv2.minAreaRect(bbox))  # quad2Good_quad
            bbox = bbox.astype(np.int32)  # float2int
            img = cv2.polylines(img, [bbox], True, bbox_color, 2)  # 后三个参数为：是否封闭/color/thickness

            # draw caption
            bbox = get_best_begin_point_single(bbox.reshape(8, -1))
            draw_caption(img, bbox, score, bbox_color, bbox_class)

        out_path = os.path.join(dir_output, img_path.split('/')[-1])
        cv2.imwrite(out_path, img)
        det_list = []  # clear det list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detecting...')
    parser.add_argument('--ckpt', help='checkpoint')
    args = parser.parse_args()

    device_ids = [0]
    torch.cuda.set_device(device_ids[0])
    backbone = resnet.resnet101

    data_dir = 'data/DOTA'
    dir_save = 'weights/dota_weight'

    dir_output = 'outputs'  
    os.system('rm {}'.format(dir_output + '/*'))
    os.makedirs(dir_output, exist_ok=True)

    checkpoint = os.path.join(dir_save, args.ckpt)

    classmap = ['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                     'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court']

    colormap = [
        (0, 255, 0),
        (54, 67, 244),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121)]

    main()
