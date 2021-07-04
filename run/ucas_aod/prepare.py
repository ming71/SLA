import sys

sys.path.append('.')

import os
import json
import argparse
import cv2 as cv
import numpy as np

from utils.crop_image import Cropper


def txt2json(dir_txt, dir_json):
    os.makedirs(dir_json, exist_ok=True)
    for txt in os.listdir(dir_txt):
        objs = []
        name = os.path.splitext(txt)[0]
        for line in open(os.path.join(dir_txt, txt)).readlines():
            category, *bbox = line.split()[:9]
            bbox = np.array(bbox, dtype=np.float32).reshape([4, 2])
            bbox = cv.boxPoints(cv.minAreaRect(bbox))
            bbox = bbox.tolist()
            obj = dict()
            obj['name'] = category
            obj['bbox'] = bbox
            objs.append(obj)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, name + '.json'), 'wt'), indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UCAS-AOD preparation...')
    parser.add_argument('dir_dataset', help='the dir of dataset')
    args = parser.parse_args()

    print('convert annotation files')
    dir_txt = os.path.join(args.dir_dataset, 'Annotations')
    dir_anno = os.path.join(args.dir_dataset, 'annotations')
    txt2json(dir_txt, dir_anno)

    print('generate image-sets files')
    out_dir = os.path.join(args.dir_dataset, 'image-sets')
    os.makedirs(out_dir, exist_ok=True)
    for image_set in ['train', 'val', 'test']:
        pairs = []
        with open(os.path.join(args.dir_dataset, 'ImageSets', f'{image_set}.txt')) as f:
            for line in f.readlines():
                line = line.strip()
                img = os.path.join('AllImages', f'{line}.png')
                anno = os.path.join('annotations', f'{line}.json')
                if not os.path.exists(os.path.join(args.dir_dataset, anno)):
                    anno = None
                pairs.append([img, anno])
        json.dump(pairs, open(os.path.join(out_dir, f'{image_set}.json'), 'wt'), indent=2)