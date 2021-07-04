import sys

sys.path.append('.')

import os
import json
import cv2 
import argparse
import numpy as np
from tqdm import tqdm
from utils.crop_image import Cropper


def txt2json(dir_txt, dir_json):
    os.makedirs(dir_json, exist_ok=True)
    pbar = tqdm(os.listdir(dir_txt))
    for file in pbar:
        pbar.set_description('convert annotation: %s' % file)
        objs = []
        for i, line in enumerate(open(os.path.join(dir_txt, file)).readlines()):
            line = line.strip()
            line_split = line.split(' ')
            if len(line_split) == 10:
                obj = dict()
                coord = np.array(line_split[:8], dtype=np.float32).reshape([4, 2])
                bbox = cv2.boxPoints(cv2.minAreaRect(coord)).astype(np.int).tolist()
                obj['name'] = line_split[8].lower()
                obj['bbox'] = bbox
                objs.append(obj)
            else:
                continue
                # print('<skip line> %s' % line)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, file.replace('txt', 'json')), 'wt'), indent=2)


def main(image_set, single_scale=False):
    print('\n \nprocessing {}'.format(image_set))
    # convert annotation files
    if image_set not in ['test', 'demo']:
        dir_txt = os.path.join(dir_dataset, 'labelTxt', image_set)
        out_dir_json = os.path.join(dir_dataset, 'annotations', image_set)
        txt2json(dir_txt, out_dir_json)

    # crop images
    print('crop images')
    pairs = []
    for filename in os.listdir(os.path.join(dir_dataset, 'images', image_set)):
        anno = os.path.join(dir_dataset, 'annotations', image_set, filename.replace('png', 'json'))
        img = os.path.join(dir_dataset, 'images', image_set, filename)
        if not os.path.exists(anno):
            anno = None
        pairs.append([img, anno])

    overlap = 0.25
    sizes = [768] if single_scale else [512, 768, 1024, 1536]
    save_empty = image_set in ['test', 'demo']
    image_set = f'{image_set}-{sizes[0]}' if single_scale else image_set

    out_dir_images = os.path.join(dir_dataset, 'images', f'{image_set}_crop')
    out_dir_annos = os.path.join(dir_dataset, 'annotations', f'{image_set}_crop')

    cropper = Cropper(sizes, overlap)
    cropper.crop_batch(pairs, out_dir_images, out_dir_annos, save_empty)

    # generate image-set files
    pairs = []
    pbar = tqdm(os.listdir(out_dir_images))
    for filename in pbar:
        pbar.set_description('generate image-set files')
        img = os.path.join('images', f'{image_set}_crop', filename)
        anno = None if image_set in ['test', 'demo'] else os.path.join('annotations', f'{image_set}_crop', filename.replace('jpg', 'json'))
        pairs.append([img, anno])
    out_dir = os.path.join(dir_dataset, 'image-sets')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(pairs, open(os.path.join(out_dir, f'{image_set}.json'), 'wt'), indent=2)


if __name__ == '__main__':

    # root/images/train/P0000.png
    # -----------/val/...
    # -----------/test/...

    # root/labelTxt/train/P0000.txt
    # -------------/val/...

    # (1) convert annotation files
    # (2) crop images
    # (3) generate image-set files

    parser = argparse.ArgumentParser(description='DOTA2 preparation...')
    parser.add_argument('dir_dataset', help='the dir of dataset')
    args = parser.parse_args()
    dir_dataset = args.dir_dataset

    main('train')
    main('val')
    main('test')
    main('test', True)

    # main('demo')
