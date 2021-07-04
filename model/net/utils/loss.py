import cv2
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.nn.functional import one_hot
from utils.box.bbox import bbox_switch, angle_switch, bbox_iou, encode, decode
from utils.box.ext.rotate_overlap_diff.oriented_iou_loss import  cal_iou, cal_diou, cal_giou
from utils.box.rbbox import rbbox_batched_nms as nms
from utils.utils import soft_weight

def iou_obb_diff(gts, preds, type='diou'):
    gt_bboxes = angle_switch(gts)
    pred_bboxes = angle_switch(preds)
    if type == 'riou':
        iou, *_ = cal_iou(gt_bboxes.unsqueeze(0), pred_bboxes.unsqueeze(0))
        linear = False
        if linear:
            iou_loss = 1 - iou
        else:
            iou_loss = - iou.clamp(min=1e-6).log()

    elif type in ['giou', 'diou']:
        riou_func = cal_giou if type == 'giou' else cal_diou
        iou_loss, iou = riou_func(gt_bboxes.unsqueeze(0), pred_bboxes.unsqueeze(0))
    else:
        raise NotImplementedError
    return iou, iou_loss


def match(bboxes_xyxy, anchors_xyxy, bboxes, anchors, iou_thresh, process=None, batch=32):
    # Reduce GPU memory usage
    ious = torch.cat([bbox_iou(bboxes_xyxy[i: i + batch], anchors_xyxy) for i in range(0, bboxes_xyxy.size(0), batch)])
    max_ious, bbox_indexes = torch.max(ious, dim=0)
    mask_neg = max_ious < iou_thresh[0]
    mask_pos = max_ious > iou_thresh[1]
    max_gt, argmax_gt = torch.max(ious, dim=1)

    if (max_gt <= iou_thresh[1]).any():
        mask_pos[argmax_gt[max_gt <= iou_thresh[1]]] = True
        mask_neg[argmax_gt[max_gt <= iou_thresh[1]]] = False
    
    pnms_thres = soft_weight(process)
    r_anchors = torch.cat([anchors, torch.zeros_like(anchors[:,0]).unsqueeze(1)], -1)
    scores = iou_obb_diff(bboxes[bbox_indexes[mask_pos]], r_anchors[mask_pos], type='riou')[0].squeeze(0)
    labels = torch.zeros_like(scores)
    keeps = nms(r_anchors[mask_pos], scores, labels, pnms_thres)[:500]
    mask_keep = mask_pos.nonzero()[keeps]
    mask_pos = torch.zeros_like(mask_pos)
    mask_pos[mask_keep] = True

    iou_balance = True
    num_pos = mask_pos.sum().item()
    if not iou_balance:
        ratio = 1  # neg2pos
        num_neg = ratio * num_pos
        neg_indices = mask_neg.nonzero().squeeze()
        sampled_neg_indices = np.random.choice(neg_indices.cpu(), size=num_neg)
        mask_neg.fill_(False)[sampled_neg_indices] = True
    else:
        ratio_hard = 2  # hard2pos
        ratio_bg = 100    # bg2pos
        num_hard = ratio_hard * num_pos
        num_bg = ratio_bg * num_pos
        hard_indices = ((max_ious > 0.1) & (max_ious < iou_thresh[0])).nonzero().squeeze()
        bg_indices = (max_ious < 1e-2).nonzero().squeeze()
        sampled_hard_indices = np.random.choice(hard_indices.cpu(), size=num_hard)
        sampled_bg_indices = np.random.choice(bg_indices.cpu(), size=num_bg)
        sampled_neg_indices = np.concatenate([sampled_bg_indices, sampled_hard_indices])
        mask_neg.fill_(False)[sampled_neg_indices] = True

    return mask_pos, mask_neg, bbox_indexes





def calc_loss(pred_cls, pred_loc, targets, anchors, iou_thresh, variance, balance, process=None):
    device = pred_cls.device
    num_classes = pred_cls.size(-1)
    weight_pos, weight_neg = 2 * balance, 2 * (1 - balance)
    anchors_xyxy = bbox_switch(anchors, 'xywh', 'xyxy')

    criterion_cls = nn.BCEWithLogitsLoss(reduction='none')
    criterion_loc = nn.SmoothL1Loss(reduction='sum')
    loss_cls, loss_loc = torch.zeros([2], dtype=torch.float, device=device, requires_grad=True)
    num_pos = 0
    for i, target in enumerate(targets):
        if target:
            bboxes = target['bboxes'].to(device)
            labels = target['labels'].to(device)
            bboxes_xyxy = bbox_switch(bboxes[:, :4], 'xywh', 'xyxy')
            pred_box = decode(pred_loc[i], anchors, variance)

            mask_pos, mask_neg, bbox_indexes = match(bboxes_xyxy, anchors_xyxy, bboxes, anchors, iou_thresh, process=process)

            labels = labels[bbox_indexes]
            indexes_pos = bbox_indexes[mask_pos]
            bboxes_matched = bboxes[indexes_pos]
            anchors_matched = anchors[mask_pos]
            bboxes_pred = pred_loc[i][mask_pos] # offsets
            gt_bboxes, det_bboxes = encode(bboxes_matched, bboxes_pred, anchors_matched, variance)

            labels = one_hot(labels, num_classes=num_classes).float()
            labels[mask_neg] = 0
            loss_cls_ = criterion_cls(pred_cls[i], labels)
            loss_cls = loss_cls + loss_cls_[mask_pos].sum() * weight_pos + loss_cls_[mask_neg].sum() * weight_neg
            use_iou = False
            if use_iou:
                rious, riou_loss = iou_obb_diff(bboxes_matched, pred_box[mask_pos])
                loss_loc = loss_loc + riou_loss.sum()
            else:
                loss_loc = loss_loc + criterion_loc(gt_bboxes, det_bboxes)
            num_pos += mask_pos.sum().item()
        else:
            loss_cls = loss_cls + criterion_cls(pred_cls[i], torch.zeros_like(pred_cls[i])).sum()
    num_pos = max(num_pos, 1)
    return OrderedDict([('loss_cls', loss_cls / num_pos), ('loss_loc', loss_loc / num_pos)])

