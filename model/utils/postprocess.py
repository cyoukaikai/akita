import torch

import torch.nn as nn

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
    

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=512):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _maxpool(heatmap, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(predictions, max_objects, down_ratios, batch_size):
    boxes = torch.empty(batch_size, 0, 4).to('cuda')
    scores = torch.empty(batch_size, 0).to('cuda')
    classes = torch.empty(batch_size, 0).to('cuda')

    for i in range(len(predictions)):
        if predictions[i] is None:
            continue

        heatmap = predictions[i][-1]['hm']
        wh = predictions[i][-1]['wh']
        reg = predictions[i][-1]['reg']

        heatmap = _maxpool(heatmap)
        score, inds, categorie, ys, xs, = _topk(heatmap)

        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch_size, 512, 2)
        xs = xs.view(batch_size, 512, 1) + reg[:, :, 0:1]
        ys = ys.view(batch_size, 512, 1) + reg[:, :, 1:2]

        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch_size, 512, 2)

        xs *= down_ratios[i]
        ys *= down_ratios[i]
        wh *= down_ratios[i]

        classes = torch.cat([classes, categorie.view(batch_size, 512).float()], dim=1)
        scores = torch.cat([scores, score], dim=1)
        boxes = torch.cat((boxes, torch.cat([xs - wh[..., 0:1] / 2,
                                             ys - wh[..., 1:2] / 2,
                                             xs + wh[..., 0:1] / 2,
                                             ys + wh[..., 1:2] / 2,], dim=2)), dim=1)

    scores, inds = torch.topk(scores, k=max_objects)
    classes = classes.gather(1, inds)
    boxes = boxes.gather(1, torch.stack([inds for _ in range(4)], dim=2))

    return boxes, classes, scores

try:
    # raise ImportError()
    import torch_extension
    print('C_nms')

    _nms = torch_extension.nms
except ImportError:
    from .python_nms import python_nms
    print('python_nms')

    _nms = python_nms

def nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(not support relative coordinates),
            shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    keep = _nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep

