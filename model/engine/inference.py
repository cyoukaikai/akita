import json
import os

import torch
from tqdm import tqdm

from model.utils.postprocess import decode_predictions
from torchvision.ops import nms

def inference(args, cfg, model, eval_loader):
    results = []
    count = 0
    with torch.inference_mode():
        for images in tqdm(eval_loader):
            features, predictions = model(images)
            if args.visualize_feature:
                for b in range(args.batch_size):
                    for j in range(len(features)):
                        save_path = os.path.join(args.output_dirname, 'visualize', '{}_{}.png'.format(count, j))
                        visualize_feat(features[j][b], save_path)

            if args.save_feature:
                for b in range(args.batch_size):
                    for j in range(len(features)):
                        save_path = os.path.join(args.output_dirname, 'feature', '{}_{}.npy'.format(count, j))
                        save_feat(features[j][b], save_path)

            count += 1

            boxes, labels, scores = decode_predictions(predictions, cfg.TEST.MAX_OBJECTS, cfg.MODEL.DOWN_RATIOS, args.batch_size)

            for i in range(args.batch_size):
                if cfg.TEST.NMS:
                    keep = nms(boxes[i], scores[i], cfg.TEST.NMS_THRESHOLD)
                    results.append([boxes[i, keep, :].tolist(), labels[i, keep].tolist(), scores[i, keep].tolist()])
                else:
                    results.append([boxes[i].tolist(), labels[i].tolist(), scores[i].tolist()])

    return results

def coco_evaluation(dataset, predictions, output_dir):
    coco_results = []
    for i, pred in enumerate(predictions):
        boxes, labels, scores = pred
        image_id, annotation = dataset.get_annotation(i)
        class_mapper = dataset.contiguous_id_to_coco_id

        if len(labels) == 0:
            # print('continue')
            continue

        coco_results.extend(
            [
                {
                    'image_id': image_id,
                    'category_id': class_mapper[labels[k]+1],
                    'bbox': [box[0], box[1], (box[2] - box[0]), (box[3] - box[1])],
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    iou_type = 'bbox'
    json_result_file = os.path.join(output_dir, iou_type + ".json")
    os.makedirs(os.path.dirname(json_result_file), exist_ok=True)
    print('Writing resuls to {}...'.format(json_result_file))
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


import cv2
import numpy as np
from model.data.transforms.transforms import ToNumpy

def visualize_feat(feat, save_path):
    transform = ToNumpy()

    feat, _, _ = transform(feat[0].detach().cpu().unsqueeze(0))
    feat = (feat - feat.min()) / (feat.max() - feat.min())
    feat = feat * 255
    feat = feat.astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, feat)

def save_feat(feat, save_path):
    transform = ToNumpy()

    feat, _, _ = transform(feat.detach().cpu())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, feat)