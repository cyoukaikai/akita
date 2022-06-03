import cv2
import torch

from .transforms import *

class TrainTransform:
    def __init__(self, cfg, down_ratio, box_threshold):
        self.augment = Compose([
            ConvertFromInts(),
            RandomResize(resize_range=cfg.SOLVER.DATA.RESIZE_RANGE),
            RandomCrop(size=[int(x*down_ratio) for x in cfg.SOLVER.DATA.HEATMAP_SIZE], box_threshold=box_threshold),  # HEATMAP_SIZE = (128, 128)
            RandomMirror(),
            PhotometricDistort(),
            Clip(),
            Normalize(cfg.INPUT.MEAN, cfg.INPUT.STD),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.augment(image, boxes, labels)
        return image, boxes, labels

class DummyTransform:
    def __call__(self, image, boxes, labels):
        return torch.zeros(1), None, None

class DummyTargetTransform:
    def __call__(self, image, boxes, labels):
        return {'dummy': torch.zeros(1)}

class EvalTransform:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            ShapingCrop(),
            Normalize(cfg.INPUT.MEAN, cfg.INPUT.STD),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.augment(image, boxes, labels)
        return image, boxes, labels
    