import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import combinations

from model.modeling.extractor import build_extractors
from model.modeling.detector import build_detector
from model.modeling.discriminator import build_discriminator
from model.engine.loss_functions import DetectorLoss, DiscriminatorLoss
from model.utils.misc import fix_weights

from copy import copy
import os


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.extractors = build_extractors(cfg)
        self.detector = build_detector(cfg)
        self.loss_fn = DetectorLoss(cfg)

        self.discriminator = build_discriminator(cfg)
        self.d_loss_fn = DiscriminatorLoss(cfg)

        if cfg.SOLVER.IMAGENET_PRETRAINED:
            self._load_imagenet_pretrained_model(cfg.SOLVER.IMAGENET_PRETRAINED_MODEL)

        if cfg.SOLVER.PRETRAINED:
            self._load_pretrained_model(cfg.SOLVER.PRETRAINED_MODEL)

        if cfg.SOLVER.EXTRACTOR.WEIGHT_FIX:
            fix_weights(self.extractors[0])
        if cfg.SOLVER.DETECTOR.WEIGHT_FIX:
            fix_weights(self.detector)

        self.valid_scale = cfg.MODEL.VALID_SCALE
        self.mixed_precision = cfg.MIXED_PRECISION
        self.gp = cfg.SOLVER.DISCRIMINATOR.GP
        self.gp_weight = cfg.SOLVER.DISCRIMINATOR.GP_WEIGHT  # ?

        self.temp = 0

        # ------------------------
        self.discriminator_masking = cfg.MODEL.DISCRIMINATOR.MASKING
        self.normalize_loss = cfg.MODEL.DISCRIMINATOR.NORMALIZE_LOSS_WITH_MASK
        # ------------------------

    def forward(self, images, targets, pretrain=False):

        targets = [{k:v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]

        with torch.cuda.amp.autocast(enabled=(self.mixed_precision and not pretrain)):
            images = [x.to('cuda', non_blocking=True) for x in images]
            features = [None for _ in range(len(images))]
            predictions = [None for _ in range(len(images))]
            adv_predictions = [None for _ in range(len(images))]
            for i in self.valid_scale:
                feat = self.extractors[i](images[i]) # down ratio = [4, 2, 1], UNet features torch.Size([32, 64, 128, 128])
                pred = self.detector(feat)  # CenterNet with backbone resnet18, 'hm', 'wh', 'reg', single detector
                adv_pred = self.discriminator(feat)  # torch.Size([32, 3, 128, 128])

                if self.discriminator_masking:  # Changing the predictions to 0s for locations with value 0 in the mask
                    b, ch, h, w = adv_pred.shape  # torch.Size([32, 3, 128, 128])
                    # masks (b, Nc, h, w), Nc number of classes
                    mask = targets[i]['binary_masks']  # num_mask, height, width
                    adv_pred = torch.mul(adv_pred[:, None, :, :, :], mask[:, :, None, :, :]).contiguous().\
                        view(-1, ch, h, w)

                features[i] = feat
                predictions[i] = pred
                adv_predictions[i] = adv_pred
            
            #     save_heatmap(copy(images[i][0]).detach().cpu(), copy(pred[0]['hm'][0]).detach().cpu(), self.temp, i)
            #     visualize_feature(copy(feat).detach().cpu()[0], self.temp, i)
            # self.temp += 1

        # targets = [{k:v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]

        det_loss, det_loss_dict = self.loss_fn(features, predictions, adv_predictions, targets)
        # dis_loss, dis_loss_dict = self.d_loss_fn(adv_predictions)
        # -------------------------
        # list of masks, they are token as input but not necessarily used
        masks = [ret['binary_masks'] for ret in targets]
        dis_loss, dis_loss_dict = self.d_loss_fn(adv_predictions, masks)
        # -------------------------
        loss_dict = {**det_loss_dict, **dis_loss_dict}

        if self.gp:
            penalty = self._gradient_penalty(features)
            
            dis_loss += self.gp_weight * penalty
            loss_dict['gradient_penalty'] = penalty.detach()

        return det_loss, dis_loss, loss_dict

    def _load_imagenet_pretrained_model(self, model_path):
        def remove_model(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('model.'):
                    name = name[6:]  # remove 'model.' of keys
                new_state_dict[name] = v
            return new_state_dict

        self.load_state_dict(remove_model(torch.load(model_path)), strict=False)
        for i in range(1, len(self.extractors)):
            self.extractors[i].load_state_dict(self.extractors[0].state_dict())

    def _load_pretrained_model(self, model_path):
        def remove_discriminator(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if not 'discriminator' in name:
                    new_state_dict[name] = v
            return new_state_dict

        self.load_state_dict(remove_discriminator(torch.load(model_path)), strict=False)
        for i in range(1, len(self.extractors)):
            self.extractors[i].load_state_dict(self.extractors[0].state_dict())

    def _gradient_penalty(self, features):
        size = features[0].size()
        batch_size = size[0]
        device = features[0].device

        penalty = 0
        for i, j in combinations(self.valid_scale, 2):
            alpha = torch.rand(batch_size, 1, 1, 1)
            alpha = alpha.expand(size)
            alpha = alpha.to(device)

            interpolated = alpha * features[i] + (1 - alpha) * features[j]
            # interpolated.requires_grad = True
            # interpolated = interpolated.to(device)

            pred_interpolated = self.discriminator(interpolated)

            gradients = torch.autograd.grad(inputs=interpolated, outputs=pred_interpolated, grad_outputs=torch.ones(pred_interpolated.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            penalty += ((gradients_norm - 1) ** 2).mean()

        return penalty / len(list(combinations(self.valid_scale, 2)))


class Model(ModelWithLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, images):
        images = [x.to('cuda', non_blocking=True) for x in images]
        features = [None for _ in range(len(images))]
        predictions = [None for _ in range(len(images))]
        for i in self.valid_scale:
            feat = self.extractors[i](images[i])
            pred = self.detector(feat)

            features[i] = feat
            predictions[i] = pred

        return features, predictions  

### for debugging
import cv2
import numpy as np
from model.data.transforms.transforms import Compose, ToNumpy, Denormalize

def save_heatmap(image, heatmap, id, i):
    transform = Compose([
        ToNumpy(),
        Denormalize(mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278]),
    ])
    image, _, _ = transform(image)
    image = image.astype(np.uint8)
    # print(image.min(), image.max())
    heatmap = (heatmap.numpy().max(0) * 255).astype(np.uint8)
    heatmap = heatmap * 255 / heatmap.max()
    # print(heatmap.shape)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = cv2.resize(heatmap, dsize=image.shape[:2])
    # print(heatmap.shape, image.shape)

    # overlay = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)

    # ind = ind.numpy()
    # height, width, _ = image.shape
    # for i in range(ind.shape[0]):
    #     if ind[i] == 0:
    #         break
    #     x, y = divmod(ind[i], 128)
    #     overlay = cv2.circle(overlay,(x, y), 3, (255,255,255), -1)

    path = 'visualize/heatmap/{}_{}.png'.format(id, i)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, heatmap)


def visualize_feature(feat, id, i):
    transform = ToNumpy()

    feat, _, _ = transform(feat[2].unsqueeze(0))
    feat = feat - feat.min()
    feat = feat / feat.max()
    feat = feat*255
    feat = feat.astype(np.uint8)   

    path = 'visualize/feat/{}_{}.png'.format(id, i)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cv2.imwrite(path, feat)