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

# ----------------
# import torch.nn.functional as F
# from matplotlib.colors import LinearSegmentedColormap
#
# from captum.attr import IntegratedGradients
# from captum.attr import GradientShap
# from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerGradCam, LayerAttribution  # FeatureAblation, LayerActivation,
import datetime


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

        self.attr_output_dir = os.path.join(cfg.OUTPUT_DIR, 'attribution')
        if not os.path.exists(self.attr_output_dir):
            os.makedirs(self.attr_output_dir)

        # _C.INPUT.MEAN = [0.408, 0.447, 0.470]
        # _C.INPUT.STD = [0.289, 0.274, 0.278]
        self.input_mean = cfg.INPUT.MEAN
        self.input_std = cfg.INPUT.STD
        print(f'Attribution analysis will be saved in {self.attr_output_dir}')
        # ------------------------

    def forward(self, images, targets, pretrain=False, attribute=False, iter=None):

        targets = [{k: v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]

        with torch.cuda.amp.autocast(enabled=(self.mixed_precision and not pretrain)):
            images = [x.to('cuda', non_blocking=True) for x in images]
            features = [None for _ in range(len(images))]
            predictions = [None for _ in range(len(images))]
            adv_predictions = [None for _ in range(len(images))]

            for i in self.valid_scale:
                feat = self.extractors[i](
                    images[i])  # down ratio = [4, 2, 1], UNet features torch.Size([32, 64, 128, 128])
                pred = self.detector(feat)  # CenterNet with backbone resnet18, 'hm', 'wh', 'reg', single detector
                adv_pred = self.discriminator(feat)  # torch.Size([32, 3, 128, 128])

                # ----------------------
                if attribute:
                    # Using Captum to conduct attribution analysis for features of each image and each scale.
                    for k in range(feat.shape[0]):
                        for j in range(len(self.valid_scale)):
                            self.attribute(feat=feat, scale_id=i, target_scale_id=j,
                                           images=images, img_id=k,
                                           targets=targets, iter=iter)
                    continue
                # ----------------------

                if self.discriminator_masking:  # Changing the predictions to 0s for locations with value 0 in the mask
                    b, ch, h, w = adv_pred.shape  # torch.Size([32, 3, 128, 128])
                    # masks (b, Nc, h, w), Nc number of classes
                    mask = targets[i]['binary_masks']  # num_mask, height, width
                    adv_pred = torch.mul(adv_pred[:, None, :, :, :], mask[:, :, None, :, :]).contiguous(). \
                        view(-1, ch, h, w)

                features[i] = feat
                predictions[i] = pred
                adv_predictions[i] = adv_pred

            #     save_heatmap(copy(images[i][0]).detach().cpu(), copy(pred[0]['hm'][0]).detach().cpu(), self.temp, i)
            #     visualize_feature(copy(feat).detach().cpu()[0], self.temp, i)
            # self.temp += 1

        # targets = [{k:v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]
        # --------------------
        if attribute:  # skip calculating the loss if we conduct attribution analysis.
            return None, None, None
        # --------------------

        det_loss, det_loss_dict = self.loss_fn(features, predictions, adv_predictions, targets)
        # dis_loss, dis_loss_dict = self.d_loss_fn(adv_predictions)
        # -------------------------
        # list of masks, they are taken as input but not necessarily used
        masks = [ret['binary_masks'] for ret in targets]
        dis_loss, dis_loss_dict = self.d_loss_fn(adv_predictions, masks)
        # -------------------------
        loss_dict = {**det_loss_dict, **dis_loss_dict}

        if self.gp:
            penalty = self._gradient_penalty(features)

            dis_loss += self.gp_weight * penalty
            loss_dict['gradient_penalty'] = penalty.detach()

        return det_loss, dis_loss, loss_dict

    def agg_segmentation_wrapper(self, input_feat):
        """
        Adapted from https://captum.ai/tutorials/Segmentation_Interpret.
        The sizes recorded in the comments are for the example of the link, not for this model.
        """
        model_out = self.discriminator(input_feat)
        out_max = torch.argmax(model_out, dim=1, keepdim=True)  # torch.Size([1, 1, 640, 949])
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)  # torch.Size([1, 21, 640, 949])
        return (model_out * selected_inds).sum(dim=(2, 3))  # torch.Size([1, 21])

    def attribute(self, feat, images, img_id, scale_id, target_scale_id, iter=None, targets=None):
        """ We analyze one image each time. Here we analyze the kth image, where k = img_id.
        scale_id: the sr branch id,
        target_scale_id: the output class id for attribution analysis.
        """

        normalized_inp = feat[img_id].unsqueeze(0)
        lgc = LayerGradCam(self.agg_segmentation_wrapper, self.discriminator.conv_blocks[-1])
        gc_attr = lgc.attribute(normalized_inp, target=target_scale_id)  # torch.Size([1, 1, 80, 119])

        # viz.visualize_image_attr(gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(), sign="all")
        image = images[scale_id][img_id]
        mean = torch.tensor(self.input_mean, device=image.device, dtype=image.dtype)
        std = torch.tensor(self.input_std, device=image.device, dtype=image.dtype)
        # preproc_img_input = images[scale_id][img_id]  # torch.Size([3, 512, 512])
        preproc_img = image * std[:, None, None] + mean[:, None, None]

        upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, preproc_img.shape[1:])
        # print("Upsampled Shape:", upsampled_gc_attr.shape)

        # If there is no positive values or negative values for visualization, then visualize_image_attr_multiple will
        # # # cause error.
        # we detect this and modify one value to avoid this to happen to visualize the result anyway.
        if torch.sum(upsampled_gc_attr > 0) == 0:  # torch.Size([1, 1, 512, 512])
            upsampled_gc_attr[0, 0, 0, :] = 1e-3
        if torch.sum(upsampled_gc_attr < 0) == 0:
            upsampled_gc_attr[0, 0, -1, :] = -1e-3
        # if torch.sum(upsampled_gc_attr > 0) == 0 or torch.sum(upsampled_gc_attr < 0) == 0:
        #     return

        # returns a figure object without showing
        plt_fig, plt_axis = viz.visualize_image_attr_multiple(
            upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
            original_image=preproc_img.cpu().permute(1, 2, 0).detach().numpy(),
            signs=["all", "positive", "negative", "positive", "negative"],  # absolute_value

            titles=["all contribution", "positive contribution", "negative contribution",
                    "positive contribution", "negative contribution"
                    ],
            fig_size=(18, 6),
            methods=["original_image", "blended_heat_map", "blended_heat_map", "heat_map", "heat_map"],
            show_colorbar=True,
            use_pyplot=False  # Set it to True for showing the plot
        )
        if iter is not None:
            out_img_file = os.path.join(
                self.attr_output_dir,
                f'iter{iter}_scale{scale_id}_img{img_id}_target_scale_id{target_scale_id}.png'
            )
        else:  # save the plot with the time stamp
            dt_now = datetime.datetime.now()
            time_str = str(dt_now.date()) + '_' + str(dt_now.time())
            out_img_file = os.path.join(
                self.attr_output_dir,
                f'{time_str}_scale{scale_id}_img{img_id}_target_scale_id{target_scale_id}.png')
        plt_fig.savefig(out_img_file)

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

            gradients = torch.autograd.grad(inputs=interpolated, outputs=pred_interpolated,
                                            grad_outputs=torch.ones(pred_interpolated.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
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
    feat = feat * 255
    feat = feat.astype(np.uint8)

    path = 'visualize/feat/{}_{}.png'.format(id, i)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cv2.imwrite(path, feat)
