import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.utils.postprocess import _transpose_and_gather_feat


class DetectorLoss(nn.Module):
    def __init__(self, cfg):
        super(DetectorLoss, self).__init__()

        self.hm_loss_fn = FocalLoss()
        self.wh_loss_fn = RegL1Loss()
        self.reg_loss_fn = RegL1Loss()
        if cfg.SOLVER.ADV_LOSS_FN == 'wasserstain':
            self.adv_loss_fn = WassersteinLoss(d_train=False)
        elif cfg.SOLVER.ADV_LOSS_FN == 'hinge':
            self.adv_loss_fn = WassersteinLoss(d_train=False)

        self.hm_loss_weight = cfg.SOLVER.DETECTOR.HM_LOSS_WEIGHT
        self.wh_loss_weight = cfg.SOLVER.DETECTOR.WH_LOSS_WEIGHT
        self.reg_loss_weight = cfg.SOLVER.DETECTOR.REG_LOSS_WEIGHT
        self.adv_loss_weight = cfg.SOLVER.DETECTOR.ADV_LOSS_WEIGHT

        self.recon_loss = cfg.MODEL.DETECTOR.IMAGE_INPUT
        self.recon_loss_fn = nn.L1Loss()
        self.recon_loss_weight = cfg.SOLVER.DETECTOR.RECON_LOSS_WEIGHT

        self.valid_scale = cfg.MODEL.VALID_SCALE

    def forward(self, features, predictions, adv_predictions, targets):
        loss, loss_dict = 0, {}
        for i in self.valid_scale:
            feat, pred, adv_pred, target = features[i], predictions[i], adv_predictions[i], targets[i]
            for j in range(len(pred)):
                # print('heatmap', pred[j]['hm'].min(), pred[j]['hm'].max())
                # print('wh', pred[j]['wh'].min(), pred[j]['wh'].max())
                # print('reg', pred[j]['reg'].min(), pred[j]['reg'].max())
                # print('feat', feat[j].min(), feat[j].max())
                hm_loss, scaling = self.hm_loss_fn(pred[j]['hm'], target['hm'])
                wh_loss = self.wh_loss_fn(pred[j]['wh'], target['reg_mask'], target['ind'], target['wh'])
                reg_loss = self.reg_loss_fn(pred[j]['reg'], target['reg_mask'], target['ind'], target['reg'])

                loss += (self.hm_loss_weight * hm_loss) + (self.wh_loss_weight * wh_loss) + (self.reg_loss_weight * reg_loss)
            loss /= len(pred)    
        
            adv_loss = self.adv_loss_fn(adv_pred, i)
            loss += self.adv_loss_weight * adv_loss

            if self.recon_loss:
                recon_loss = self.recon_loss_fn(feat, target['image'])
                loss += self.recon_loss_weight * recon_loss
                loss_dict['recon_loss:{}'.format(i)] = recon_loss.detach()

            loss_dict['hm_loss:{}'.format(i)] = hm_loss.detach()
            loss_dict['wh_loss:{}'.format(i)] = wh_loss.detach()
            loss_dict['reg_loss:{}'.format(i)] = reg_loss.detach()
            loss_dict['det_adv_loss:{}'.format(i)] = adv_loss.detach()

        loss /= len(self.valid_scale)
        loss_dict['total_loss'] = loss.detach()

        return loss, loss_dict


class DiscriminatorLoss(nn.Module):
    def __init__(self, cfg):
        super(DiscriminatorLoss, self).__init__()

        if cfg.SOLVER.ADV_LOSS_FN == 'wasserstain':
            self.adv_loss_fn = WassersteinLoss(d_train=True)
        elif cfg.SOLVER.ADV_LOSS_FN == 'hinge':
            self.adv_loss_fn = HingeLoss(d_train=True)
        self.adv_loss_weight = cfg.SOLVER.DISCRIMINATOR.ADV_LOSS_WEIGHT

        self.valid_scale = cfg.MODEL.VALID_SCALE

    def forward(self, adv_predictions):
        loss, loss_dict = 0, {}
        for i in self.valid_scale:
            adv_pred = adv_predictions[i]

            adv_loss = self.adv_loss_fn(adv_pred, i)

            loss += self.adv_loss_weight * adv_loss

            loss_dict['dis_adv_loss:{}'.format(i)] = adv_loss.detach()

        loss /= len(self.valid_scale)
        loss_dict['total_adv_loss'] = loss.detach()

        return loss, loss_dict



class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        loss_per_region = -(pos_loss + neg_loss).detach().mean(1)

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = 0
        if num_pos == 0:
          loss = loss - neg_loss
        else:
          loss = loss - (pos_loss + neg_loss) / num_pos

        return loss, loss_per_region


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)

        return loss


class WassersteinLoss(nn.Module):
    def __init__(self, d_train):
        super(WassersteinLoss, self).__init__()
        self.d_train = d_train

    def forward(self, pred, i, scaling=None):
        if scaling is not None:
            pred *= scaling

        if not self.d_train:
            pred[:, i] *= -1.
        else:
            pred *= -1.
            pred[:, i] *= -1.       

        return torch.mean(pred)


class HingeLoss(nn.Module):
    def __init__(self, d_train):
        super(HingeLoss, self).__init__()
        self.d_train = d_train
        self.relu = nn.ReLU()

    def forward(self, pred, i, scaling=None):
        if scaling is not None:
            pred *= scaling

        if not self.d_train:
            pred[:, i] *= -1.
        else:
            pred *= -1.
            pred[:, i] *= -1.

        return torch.mean(self.relu(1. + pred))
