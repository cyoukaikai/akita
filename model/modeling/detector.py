import torch.nn as nn

from model.modeling.base_network import *
from model.utils.misc import _sigmoid


def build_detector(cfg):
    model_type = cfg.MODEL.DETECTOR.TYPE
    if model_type == 'resnet18':
        return Resnet(cfg, 18)
    
    else:
        raise NotImplementedError('cfg.MODEL.DETECTOR.TYPE: {} is not supported'.format(model_type))


resnet_spec = {18: [2, 2, 2, 2]}

class Resnet(nn.Module):
    def __init__(self, cfg, num_layers):
        super(Resnet, self).__init__()

        self.heads = {
            'hm': cfg.MODEL.NUM_CLASSES,
            'wh': 2,
            'reg': 2,
        }
        self.activation = cfg.MODEL.DETECTOR.ACTIVATION
        self.normalization = cfg.MODEL.DETECTOR.NORMALIZATION

        feat = cfg.MODEL.DETECTOR.INPUT_CHANNEL
        block = resnet_spec[num_layers]

        if cfg.MODEL.DETECTOR.IMAGE_INPUT:
            self.conv_layers = [ConvBlock(3, feat, kernel_size=7, stride=1, padding=3, activation=self.activation, normalization=self.normalization)]
        else:
            self.conv_layers = []
        layer, feat = self._make_conv_layers(block[0], feat, stride=1)
        self.conv_layers.append(layer)
        for i in range(1, 4):
            layer, feat = self._make_conv_layers(block[i], feat, stride=2)
            self.conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.deconv_layers = []
        for i in range(3):
            layer, feat = self._make_deconv_layers(feat)
            self.deconv_layers.append(layer)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)

        for head in self.heads.keys():
            channel = self.heads[head]
            conv = nn.Sequential(
                ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=self.activation, normalization=None, bias=True),
                ConvBlock(feat, channel, kernel_size=1, stride=1, padding=0, activation=None, normalization=None, bias=True),
            )
            if 'hm' in head:
                conv[-1].layer.bias.data.fill_(-2.19)
            else:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.001)

            self.__setattr__(head, conv)


    def _make_conv_layers(self, num_block, feat, stride=1):
        layers = []
        if stride != 1:
            layers.append(ResidualBlock(feat, feat*2, kernel_size=3, stride=stride, padding=1, activation=self.activation, normalization=self.normalization))
            feat = feat*2
        else:
            layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=self.activation, normalization=self.normalization))
        
        for i in range(num_block - 1):
            layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=self.activation, normalization=self.normalization))

        return nn.Sequential(*layers), feat


    def _make_deconv_layers(self, feat):
        feat = int(feat/2)
        
        layers = []
        layers.append(ModulatedDeformableBlock(feat*2, feat, kernel_size=3, stride=1, padding=1, activation=self.activation, normalization=self.normalization))
        layers.append(DeconvBlock(feat, feat, kernel_size=4, stride=2, padding=1, activation=self.activation, normalization=self.normalization))

        return nn.Sequential(*layers), feat

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        for layer in self.deconv_layers:
            x = layer(x)

        ret = {}
        for head in self.heads.keys():
            ret[head] = self.__getattr__(head)(x)
        ret['hm'] = _sigmoid(ret['hm'])

        return [ret]