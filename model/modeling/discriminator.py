import torch.nn as nn

from model.modeling.base_network import *


def build_discriminator(cfg):
    model_type = cfg.MODEL.DISCRIMINATOR.TYPE
    if model_type == 'residual':
        return ResidualDiscriminator(cfg)
    elif model_type == 'residual_wo_padding':
        return ResidualDiscriminatorWOPadding(cfg)
    elif model_type == 'style':
        return StyleDiscriminator(cfg)
    else:
        raise NotImplementedError('cfg.MODEL.DISCRIMINATOR.TYPE: {} is not supported'.format(model_type))


class ResidualDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(ResidualDiscriminator, self).__init__()

        feat = cfg.MODEL.DISCRIMINATOR.HIDDEN_CHANNEL  # 64
        activation = cfg.MODEL.DISCRIMINATOR.ACTIVATION  # 'lrelu'
        normalization = cfg.MODEL.DISCRIMINATOR.NORMALIZATION  # 'spectral'
        input_channel = cfg.MODEL.DISCRIMINATOR.INPUT_CHANNEL  # 64
        output_channel = len(cfg.MODEL.DOWN_RATIOS)  # 3

        num_layers = 5

        self.conv_blocks = [ConvBlock(input_channel, feat, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization)]
        for _ in range(num_layers):
            self.conv_blocks.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization))
        self.conv_blocks.append(ConvBlock(feat, output_channel, kernel_size=3, stride=1, padding=1, activation=None, normalization=None))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.input_channel = input_channel

    def forward(self, x):
        return self.conv_blocks(x[:, :self.input_channel, :, :])


class ResidualDiscriminatorWOPadding(nn.Module):
    def __init__(self, cfg):
        super(ResidualDiscriminatorWOPadding, self).__init__()

        feat = cfg.MODEL.DISCRIMINATOR.HIDDEN_CHANNEL
        activation = cfg.MODEL.DISCRIMINATOR.ACTIVATION
        normalization = cfg.MODEL.DISCRIMINATOR.NORMALIZATION
        input_channel = cfg.MODEL.DISCRIMINATOR.INPUT_CHANNEL
        output_channel = len(cfg.MODEL.DOWN_RATIOS)

        num_layers = 5

        self.conv_blocks = [ConvBlock(input_channel, feat, kernel_size=3, stride=1, padding=0, bias=True, activation=activation, normalization=normalization)]
        for _ in range(num_layers):
            self.conv_blocks.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=0, bias=True, activation=activation, normalization=normalization))
        self.conv_blocks.append(ConvBlock(feat, output_channel, kernel_size=3, stride=1, padding=0, activation=None, normalization=None))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.input_channel = input_channel

    def forward(self, x):
        return self.conv_blocks(x[:, :self.input_channel, :, :])


class StyleDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(StyleDiscriminator, self).__init__()

        feat = cfg.MODEL.DISCRIMINATOR.HIDDEN_CHANNEL
        activation = cfg.MODEL.DISCRIMINATOR.ACTIVATION
        normalization = cfg.MODEL.DISCRIMINATOR.NORMALIZATION
        input_channel = cfg.MODEL.DISCRIMINATOR.INPUT_CHANNEL
        output_channel = len(cfg.MODEL.DOWN_RATIOS)
        num_layers = 5

        self.conv_blocks = [ConvBlock(input_channel, feat, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization)]
        for _ in range(num_layers):
            self.conv_blocks.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.fc = nn.Sequential(
            DenseBlock(4096, 1000, activation=activation, normalization=normalization),
            DenseBlock(1000, output_channel, bias=True, activation=None, normalization=None)
        )
        self.input_channel = input_channel


    def _gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, x):
        x = self.conv_blocks(x[:, :self.input_channel, :, :])
        x = self._gram_matrix(x)
        x = x.view(x.shape[0], -1)
        
        return self.fc(x)