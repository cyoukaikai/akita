# Modification:

1. model/data/transforms/transforms.py
MakeHeatmap: adding GT box based binary mask  

2. model/modeling/build/model.py
ModelWithLoss -> forward(): applying binary masks to the output of discriminator

3. model/config/defaults.py
 ```
# ---------------------  
_C.MODEL.DISCRIMINATOR.MASKING = True  # using masking (True) or not (False) in discriminator
_C.MODEL.DISCRIMINATOR.MASKING_CLASS_AGNOSTIC = False  # instance_level (True) or class_level (False)
# Normalize the loss only if (MODEL.DISCRIMINATOR.MASKING) = True AND
#   (MODEL.DISCRIMINATOR.NORMALIZE_LOSS_WITH_MASK = True)
_C.MODEL.DISCRIMINATOR.NORMALIZE_LOSS_WITH_MASK = True
# ---------------------
```
adding setting for 
- whether appling the binary masks to the output of discriminator
- applying class agnostic (instance_level) masking or class specific (class_level) masking 

# Other things

discriminator output O: b, ch, h, w = 32, 3, 128, 128 (3 means scales 0, 1, 4)
mask M: 
class agnostic: b, Nc, h, w = 32, 1, 128, 128
class specific: b, Nc, h, w = 32, 80, 128, 128 (COCO has 80 classes)

to apply M on O, we can get new output: (b, chxNc, h, w), or (bxNc, ch, h, w), but the second dimension is used as the class label for scales 0, 1, 4 in `WassersteinLoss`,
```
    def forward(self, pred, i, scaling=None):
        if scaling is not None:
            pred *= scaling

        if not self.d_train:
            pred[:, i] *= -1.
        else:
            pred *= -1.
            pred[:, i] *= -1.       

        return torch.mean(pred)
```
So I use the output format (bxNc, ch, h, w) so that the second dimension is always 3.

# Preliminary experiments

For the same discriminator output, 
normalized discriminator loss : tensor(1.4225, device='cuda:0')
not normalized discriminator loss: tensor(0.0066, device='cuda:0')


Training with normalized discriminator loss
```
===> Iter: 0000050, LR: 0.000500, Cost: 32.72s, Eta: 6 days, 19:36:21, Detector Loss: 864.042052, Discriminator Loss: -0.000103
===> Iter: 0000100, LR: 0.000500, Cost: 24.95s, Eta: 6 days, 0:11:56, Detector Loss: 74.589113, Discriminator Loss: -0.000048
===> Iter: 0000150, LR: 0.000500, Cost: 24.25s, Eta: 5 days, 16:32:51, Detector Loss: 14.623296, Discriminator Loss: 0.000002
===> Iter: 0000200, LR: 0.000500, Cost: 24.39s, Eta: 5 days, 12:53:26, Detector Loss: 10.610689, Discriminator Loss: 0.000015
===> Iter: 0000250, LR: 0.000500, Cost: 24.59s, Eta: 5 days, 10:53:47, Detector Loss: 9.362442, Discriminator Loss: -0.000020
===> Iter: 0000300, LR: 0.000500, Cost: 24.91s, Eta: 5 days, 9:49:30, Detector Loss: 8.673958, Discriminator Loss: 0.000032
===> Iter: 0000350, LR: 0.000500, Cost: 25.07s, Eta: 5 days, 9:10:23, Detector Loss: 8.252213, Discriminator Loss: 0.000022
===> Iter: 0000400, LR: 0.000500, Cost: 25.18s, Eta: 5 days, 8:45:14, Detector Loss: 7.936076, Discriminator Loss: -0.000055
===> Iter: 0000450, LR: 0.000500, Cost: 25.25s, Eta: 5 days, 8:27:41, Detector Loss: 7.892187, Discriminator Loss: -0.000052
===> Iter: 0000500, LR: 0.000500, Cost: 25.25s, Eta: 5 days, 8:13:36, Detector Loss: 7.723672, Discriminator Loss: -0.000018
===> Iter: 0000550, LR: 0.000500, Cost: 25.22s, Eta: 5 days, 8:01:11, Detector Loss: 7.722490, Discriminator Loss: -0.000064
===> Iter: 0000600, LR: 0.000500, Cost: 25.33s, Eta: 5 days, 7:53:38, Detector Loss: 7.434922, Discriminator Loss: -0.000063
===> Iter: 0000650, LR: 0.000500, Cost: 25.35s, Eta: 5 days, 7:47:29, Detector Loss: 7.607540, Discriminator Loss: -0.000120
===> Iter: 0000700, LR: 0.000500, Cost: 25.51s, Eta: 5 days, 7:45:37, Detector Loss: 7.443577, Discriminator Loss: -0.000012
===> Iter: 0000750, LR: 0.000500, Cost: 25.47s, Eta: 5 days, 7:43:07, Detector Loss: 7.274951, Discriminator Loss: -0.000011
===> Iter: 0000800, LR: 0.000500, Cost: 25.43s, Eta: 5 days, 7:40:14, Detector Loss: 7.284674, Discriminator Loss: -0.000042
===> Iter: 0000850, LR: 0.000500, Cost: 25.47s, Eta: 5 days, 7:38:18, Detector Loss: 7.315438, Discriminator Loss: -0.000085
===> Iter: 0000900, LR: 0.000500, Cost: 25.51s, Eta: 5 days, 7:37:15, Detector Loss: 7.170705, Discriminator Loss: -0.000071
===> Iter: 0000950, LR: 0.000500, Cost: 25.49s, Eta: 5 days, 7:35:59, Detector Loss: 7.142722, Discriminator Loss: -0.000105
===> Iter: 0001000, LR: 0.000500, Cost: 25.50s, Eta: 5 days, 7:34:52, Detector Loss: 7.081477, Discriminator Loss: -0.000086
===> Iter: 0001050, LR: 0.000500, Cost: 25.54s, Eta: 5 days, 7:34:26, Detector Loss: 7.017793, Discriminator Loss: -0.000093
===> Iter: 0001100, LR: 0.000500, Cost: 25.53s, Eta: 5 days, 7:33:51, Detector Loss: 6.962031, Discriminator Loss: -0.000085
===> Iter: 0001150, LR: 0.000500, Cost: 25.60s, Eta: 5 days, 7:34:14, Detector Loss: 6.811521, Discriminator Loss: -0.000130
===> Iter: 0001200, LR: 0.000500, Cost: 25.51s, Eta: 5 days, 7:33:19, Detector Loss: 6.775800, Discriminator Loss: -0.000091
===> Iter: 0001250, LR: 0.000500, Cost: 25.48s, Eta: 5 days, 7:32:10, Detector Loss: 7.084497, Discriminator Loss: -0.000075
===> Iter: 0001300, LR: 0.000500, Cost: 25.53s, Eta: 5 days, 7:31:35, Detector Loss: 6.694517, Discriminator Loss: -0.000014
===> Iter: 0001350, LR: 0.000500, Cost: 25.44s, Eta: 5 days, 7:30:03, Detector Loss: 6.653039, Discriminator Loss: -0.000014
===> Iter: 0001400, LR: 0.000500, Cost: 25.34s, Eta: 5 days, 7:27:28, Detector Loss: 6.676461, Discriminator Loss: -0.000020
===> Iter: 0001450, LR: 0.000500, Cost: 25.38s, Eta: 5 days, 7:25:33, Detector Loss: 6.648923, Discriminator Loss: -0.000036
===> Iter: 0001500, LR: 0.000500, Cost: 25.33s, Eta: 5 days, 7:23:09, Detector Loss: 6.639132, Discriminator Loss: -0.000058
===> Iter: 0001550, LR: 0.000500, Cost: 25.34s, Eta: 5 days, 7:21:03, Detector Loss: 6.555725, Discriminator Loss: -0.000101
===> Iter: 0001600, LR: 0.000500, Cost: 25.24s, Eta: 5 days, 7:18:05, Detector Loss: 6.572634, Discriminator Loss: -0.000071
===> Iter: 0001650, LR: 0.000500, Cost: 25.23s, Eta: 5 days, 7:15:09, Detector Loss: 6.633294, Discriminator Loss: -0.000037
===> Iter: 0001700, LR: 0.000500, Cost: 26.72s, Eta: 5 days, 7:25:38, Detector Loss: 6.436580, Discriminator Loss: -0.000001
===> Iter: 0001750, LR: 0.000500, Cost: 25.34s, Eta: 5 days, 7:23:35, Detector Loss: 6.483980, Discriminator Loss: -0.000051
===> Iter: 0001800, LR: 0.000500, Cost: 25.27s, Eta: 5 days, 7:21:02, Detector Loss: 6.478489, Discriminator Loss: -0.000217
===> Iter: 0001850, LR: 0.000500, Cost: 25.32s, Eta: 5 days, 7:19:01, Detector Loss: 6.462882, Discriminator Loss: -0.000128
===> Iter: 0001900, LR: 0.000500, Cost: 25.27s, Eta: 5 days, 7:16:39, Detector Loss: 6.525437, Discriminator Loss: -0.000091
===> Iter: 0001950, LR: 0.000500, Cost: 25.32s, Eta: 5 days, 7:14:45, Detector Loss: 6.470995, Discriminator Loss: -0.000123
===> Iter: 0002000, LR: 0.000500, Cost: 25.30s, Eta: 5 days, 7:12:50, Detector Loss: 6.383064, Discriminator Loss: -0.000161
===> Iter: 0002050, LR: 0.000500, Cost: 25.35s, Eta: 5 days, 7:11:19, Detector Loss: 6.168411, Discriminator Loss: -0.000260
===> Iter: 0002100, LR: 0.000500, Cost: 25.45s, Eta: 5 days, 7:10:36, Detector Loss: 6.431139, Discriminator Loss: -0.000237
===> Iter: 0002150, LR: 0.000500, Cost: 25.46s, Eta: 5 days, 7:09:57, Detector Loss: 6.412099, Discriminator Loss: -0.000251
===> Iter: 0002200, LR: 0.000500, Cost: 25.56s, Eta: 5 days, 7:09:58, Detector Loss: 6.384065, Discriminator Loss: -0.000128
===> Iter: 0002250, LR: 0.000500, Cost: 25.59s, Eta: 5 days, 7:10:11, Detector Loss: 6.138038, Discriminator Loss: -0.000079
===> Iter: 0002300, LR: 0.000500, Cost: 25.47s, Eta: 5 days, 7:09:34, Detector Loss: 6.376725, Discriminator Loss: -0.000016
===> Iter: 0002350, LR: 0.000500, Cost: 25.59s, Eta: 5 days, 7:09:46, Detector Loss: 6.152978, Discriminator Loss: 0.000036
===> Iter: 0002400, LR: 0.000500, Cost: 25.71s, Eta: 5 days, 7:10:40, Detector Loss: 6.206933, Discriminator Loss: -0.000084
===> Iter: 0002450, LR: 0.000500, Cost: 25.60s, Eta: 5 days, 7:10:51, Detector Loss: 6.195686, Discriminator Loss: -0.000099
===> Iter: 0002500, LR: 0.000500, Cost: 25.62s, Eta: 5 days, 7:11:09, Detector Loss: 6.102291, Discriminator Loss: -0.000101
===> Iter: 0002550, LR: 0.000500, Cost: 25.60s, Eta: 5 days, 7:11:17, Detector Loss: 6.049708, Discriminator Loss: -0.000058
===> Iter: 0002600, LR: 0.000500, Cost: 25.62s, Eta: 5 days, 7:11:31, Detector Loss: 6.039007, Discriminator Loss: -0.000108
===> Iter: 0002650, LR: 0.000500, Cost: 25.57s, Eta: 5 days, 7:11:27, Detector Loss: 5.958836, Discriminator Loss: -0.000189
===> Iter: 0002700, LR: 0.000500, Cost: 25.63s, Eta: 5 days, 7:11:40, Detector Loss: 6.181076, Discriminator Loss: -0.000126
===> Iter: 0002750, LR: 0.000500, Cost: 25.77s, Eta: 5 days, 7:12:39, Detector Loss: 6.080488, Discriminator Loss: -0.000242
===> Iter: 0002800, LR: 0.000500, Cost: 25.60s, Eta: 5 days, 7:12:38, Detector Loss: 6.090125, Discriminator Loss: -0.000154
===> Iter: 0002850, LR: 0.000500, Cost: 25.70s, Eta: 5 days, 7:13:09, Detector Loss: 6.091262, Discriminator Loss: -0.000084
===> Iter: 0002900, LR: 0.000500, Cost: 25.61s, Eta: 5 days, 7:13:10, Detector Loss: 6.160134, Discriminator Loss: -0.000183

```

Training with normalized discriminator loss:
```
===> Iter: 0000050, LR: 0.000500, Cost: 75.64s, Eta: 15 days, 18:10:32, Detector Loss: 863.905340, Discriminator Loss: -0.000852
===> Iter: 0000100, LR: 0.000500, Cost: 29.03s, Eta: 10 days, 21:38:24, Detector Loss: 74.373735, Discriminator Loss: -0.000719
===> Iter: 0000150, LR: 0.000500, Cost: 29.37s, Eta: 9 days, 7:21:27, Detector Loss: 14.621235, Discriminator Loss: -0.000447
===> Iter: 0000200, LR: 0.000500, Cost: 29.22s, Eta: 8 days, 12:02:00, Detector Loss: 10.617451, Discriminator Loss: -0.000531
===> Iter: 0000250, LR: 0.000500, Cost: 29.33s, Eta: 8 days, 0:32:27, Detector Loss: 9.369847, Discriminator Loss: -0.000396
===> Iter: 0000300, LR: 0.000500, Cost: 29.34s, Eta: 7 days, 16:53:03, Detector Loss: 8.674080, Discriminator Loss: -0.000283

```
