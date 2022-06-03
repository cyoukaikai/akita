# Modification:

1. model/data/transforms/transforms.py
MakeHeatmap: adding GT box based binary mask  

2. model/modeling/build/model.py
ModelWithLoss -> forward(): applying binary masks to the output of discriminator

3. model/config/defaults.py
 ```
# ---------------------  
_C.MODEL.DISCRIMINATOR.MASKING = True  # using masking (True) or not (False) in discriminator
_C.MODEL.DISCRIMINATOR.MASKING_CLASS_AGNOSTIC = True  # instance_level (True) or class_level (False)
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

