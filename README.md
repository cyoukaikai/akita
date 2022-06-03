# Modification:

1. model/data/transforms/transforms.py
MakeHeatmap: adding GT box based binary mask  

2. model/modeling/build/model.py
ModelWithLoss -> forward(): applying binary masks to the output of discriminator

3. model/config/defaults.py
adding setting for 
# whether appling the binary masks to the output of discriminator
# applying class agnostic (instance_level) masking or class specific (class_level) masking  
```
# ---------------------  
_C.MODEL.DISCRIMINATOR.MASKING = True  # using masking (True) or not (False) in discriminator
_C.MODEL.DISCRIMINATOR.MASKING_CLASS_AGNOSTIC = True  # instance_level (True) or class_level (False)
# ---------------------


```
