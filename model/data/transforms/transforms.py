from typing import Sized
import numpy as np
import cv2
import math
from copy import copy, deepcopy

import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)

        return image, boxes, labels

class ToTensor:
    def __call__(self, image, boxes=None, labels=None):
        return torch.from_numpy(image).permute(2, 0, 1), boxes, labels

class ToNumpy:
    def __call__(self, image, boxes=None, labels=None):
        return image.numpy().transpose((1, 2, 0)), boxes, labels

class ConvertFromInts:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertToInts:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.uint8), boxes, labels

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None):
        image /= 255
        image -= self.mean
        image /= self.std

        return image, boxes, labels

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None):
        image *= self.std
        image += self.mean
        image *= 255

        return image, boxes, labels

class Resize:
    def __init__(self, size=(512, 512), interp_method=cv2.INTER_CUBIC):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size
        self.interp_method = interp_method

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, self.size, interpolation=self.interp_method)
        
        return image, boxes, labels

class RandomResize:
    def __init__(self, resize_range=(0.6, 1.4)):
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.resize_range = resize_range

    def __call__(self, image, boxes=None, labels=None):
        interp_method = self.interp_methods[np.random.randint(0, len(self.interp_methods))]

        fx, fy = np.random.uniform(self.resize_range[0], self.resize_range[1]), np.random.uniform(self.resize_range[0], self.resize_range[1])
        height, width, _ = image.shape
        size = (int(width*fx), int(height*fy))
        image = cv2.resize(image, size, interp_method)

        if boxes is not None:
            boxes[:, 1::2] = boxes[:, 1::2] * fy
            boxes[:, 0::2] = boxes[:, 0::2] * fx

        return image, boxes, labels

class RandomMirror:
    def __call__(self, image, boxes=None, labels=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes, labels

class ShapingCrop:
    def __init__(self, down_ratio=16):
        self.down_ratio = down_ratio
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape

        bottom = height - height % self.down_ratio
        right = width - width % self.down_ratio

        image = image[:bottom, :right, :]

        return image, boxes, labels


class RandomCrop:
    def __init__(self, size=(512, 512), box_threshold=0.3, max_crops=50):
        self.box_threshold = box_threshold
        if size == int:
            self.size = (size, size)
        else:
            self.size = size
        
        self.max_crops = max_crops


    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape

        # When the image is smaller than the crop size, the image is padded by 0
        padding = ((0, max(0, self.size[0]-height)), (0, max(0, self.size[1]-width)), (0, 0))
        image = np.pad(image, padding, 'constant')

        original_image = image
        original_boxes = boxes
        original_labels = labels

        for _ in range(self.max_crops):
            image = copy(original_image)
            boxes = copy(original_boxes)
            labels = copy(original_labels)

            left = np.random.randint(max(0, width - self.size[1]) + 1)
            top = np.random.randint(max(0, height - self.size[0]) + 1)
            right = left + self.size[1]
            bottom = top + self.size[0]

            image = image[top:bottom, left:right, :]

            if boxes is not None:
                original_box_sizes = boxes[:, 2:] - boxes[:, :2]
                boxes[:, 0::2] = boxes[:, 0::2] - left
                boxes[:, 1::2] = boxes[:, 1::2] - top

                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.size[1])
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.size[0])

                cropped_box_sizes = boxes[:, 2:] - boxes[:, :2]

                # Remove invalid boxes
                mask1 = cropped_box_sizes > 0
                mask1 = mask1.all(1)

                mask2 = cropped_box_sizes/original_box_sizes >= self.box_threshold
                mask2 = mask2.all(1)

                mask = np.logical_and(mask1, mask2)

                boxes = boxes[mask]
                labels = labels[mask]

                # There are no objects in cropped image, try cropping again up to 'max_crops' times. 
                if boxes.shape[0] > 0:
                    break

        return image, boxes, labels


class ConvertColor:
    def __init__(self, current, transform):
        self.current = current
        self.transform = transform

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError

        return image, boxes, labels

class RandomContrast:
    def __init__(self, lower=0.8, upper=1.2):
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'
        self.lower = lower
        self.upper = upper
        
    def __call__(self, image, boxes=None, labels=None):
        alpha = np.random.uniform(self.lower, self.upper)
        image *= alpha

        return image, boxes, labels

class RandomBrightness:
    def __init__(self, delta=32.):
        assert delta >= 0.
        assert delta <= 255.
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        delta = np.random.uniform(-self.delta, self.delta)
        image += delta

        return image, boxes, labels

class RandomSaturation:
    def __init__(self, lower=0.8, upper=1.2):
        assert upper >= lower, 'saturation upper must be >= lower.'
        assert lower >= 0, 'saturation lower must be non-negative.'
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels

class RandomHue:
    def __init__(self, delta=18.):
        assert delta >= 0. and delta <= 360.
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
        image[:, :, 0][image[:, :, 0] > 360.] -= 360.
        image[:, :, 0][image[:, :, 0] < 0.] += 360.

        return image, boxes, labels

class RandomChannelSwap:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    
    def __call__(self, image, boxes=None, labels=None):
        swap = self.perms[np.random.randint(len(self.perms))]
        image = image[:, :, swap]

        return image, boxes, labels

class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor('RGB', 'HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor('HSV', 'RGB'),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_channel_swap = RandomChannelSwap()

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        
        image, boxes, labels = self.rand_brightness(image, boxes, labels)
        image, boxes, labels = distort(image, boxes, labels)
        image, boxes, labels = self.rand_channel_swap(image, boxes, labels)
        
        return image, boxes, labels

class Clip:
    def __init__(self, min=0., max=255.):
        self.min = min
        self.max = max
    
    def __call__(self, image, boxes=None, labels=None):
        image = np.clip(image, self.min, self.max)

        return image, boxes, labels

class MakeHeatmap:
    def __init__(self, cfg, down_ratio):
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.max_objects = cfg.SOLVER.DATA.MAX_OBJECTS  # MAX_OBJECTS = 128
        self.down_ratio = down_ratio
        self.image_target = cfg.MODEL.DETECTOR.IMAGE_INPUT

        from torchvision.transforms import Resize
        self.down_scaling = Resize(cfg.SOLVER.DATA.HEATMAP_SIZE, antialias=True)  # HEATMAP_SIZE = (128, 128)

        # ---------------------  instance_level or class_level
        self.num_mask = 1 if cfg.MODEL.DISCRIMINATOR.MASKING_CLASS_AGNOSTIC else cfg.MODEL.NUM_CLASSES
        # ---------------------

    def _gaussian_radius(self, box_size, min_overlap=0.7):
        height, width = box_size

        a = 1
        b = (height + width)
        c = (height * width * (1 - min_overlap)) / (1 + min_overlap)
        r1 = (b + np.sqrt(b**2 - 4*a*c)) / 2

        a = 4
        b = 2 * (height + width)
        c = (1 - min_overlap) * height * width
        r2 = (b + np.sqrt(b**2 - 4*a*c)) / 2

        a = 4 * min_overlap
        b = -2 * min_overlap * (height + width)
        c = (min_overlap - 1) * height * width
        r3 = (b + np.sqrt(b**2 - 4*a*c)) / 2

        return min(r1, r2, r3)

    def _draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian2D((diameter, diameter), sigma=diameter/6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape

        left, right = min(x, radius), min(width-x, radius+1)
        top, bottom = min(y, radius), min(height-y, radius+1)

        masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
        masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        
        return heatmap

    def _gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def __call__(self, image, boxes, labels):
        _, height, width = image.shape
        num_objects = min(len(boxes), self.max_objects)
        height, width = int(height / self.down_ratio), int(width / self.down_ratio)
        boxes /= self.down_ratio
        # (80, 128, 128)
        heatmap = np.zeros((self.num_classes, height, width), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)

        # ---------------------
        # Binary mask
        binary_masks = np.zeros((self.num_mask, height, width), dtype=np.uint8)
        # ----------------------------------------

        for i in range(num_objects):
            box = boxes[i]
            label = labels[i]
            class_id = label - 1 # remove __background__ label

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            center = (box[:2] + box[2:]) / 2
            center_int = center.astype(np.uint8)
            radius = max(0, int(self._gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

            heatmap[class_id] = self._draw_umich_gaussian(heatmap[class_id], center.astype(np.uint8), radius)
            wh[i] = 1. * box_width, 1. * box_height
            ind[i] = int(center_int[1] * width + center_int[0])
            reg[i] = center - center_int
            reg_mask[i] = 1

            # ---------------------
            x1, y1, x2, y2 = list(map(int, box))  # to int values, x2, y2 can be 128 because x2 = x1 + w (w can be 128)
            if self.num_mask == 1:  # instance_level mask
                binary_masks[0, y1:y2, x1:x2] = 1
            else:  # class_level mask
                binary_masks[class_id, y1:y2, x1:x2] = 1
            # ----------------------------------------

        ret = {
            'hm': torch.from_numpy(heatmap),
            'reg': torch.from_numpy(reg),
            'reg_mask': torch.from_numpy(reg_mask),
            'ind': torch.from_numpy(ind),
            'wh': torch.from_numpy(wh),

            'binary_masks': torch.from_numpy(binary_masks).bool(),  # ---------------------
            }
        
        if self.image_target:
            ret['image'] = self.down_scaling(image)

        # print(ret['wh'])

        return ret
