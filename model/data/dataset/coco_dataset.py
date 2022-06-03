import os
import numpy as np
from PIL import Image
from copy import copy
import cv2

from torch.utils.data import Dataset



class COCODataset(Dataset):
    class_names = ('__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, cfg, data_dir, ann_file ,transform=None, target_transform=None, remove_empty=False, pretrain=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        if remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}  # 1-index
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

        self.contents_match = cfg.SOLVER.CONTENTS_MATCH  # False
        from torchvision.transforms import Resize
        self.resize_x2 = Resize((256, 256), antialias=True)
        self.resize_x4 = Resize((128, 128), antialias=True)

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)

        images = [copy(image) for _ in range(len(self.transform))]
        boxes = [copy(boxes) for _ in range(len(self.transform))]
        labels = [copy(labels) for _ in range(len(self.transform))]

        if self.transform:
            for i in range(len(self.transform)):
                images[i], boxes[i], labels[i] = self.transform[i](images[i], boxes[i], labels[i])
                if self.contents_match:
                    images[1], boxes[1], labels[1] = self.resize_x2(copy(images[i])), copy(boxes[i]/2), copy(labels[i])
                    images[2], boxes[2], labels[2] = self.resize_x4(copy(images[i])), copy(boxes[i]/4), copy(labels[i])
                    break
        # TODO:
        if self.target_transform:
            targets = []
            for i in range(len(self.target_transform)):
                targets.append(self.target_transform[i](images[i], boxes[i], labels[i]))

            # save_heatmap(copy(images[0]), copy(targets[0]['hm']), copy(targets[0]['ind']), image_id, 0)
            # save_heatmap(copy(images[1]), copy(targets[1]['hm']), copy(targets[1]['ind']), image_id, 1)
            # save_heatmap(copy(images[2]), copy(targets[2]['hm']), copy(targets[2]['ind']), image_id, 2)

            return images, targets

        return images

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)[:, :, ::-1]
        return image
    
    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        ann = [obj for obj in ann if obj['iscrowd'] == 0] # filter crowd annotations
        boxes = np.array([self._xywh2xyxy(obj['bbox']) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))

        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        return boxes, labels

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)


    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def __len__(self):
        return len(self.ids)

from model.data.transforms.transforms import Compose, ToNumpy, Denormalize
def save_heatmap(image, heatmap, ind, id, i):
    transform = Compose([
        ToNumpy(),
        Denormalize(mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278]),
    ])
    image, _, _ = transform(image)
    image = image.astype(np.uint8)
    # print(image.min(), image.max())
    heatmap = (heatmap.numpy().max(0) * 255).astype(np.uint8)
    # print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, dsize=image.shape[:2])
    # print(heatmap.shape, image.shape)

    overlay = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)

    # ind = ind.numpy()
    # height, width, _ = image.shape
    # for i in range(ind.shape[0]):
    #     if ind[i] == 0:
    #         break
    #     x, y = divmod(ind[i], 128)
    #     overlay = cv2.circle(overlay,(x, y), 3, (255,255,255), -1)

    cv2.imwrite('temp/{}_{}.png'.format(id, i), overlay)

