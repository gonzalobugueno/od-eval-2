import os
import cv2
import logging

import torch as t
import torchvision as tv
import torchvision.tv_tensors as tvt
import albumentations as A
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

albumentations_params = A.BboxParams(
    format="pascal_voc",
    label_fields=["class_labels"],
)


def extract_all(xml_file, visibilities=None, empty=False):
    """

    :param empty: allow no boundary boxes?
    :type empty: bool
    :param xml_file: path to annotations.xml file
    :type xml_file: str
    :param visibilities: filter bboxes by their visibility inclusion
    :type visibilities: list[str]|None
    :return: ready to go arrays
    :rtype: tuple[list[str], list[list[float]]
    """
    images = []
    boxes = []
    path = Path(xml_file)
    filez = os.listdir(path.parent / "images")
    for image in ET.parse(xml_file).getroot().iter('image'):
        fn = image.get("name")
        assert fn is not None
        fn = next(filter(lambda n: n.startswith(fn), filez))
        assert fn is not None

        img_boxes = []
        for box in image.iter('box'):

            if visibilities is not None:
                attribute = box.find('attribute')
                assert attribute is not None, "visibility specified but no attribute tag was found"
                assert attribute.get(
                    'name') == 'visibility', "visibility was specified but first attribute isnt visibility"

                if attribute.text not in visibilities:
                    logging.debug(f"skipping bbox of {fn} because visiblity {attribute.text} is not in {visibilities}")
                    continue

            img_boxes.append([
                float(box.get('xtl')),
                float(box.get('ytl')),
                float(box.get('xbr')),
                float(box.get('ybr'))
            ])
        if (not empty) and (len(img_boxes) == 0):
            logging.info(fn, "is empty")
            continue

        images.append(str(path.parent / "images" / fn))
        boxes.append(img_boxes)

    return images, boxes


class CustomImageDataset(t.utils.data.Dataset):
    def __init__(self, path_images, boxes, labels=None, transform=None):
        """
        :param path_images: path to images
        :type path_images: list[str]
        :param boxes: list of shape (,4), xyxy
        :type boxes: list[list[float]]
        :param labels: list of labels
        :type labels: list[int]|None
        :param transform: transform to apply to each image
        :type transform: albumentations.Compose|None
        """
        self.path_images = path_images
        self.boxes = boxes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # read image as numpy array (H,W,C) for albumentations
        image = tv.io.read_image(self.path_images[index], tv.io.ImageReadMode.RGB)
        image = image.permute(1, 2, 0).numpy()

        # C,H,W -> H,W,C

        boxes = self.boxes[index]
        labels = self.labels[index] if self.labels is not None else [1] * len(boxes)

        # apply Albumentations transform (if any)
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']  # Tensor[C,H,W] from ToTensorV2
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # convert boxes and labels to torch tensors
        boxes = t.tensor(boxes, dtype=t.float32)
        labels = t.tensor(labels, dtype=t.int64)

        # compute area
        if boxes.shape[0] == 0:
            areas = t.tensor([0.0], dtype=t.float32)
            boxes = t.zeros((0, 4), dtype=t.float32)
            logging.info(f"no bboxes found, mocked. index={index}")
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]).detach().clone().type(t.float32)
            # areas = t.tensor(areas, dtype=t.float32)
            #  UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
            #   areas = t.tensor(areas, dtype=t.float32)

        # image id
        image_id = t.tensor([index])

        if isinstance(image, np.ndarray): # for datasets without transform and ToTensor
            image = t.from_numpy(image).permute(2, 0, 1) # change H,W,C to C,H,W

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas
        }

        return image.float() / 255.0, target
