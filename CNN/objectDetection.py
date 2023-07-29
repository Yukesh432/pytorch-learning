# TorchVision object detection Finetuning
# References: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms):
        self.root= root
        self.transforms= transforms
        # loading all image files adn sorting them, ensuring that they are aligned
        self.imgs= list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks= list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # get the idx-th image and mask paths from the lists
        img_path= os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path= os.path.join(self.root, "PedMasks", self.masks[idx])
        # open the image in RGB model using PIL
        img= Image.open(img_path).convert("RGB")
        """The masks are not converted to RGB because each color corresponding to a different instance (0 being background)"""
        mask= Image.open(mask_path)
        # convert the PIL image into numpy array
        mask= np.array(mask)
        # instances are enoded as differenct colors
        # get unique object ids from the mask, as each color represents a different instance
        obj_ids= np.unique(mask)
        # first id is the background, so removing it..
        obj_ids= obj_ids[1:]

        # splittin the color-encoded mask into a set of binary masks
        masks= mask == obj_ids[:, None, None]

        # getting the bounding box coordinates for each mask
        num_objs= len(obj_ids)
        boxes= []
        for i in range(num_objs):
            pos= np.nonzero(masks[i])
            xmin= np.min(pos[1])
            xmax= np.max(pos[1])
            ymin= np.min(pos[0])
            ymax= np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert bounding box data and masks into tensor type
        boxes= torch.as_tensor(boxes, dtype=torch.float32)
        labels= torch.ones((num_objs,), dtype=torch.int64)
        masks= torch.as_tensor(masks, dtype= torch.uint8)

        # target information
        image_id= torch.tensor([idx])
        area= (boxes[:,3]- boxes[:, 1]) * (boxes[:,2]- boxes[:,0])
        # suppose all isntances are not crowd
        iscrowd= torch.zeros((num_objs,), dtype= torch.int64)

        target= {}
        target["boxes"]= boxes
        target["labels"]= labels
        target["masks"]= masks
        target["image_id"]= image_id
        target["area"]= area
        target["iscrowd"]= iscrowd

        # apply the specified transformations to the image and target data
        if self.transforms is not None:
            img, target= self.transforms(img, target)

        return img, target
    
    def __len__(self):
        # returns the total number of images in the dataset
        return len(self.imgs)
    


# Finetuning from pretrained model.............

# loading model pre-trained on COCO
model= torchvision.models.detection.fasterrcnn_resnet50_fpn(weights= "DEFAULT")
# replacing the classifier with a new one, that has num_classes which is user-defined
num_classes= 2   # 1 class(person) + background
# get no. of input features for the classifier
in_features= model.roi_heads.box_predictor.cls_score.in_features
# replace the pretraiined head with a new one
model.roi_heads.box_predictor= FastRCNNPredictor(in_features, num_classes)



