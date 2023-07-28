# TorchVision object detection Finetuning
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms):
        self.root= root
        self.trasnforms= transforms
        # loading all image files adn sorting them, ensuring that they are aligned
        self.imgs= list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks= list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path= os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path= os.path.join(self.root, "PedMasks", self.masks[idx])
        img= Image.open(img_path).convert("RGB")
        """Here we havent converted the mask to RGB, because each color corresponds
        to different instance with 0 being background"""

        mask= Image.open(mask_path)
        # convert the PIL image into numpy array
        mask= np.array(mask)
        # instances are enoded as differenct colors
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

        # convert all data into tensor type
        boxes= torch.as_tensor(boxes, dtype=torch.float32)
        labels= torch.ones((num_objs,), dtype=torch.int64)
        masks= torch.as_tensor(masks, dtype= torch.uint8)


        image_id= torch.tensor([idx])
        area= (boxes[:,3]- boxes[:, 1]) * (boxes[:,2]- boxes[:,0])
        # suppose all isntances are not crowd
        iscrowd= torch.zeros((num_objs,), dtype= torch.int64)

        target= {}
        