import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.utils.data as data
import torch
from scipy.misc import imread

def create_imdb(file_name, data_dir):
    imdb = []
    bboxes_df = pd.read_csv(file_name)
    image_dir = data_dir
    for index, row in bboxes_df.iterrows():
        image = row["ID"]
        bboxes = row["bbox_list"].strip("[]").replace("(", "").replace(")", "").split(",")
        bboxes = [int(i) for i in bboxes]
        bboxes = np.array(bboxes).reshape(-1, 4)
        #swap x and y
        tmp = bboxes[:, 0].copy()
        bboxes[:, 0] = bboxes[:, 1]
        bboxes[:, 1] = tmp
        tmp = bboxes[:, 2].copy()
        bboxes[:, 2] = bboxes[:, 3]
        bboxes[:, 3] = tmp

        im_dict = {}
        im_dict["ID"] = image_dir + "/" + image
        im_dict["gt_boxes"] = bboxes
        imdb.append(im_dict)
    return imdb


def prepare_images(img):
    #Mean subtract and scale an image
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    scale = 600.0

    im = img.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = scale / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                      interpolation=cv2.INTER_LINEAR)
    return im, im_scale

class ships_dataset(data.Dataset):

  def __init__(self, imdb, training=True):
    self._imdb = imdb
    self.max_num_box = 20
    self.training = training

  def __getitem__(self, index):

    im_dict = self._imdb[index]
    im = im_dict["ID"]
    num_boxes = len(im_dict["gt_boxes"])
    img = imread(im)

    # RGB to BGR
    img = img[:,:,::-1]
    processed_ims, im_scales = prepare_images(img)

    gt_boxes = np.empty((num_boxes, 5), dtype=np.float32)
    gt_boxes[:, 0:4] = im_dict["gt_boxes"] * im_scales
    gt_boxes[:, 4] = 1.0 # 1.0 for ship class

    im_info = np.array([processed_ims.shape[0], processed_ims.shape[1], im_scales], dtype=np.float32)

    data = torch.from_numpy(processed_ims)
    im_info = torch.from_numpy(im_info)

    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(0), data.size(1)
    if self.training:
        np.random.shuffle(gt_boxes)
        gt_boxes = torch.from_numpy(gt_boxes)

        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        data = data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        return data, im_info, gt_boxes_padding, num_boxes, im
    else:
        data = data.permute(2, 0, 1).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)
        id = im
        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes, im

  def __len__(self):
    return len(self._imdb)
