import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torch
from scipy.misc import imread
import math
#25693 with ships, 75000 without ships
def create_imdb(file_dir, image_dir):
    bboxes_df = pd.read_csv(file_dir)
    train_image_dir = image_dir
    imdb = []
    has_ship = 0.1
    no_ship = 0.0
    for index, row in bboxes_df.iterrows():
        image = row["ImageId"]
        label = row["EncodedPixels"]
        if isinstance(label, str):
            label = 1
        else: 
            label = 0

        if label == 0 and (no_ship / has_ship) > 0.55:
            continue

        if label == 0:
            no_ship += 1
        else:
            has_ship += 1

        im_dict = {}
        im_dict["ImageId"] = train_image_dir + "/" + image
        im_dict["label"] = label
        imdb.append(im_dict)
    print(len(imdb))
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
    self.training = training

  def __getitem__(self, index):

    im_dict = self._imdb[index]
    im = im_dict["ImageId"]
    label = im_dict["label"]
    img = imread(im)

    # RGB to BGR
    img = img[:,:,::-1]
    processed_ims, im_scales = prepare_images(img)
    data = torch.from_numpy(processed_ims)  
    data = data.permute(2, 0, 1).contiguous()
    return data, label, im

  def __len__(self):
    return len(self._imdb)

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs 

if __name__ == "__main__":
    imdb = create_imdb()
    dataset = ships_dataset(imdb)
    bs = 20
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,num_workers = 8)
    data_iter = iter(dataloader)
    train_size = int(len(imdb) / bs)
    for step in range(train_size):
        data, label = next(data_iter)


'''
for index, row in bboxs_df.iterrows():
    #print (row["ID"], row["bbox_list"])
    image = row["ID"]
    bboxs = row["bbox_list"].strip("[]").replace("(", "").replace(")", "").split(",")
    bboxs = [int(i) for i in bboxs]
    bboxs = np.array(bboxs).reshape(-1, 4)
    im = cv2.imread(train_image_dir + '/' + image)
    im_dict = {}
    im_dict["ID"] = image
    im_dict["gt_boxs"] = bboxs
    imdb.append(im_dict)
    print(im_dict)
    #img = cv2.cvtColor(img, cv2.COL OR_BGR2RGB)
  
    for bbox in bboxes:
        print('Found bbox', bbox)
        cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    plt.imshow(img)
    plt.pause(0.0001)
    if plt.waitforbuttonpress() == None:
        break
'''


