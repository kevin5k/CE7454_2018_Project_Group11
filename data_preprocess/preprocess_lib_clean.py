import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import fnmatch
import matplotlib.image as mpimg
import pandas as pd
from skimage.data import imread
from scipy.misc import imread
import torchvision
import torchvision.transforms as transforms
import os.path as osp
import glob
from torch.utils import data as utils_data

def LoadMyData(filename, load_dict=False, load_list=False, matlab=False, pandas=True, debug=False, pandas_create_imdb=False):
    if load_dict:
        if debug:
            print('Method 1: Dictionary Load Method')

        with open(filename, 'r') as csvDataFile:
            # csvReader = csv.reader(csvDataFile, delimiter=',')
            csvReader = csv.DictReader(csvDataFile)

            # Print out data
            if debug:
                for row in csvReader:
                    if row["EncodedPixels"] is not '':
                        print(row["ImageId"])

        return csvReader

    if load_list:
        if debug:
            print('Method 2: List Load Method')

        with open(filename, newline='') as csvDataFile:
            my_data = list(csv.reader(csvDataFile))

        if debug:
            print(my_data)
            print(type(my_data))

        return my_data

    # https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
    if matlab:
        img = mpimg.imread(filename)
        return img

    if pandas:
        data = pd.read_csv(filename)
        return data
    
    if pandas_create_imdb:
        imdb = []
        bboxes_df = pd.read_csv(filename)
#         ship_dir = './kaggle/'
#         train_image_dir = os.path.join(ship_dir, 'train')
#         train_image_dir = train_dir
        train_image_dir = './train/'
        im_dict = {}
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
            im_dict["ID"] = train_image_dir + image
            im_dict["gt_boxes"] = bboxes
            imdb.append(im_dict)
        return imdb

def ReadImage(filename, file_read=False, matlab_read=False, pil_read=False, skimage_read=False, resizeX=200, resizeY=200, show=False):

    # Read image from file
    if file_read:
        img_data = Image.open(filename)
        # img_array = np.array(img)
        # print(img.size, img.format)
        arr = np.array(img_data)
        if show:
            img_data.show()

        return arr

    # Matlab Read image from file
    if matlab_read:
        img_arr = plt.imread(filename)
        if show:
            plt.imshow(img_arr)
            plt.show()
        # utils.show(img_array)

        return img_arr

    if pil_read:
        # https: // www.oreilly.com / library / view / programming - computer - vision / 9781449341916 / ch01.html
        # pil_im = Image.open(filename).convert  # Convert from color
        # pil_im = Image.open(filename).convert('L')  # Convert to grayscale
        pil_im = Image.open(filename).resize((resizeX, resizeY))
        if show:
            print(type(pil_im))
            print(str(Image.open(filename).size))
        return pil_im

    if skimage_read:
        img = imread(filename)
        return img


def SearchImages(dirname, imgformat=None, debug=False):

    count = 0
    if imgformat is not None:
        # Change the working directory
        os.chdir(dirname)
        listofFiles = os.listdir('.')
        # pattern = "*.jpg"
        pattern = "*." + str(imgformat)
        for entry in listofFiles:
            if fnmatch.fnmatch(entry, pattern):
                count = count + 1
                if debug:
                    print(entry)

        return count



def SaveImage(npdata, outfilename, format=None):

    # https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
    if format is 'grayscale':
        img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
        img.save(outfilename)

    # if format is 'rgb':
    #     outimg = Image.fromarray(ycc_uint8, "RGB")
    #     outimg.save("ycc.tif")


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def apply_mask(image, mask):
    for x, y in mask:
        image[x, y, [0, 1]] = 255
    return image

def rle_load_file(file="train"):
    """
    Loads a csv, creates the fields `HasShip` and `TotalShips` dropping `EncodedPixels` and setting `ImageId` as index.
    """
    df = pd.read_csv(f"../input/{file}_ship_segmentations.csv")
    df['HasShip'] = df['EncodedPixels'].notnull()
    df = df.groupby("ImageId").agg({'HasShip': ['first', 'sum']}) # counts amount of ships per image, sets ImageId to index
    df.columns = ['HasShip', 'TotalShips']
    return df


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes
def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

''' Load the labels, extract the imagefilename and column for HasShip data Only. Then save it as an external file - Training 
    and Test files '''
def load_file(file="train"):
    """
    Loads a csv, creates the fields `HasShip` and `TotalShips` dropping `EncodedPixels` and setting `ImageId` as index.
    """
    df = pd.read_csv(f"../input/{file}_ship_segmentations.csv")
    df['HasShip'] = df['EncodedPixels'].notnull()
    df = df.groupby("ImageId").agg({'HasShip': ['first', 'sum']}) # counts amount of ships per image, sets ImageId to index
    df.columns = ['HasShip', 'TotalShips']
    return df

# Load the Training Set Information
def show_data_distrib(data):
    total = len(data)
    ship = data['HasShip'].sum()
    no_ship = total - ship
    total_ships = int(data['TotalShips'].sum())
    mean = data.loc[data['HasShip'], 'TotalShips'].mean()
    std_dev = data.loc[data['HasShip'], 'TotalShips'].std()
#     variance = data.loc[data['HasShip'], 'TotalShips'].var()
#     mean_k = data['TotalShips'].mean()
    
    print(' ------ Additional Info ------')
    print(f"Images Found: {total}")
    print(f"Images with ships:    {round(ship/total,2)} ({ship})")
    print(f"Images with no ships: {round(no_ship/total,2)} ({no_ship})")
    print(f"Total Number of Ships Present:  {total_ships}")
    print(f"Mean number of Ships per image with Ship: {mean}")
    print(f"Standard Deviation on Ships per image with Ship: {std_dev}")
#     print(f"Variance on Ships per image with Ship: {variance}")
#     print(f"Mean number of Ships per image with Ship: {mean_k}")

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 8), gridspec_kw = {'width_ratios':[1, 3]})

    # Plot ship/no-ship with a bar plot
    ship_ratio = data['HasShip'].value_counts() / total
    ship_ratio = ship_ratio.rename(index={True: 'Ship', False: 'No Ship'})
    ship_ratio.plot.bar(ax=axes[0], color=['orange', 'blue'], rot=0, title="Ship/No-ship distribution");

    # Plot TotalShips distribution with a bar plot
    total_ships_distribution = data.loc[data['HasShip'], 'TotalShips'].value_counts().sort_index() / ship
#     print(total_ships_distribution)  # debug
    total_ships_distribution.plot(kind='bar', ax=axes[1], rot=0, title="Total ships distribution");
    
    # Testing for annotation
#     test_df = data.loc[data['HasShip'], 'TotalShips'].value_counts().sort_index() / ship
    ax1 = total_ships_distribution.plot(kind='bar')
    x = 0
    y = 0
    for i in range (1, len(total_ships_distribution)+1, 1):
        v='%.4f' % total_ships_distribution[i]  # to 4 decimal places        
        if i==1:
            x=-0.25
        else:
            x = i-1.3
        y = total_ships_distribution[i] + 0.01
        ax1.text(x, y, v, color='green', fontweight='bold', fontsize='12')

# This function transforms EncodedPixels into a list of pixels
# Check our notebook for a detailed explanation:
# https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes
def rle_to_pixels(rle_code):
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768) 
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                 for pixel_position in range(start, start + length)]
    return pixels

def get_pixels_distribution(df):
    """
    Prints the amount of ship and no-ship pixels in the df
    """
    # Total images in the df
    n_images = df['ImageId'].nunique() 
    
    # Total pixels in the df
    total_pixels = n_images * 768 * 768 

    # Keep only rows with RLE boxes, transform them into list of pixels, sum the lengths of those lists
    ship_pixels = df['EncodedPixels'].dropna().apply(rle_to_pixels).str.len().sum() 

    ratio = ship_pixels / total_pixels
    print(f"Ship: {round(ratio, 3)} ({ship_pixels})")
    print(f"No ship: {round(1 - ratio, 3)} ({total_pixels - ship_pixels})")     
    return ship_pixels, total_pixels

    
def image_gen(image_name, rotation):
    # Load the Image
    pilow_load = Image.open(image_name)
    rotate_img = 0
    angle = rotation
    
    # Image Rotation - Local variable
    if angle == 0:
        rotate_img = pilow_load
    elif angle == 90:
        rotate_img = pilow_load.rotate(90)
    elif angle == 180:
        rotate_img = pilow_load.rotate(180)
    elif angle == 270:
        rotate_img = pilow_load.rotate(270)    
    else:
        rotate_img = pilow_load
        print('Error!!! No rotation data specified. Default image Loaded. ')
    
    return rotate_img  

def has_ship(encoded_pixels):
    hs = [0 if pd.isna(n) else 1 for n in tqdm(encoded_pixels)]
    return hs

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

    
"""
    Implementation from  https://github.com/ternaus/robot-surgery-segmentation
"""

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask
    
class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask

class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0, 2)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]

        return img, mask


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2, :]

        return img, mask


class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
            else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
                mask = (msk1[y1:y2, x1:x2, :] if size == size0
                else cv2.resize(msk1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class CenterCrop:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2
        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2,:]
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[y1:y2, x1:x2,:]

        return img, mask
    
class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img


class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
    