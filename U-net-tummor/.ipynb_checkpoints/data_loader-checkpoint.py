import math
import tensorflow as tf
is_tf2 = tf.__version__.startswith("2")

if is_tf2:
    from tensorflow import keras
else:
    import keras

import numpy as np
from PIL import Image


class TestGenerator(keras.utils.Sequence):
    def __init__(self, imgs_list, image_width, batch_size, is_cls=True):
        """
        Data generator for model. Input is list of images
        Args:
            imgs_list: list of PIL Image or numpy array
            batch_size:
            is_cls: different preprocess for classification and unet
        """
        self.imgs_list = imgs_list
        self.image_width = image_width
        self.batch_size = batch_size
        self.is_cls = is_cls
        # self.indexes = np.arange(len(self.imgs_list))

    def __len__(self):
        """
        Demonstrate the number of batches per epoch
        """
        return math.ceil(len(self.imgs_list) / self.batch_size)

    def __getitem__(self, index):
        """
        Generate a batch of data
        """
        # indexes of this batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # generate batch data
        X = self.__data_generation(index)
        return X

    def __data_generation(self, index):
        # X: (batch_size, img_h, img_w, channels)
        imgs_data = self.imgs_list[index * self.batch_size: (index + 1) * self.batch_size]
        # for i in indexes:
        #     img = np.array(self.imgs_list[i])
        #     # if img.shape[0] != self.image_width or img.shape[1] != self.image_width:
        #     #     img_pad = np.ones((self.image_width, self.image_width, 3)) * 255
        #     #     img_pad[:img.shape[0], :img.shape[1]] = img
        #     #     imgs_data.append(img_pad)
        #     # else:
        #     imgs_data.append(img)

        imgs_data = np.array(imgs_data, dtype='float32') / 255.0
        if self.is_cls:
            imgs_data = (imgs_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return imgs_data


class TestGeneratorFromLocs(keras.utils.Sequence):
    def __init__(self, wsi, locs, image_width, batch_size, is_cls=True):
        """
        Data generator for model. Input a whole WSI image and locations need to be cut.
        Args:
            wsi: a WSI image
            locs: list of patches locations
            image_width: model input image size
        """
        self.wsi = np.array(wsi) if (isinstance(wsi, Image.Image)) else wsi
        self.locs = locs
        self.image_width = image_width
        self.batch_size = batch_size
        self.is_cls = is_cls
        self.indexes = np.arange(len(self.imgs_list))

    def __len__(self):
        """
        Demonstrate the number of batches per epoch
        """
        return math.ceil(len(self.imgs_list) / self.batch_size)

    def __getitem__(self, index):
        """
        Generate a batch of data
        """
        # indexes of this batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # generate batch data
        X = self.__data_generation(indexes)
        return X

    def __data_generation(self, indexes):
        # X: (batch_size, img_h, img_w, channels)
        imgs_data = []
        for i in indexes:
            xMin, yMin, xMax, yMax = self.locs[i]
            img = self.wsi[yMin:yMax, xMin:xMax]
            if img.shape[0] != self.image_width or img.shape[1] != self.image_width:
                img_pad = np.ones((self.image_width, self.image_width, 3)) * 255
                img_pad[:img.shape[0], :img.shape[1], 3] = img
                imgs_data.append(img_pad)
            else:
                imgs_data.append(img)

        imgs_data = np.array(imgs_data)
        if self.is_cls:
            imgs_data = (imgs_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return imgs_data