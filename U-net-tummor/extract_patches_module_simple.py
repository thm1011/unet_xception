'''
    Written in Python 3.7.3
'''
import functools
import math
import multiprocessing
import time
import random
import os
import cv2
import numpy as np
import pandas as pd
import csv
import re
import openslide
import datetime
import tensorflow as tf
import progressbar

from PIL import Image
from skimage.measure import shannon_entropy
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from shapely.geometry import Polygon
from shapely.geometry import Point
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from configparser import ConfigParser

from openslide_python_fix import _load_image_lessthan_2_29, _load_image_morethan_2_29

import gc
import pdb
import sys

'''
Declare the variable names imported from config parser, so that Cython compile 
is possible.
'''

def get_wsi_dim(tif_file_path):
    try:
        wsi_image = OpenSlide(tif_file_path)
        slide_w_, slide_h_ = wsi_image.level_dimensions[level]

    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None
    return slide_w_, slide_h_


'''
    Load slides into memory.
'''


def read_wsi(tif_file_path, level):
    '''
        Identify and load slides.
        Returns:
            - wsi_image: OpenSlide object.
            - rgba_image: WSI image loaded, NumPy array type.
    '''

    time_s = time.time()

    try:
        wsi_image = OpenSlide(tif_file_path)
        # DAB_angle, H_angle = getHDabEstimate(wsi_image)
        DAB_angle = -1
        H_angle = -1
        slide_w_, slide_h_ = wsi_image.level_dimensions[level]

        # Check which _load_image() function to use depending on the size of the region.
        if (slide_w_ * slide_h_) >= 2 ** 29:
            openslide.lowlevel._load_image = _load_image_morethan_2_29
        else:
            openslide.lowlevel._load_image = _load_image_lessthan_2_29

        '''
            The read_region loads the target area into RAM memory, and
            returns an Pillow Image object.

            !! Take care because WSIs are gigapixel images, which are could be 
            extremely large to RAMs.

            Load the whole image in level < 3 could cause failures.
        '''

        # Here we load the whole image from (0, 0), so transformation of coordinates 
        # is not skipped.

        # rgba_image_pil = wsi_image.read_region((0, 0), level, (slide_w_, slide_h_))
        rgba_image = wsi_image.read_region((0, 0), level, (slide_w_, slide_h_))
        # verboseprint("width, height:", rgba_image_pil.size)

        '''
            !!! It should be noted that:
            1. np.asarray() / np.array() would switch the position 
            of WIDTH and HEIGHT in shape.

            Here, the shape of $rgb_image_pil is: (WIDTH, HEIGHT, channel).
            After the np.asarray() transformation, the shape of $rgb_image is: 
            (HEIGHT, WIDTH, channel).

            2. The image here is RGBA image, in which A stands for Alpha channel.
            The A channel is unnecessary for now and could be dropped.
        '''
        # rgba_image = np.asarray(rgba_image_pil)

        # verboseprint("transformed:", rgba_image.shape)

    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None

    time_e = time.time()

    # verboseprint("Time spent on loading", tif_file_path, ": ", (time_e - time_s))

    return rgba_image
    # return wsi_image, rgba_image, (slide_w_, slide_h_), DAB_angle, H_angle


def read_wsi_regions(tif_file_path, patch_size, use_multiprocessing=True, coords=[]):
    """
    Read a slide. Return a list of PIL.Image.Image cut from the slide by patch_size.
    read_region function in OpenSlide can only accept coordinates in level 0.
    Args:
        tif_file_path: str, path to the slide
        patch_size: int, patch size to be cut
    Returns:
        patches: list of PIL.Image.Image, each element is RGBA image
    """
    cols=0
    rows = 0
    try:
        wsi_image = OpenSlide(tif_file_path)
        slide_w_, slide_h_ = wsi_image.level_dimensions[0]

        # Check which _load_image() function to use depending on the size of the region.
        if (slide_w_ * slide_h_) >= 2 ** 29:
            openslide.lowlevel._load_image = _load_image_morethan_2_29
        else:
            openslide.lowlevel._load_image = _load_image_lessthan_2_29
        boxes = []
        if len(coords) == 0:
            for y in range(0, math.ceil(slide_h_ / patch_size)):
                for x in range(0, math.ceil(slide_w_ / patch_size)):
                    xMin, yMin = x * patch_size, y * patch_size
                    patch_width = min(patch_size, slide_w_ - xMin)
                    patch_height = min(patch_size, slide_h_ - yMin)
                    boxes.append((xMin, yMin, patch_width, patch_height))
            rows, cols = math.ceil(slide_h_ / patch_size), math.ceil(slide_w_ / patch_size)
        else:
            for xMin, yMin in coords:
                xMin = min(slide_w_-1, max(0, xMin))
                yMin = min(slide_h_-1, max(0, yMin))
                patch_width = min(patch_size, slide_w_ - xMin)
                patch_height = min(patch_size, slide_h_ - yMin)
                boxes.append((xMin, yMin, patch_width, patch_height))

        def cut(box):
            xMin, yMin, patch_width, patch_height = box
            return wsi_image.read_region((xMin, yMin), 0, (patch_width, patch_height))

        if use_multiprocessing:
            pool = ThreadPool(multiprocessing.cpu_count() - 1)
            patches = pool.map(cut, boxes)
            pool.close()
            pool.join()
        else:
            patches = []
            for box in boxes:
                patches.append(cut(box))
        if len(coords) == 0:
            return patches, slide_w_, slide_h_, cols, rows
        else:
            return patches


    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return []


# def cut(box, wsi_image):
#     xMin, yMin, patch_width, patch_height = box
#     return wsi_image.read_region((xMin, yMin), 0, (patch_width, patch_height))


'''
    Convert RGBA to RGB, HSV and GRAY.
'''
def construct_colored_wsi(rgba_, return_array=False):
    '''
        This function splits and merges R, G, B channels.
        HSV and GRAY images are also created for future segmentation procedure.

        Args:
            - rgba_: Image to be processed, NumPy array type.

    '''
    # r_, g_, b_, a_ = cv2.split(rgba_)

    # wsi_rgb_ = cv2.merge((r_, g_, b_))

    # wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_RGB2GRAY)
    # wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)
    if not return_array:
        r_, g_, b_, a_ = rgba_.split()
        wsi_rgb_ = Image.merge('RGB', [r_, g_, b_])
        return wsi_rgb_
    else:
        rgba_ = np.array(rgba_)
        r_, g_, b_, a_ = cv2.split(rgba_)
        wsi_rgb_ = cv2.merge([r_, g_, b_])
        return wsi_rgb_


'''
'''


def get_contours(cont_img, rgb_image_shape):
    '''
    Args:
        - cont_img: images with contours, these images are in np.array format.
        - rgb_image_shape: shape of rgb image, (HEIGHT, WIDTH).

    Returns:
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of valid region coordinates (contours squeezed);
        - contours: List of valid regions (coordinates);
        - mask: binary mask array;

        !!! It should be noticed that the shape of mask array is: (HEIGHT, WIDTH, channel).
    '''

    # verboseprint('contour image: ',cont_img.shape)

    contour_coords = []
    contours, hiers = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # print(contours)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    for contour in contours:
        contour_coords.append(np.squeeze(contour))

    mask = np.zeros(rgb_image_shape, np.uint8)

    # verboseprint('mask shape', mask.shape)
    '''
    cv2.drawContours(mask, contours, -1, \
                    (PIXEL_WHITE, PIXEL_WHITE, PIXEL_WHITE),thickness=-1)
    '''
    return boundingBoxes, contour_coords, contours, mask


'''
    Perform segmentation and get contours.
'''


def segmentation_hsv(wsi_hsv_, wsi_rgb_):
    '''
    This func is designed to remove background of WSIs.

    Args:
        - wsi_hsv_: HSV images.
        - wsi_rgb_: RGB images.

    Returns:
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of arrays. Each array stands for a valid region and
        contains contour coordinates of that region.
        - contours: Almost same to $contour_coords;
        - mask: binary mask array;

        !!! It should be noticed that:
        1. The shape of mask array is: (HEIGHT, WIDTH, channel);
        2. $contours is unprocessed format of contour list returned by OpenCV cv2.findContours method.

        The shape of arrays in $contours is: (NUMBER_OF_COORDS, 1, 2), 2 stands for x, y;
        The shape of arrays in $contour_coords is: (NUMBER_OF_COORDS, 2), 2 stands for x, y;

        The only difference between $contours and $contour_coords is in shape.
    '''
    # verboseprint("HSV segmentation: ")
    contour_coord = []

    '''
        Here we could tune for better results.
        Currently 20 and 200 are lower and upper threshold for H, S, V values, respectively. 
    
        !!! It should be noted that the threshold values here highly depends on the dataset itself.
        Thresh value could vary a lot among different datasets.
    '''
    lower_ = np.array([20, 20, 20])
    upper_ = np.array([200, 200, 200])

    # HSV image threshold
    thresh = cv2.inRange(wsi_hsv_, lower_, upper_)

    try:
        # verboseprint("thresh shape:", thresh.shape)
        print("thresh shape:", thresh.shape)
    except:
        # verboseprint("thresh shape:", thresh.size)
        print("thresh shape:", thresh.size)
    else:
        pass

    '''
        Closing
    '''
    # verboseprint("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(thresh), cv2.MORPH_CLOSE, close_kernel)
    # verboseprint("image_close size", image_close.shape)

    '''
        Openning
    '''
    # verboseprint("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    # verboseprint("image_open size", image_open.size)

    # verboseprint("Getting Contour: ")
    bounding_boxes, contour_coords, contours, mask \
        = get_contours(np.array(image_open), wsi_rgb_.shape)

    return bounding_boxes, contour_coords, contours, mask


'''
    Construct annotation polygons from csv's'.
'''


def get_annotations(csv_path):
    '''
    Args:
        -csv_path: filepath of the annotation csv's

    Returns: 
        - polygons: lists of polygons defined by rows of the csv's.
        - polygon_types: categories of the annotation polygons. 
    '''
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        polygons = []
        polygon_types = []
        for row in csvReader:
            pointList = []
            for idx in np.arange(np.int((len(row) - 1) / 2)):
                pi = Point(int(float(re.findall("\d+\.\d+", row[2 * idx + 1])[0])),
                           int(float(re.findall("\d+\.\d+", row[2 * idx + 2])[0])))
                pointList.append(pi)
            polyi = Polygon([[p.x, p.y] for p in pointList])
            polygons.append(polyi)
            if row[0] == 'Tumor positive':
                polygon_types.append(1)
            elif row[0] == 'immune positive':
                polygon_types.append(2)
            elif row[0] == 'Tumor negative':
                polygon_types.append(3)
            elif row[0] == 'mix':
                polygon_types.append(5)
            else:
                polygon_types.append(4)
    return polygons, polygon_types


def get_patches(wsi_, wsi_rgb, level, mag_factor, patch_size, offset_x=0, offset_y=0):
    '''
    Extract All patches.
    Arguments:
        - wsi_: whole slide image.
        - wsi_rbg: whole slide image in rgb channels.
	    - level: magnification level, in exponential of 2, 0 for maximum level.
	    - patch_size: predefined patch-size in (x_width_, y_height_)
	    - offset: Integer, the starting point of patch extraction, the point would be (offset, offset) in the image, with default value (0, 0).

    Returns: 
        - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, channel]
        - patches_coords: coordinates of patches: (x_min, y_min). The bouding box of the patch
        is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
    '''
    patches = []
    patches_coords = []

    X = np.arange(offset_x, wsi_.dimensions[0], step=patch_size)
    Y = np.arange(offset_y, wsi_.dimensions[1], step=patch_size)

    # In the case that WSI dimensions are not multiples of the patch_size, remove the last element of X or Y
    if wsi_.dimensions[0] % patch_size != 0 or (wsi_.dimensions[0] - X[-1]) < patch_size:
        X = X[:-1]
    if wsi_.dimensions[1] % patch_size != 0 or (wsi_.dimensions[1] - Y[-1]) < patch_size:
        Y = Y[:-1]

    for h_pos, y_height_ in tqdm(enumerate(Y), desc='Extraction Progress', ncols=100, miniters=20):

        for w_pos, x_width_ in enumerate(X):

            # Read again from WSI object wastes tooooo much time.
            # patch_img = wsi_.read_region((x_width_, y_height_), level, (patch_size, patch_size))

            '''
                !!! Take care of difference in shapes
                Here, the shape of wsi_rgb is (HEIGHT, WIDTH, channel)
                the shape of mask is (HEIGHT, WIDTH, channel)
            '''
            patch_arr = wsi_rgb[y_height_: y_height_ + patch_size, \
                        x_width_:x_width_ + patch_size, :]
            # patches.append(patch_arr)
            # patches_coords.append((x_width_,y_height_))

            width_mask = x_width_
            height_mask = y_height_

            # patch_mask_arr = mask[height_mask: height_mask + patch_size, \
            #                          width_mask: width_mask + patch_size]

            '''
            bitwise_grey is the gray-scles image of the patch, in which white represent non-informative pixel value
            . After bitwise_not, staining represent non-informative pixel value.
            '''
            # bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)
            # bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
            # bitwise_grey_inv = cv2.bitwise_not(bitwise_grey)
            # white_pixel_cnt = cv2.countNonZero(bitwise_grey)
            '''
            white_pixel_cnt = np.sum(bitwise_grey >= 160)
            grey_pixel_cnt = cv2.countNonZero(bitwise_grey)
            
            if grey_pixel_cnt >= ((patch_size ** 2) * 0.15):
                patches.append(patch_arr)
                patches_coords.append((x_width_,y_height_))
            '''
            im_ = Image.fromarray(patch_arr)
            entropy_ = shannon_entropy(im_)
            if entropy_ > 5:
                patches.append(patch_arr)
                patches_coords.append((x_width_, y_height_))

    return patches, patches_coords


''' 
    Extract Valid patches.
'''


def construct_bags(wsi_, wsi_rgb, contours, mask, level, mag_factor, patch_size):
    '''
    Args:
        To-do.

    Returns: 
        - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, channel]
        - patches_coords: coordinates of patches: (x_min, y_min). The bouding box of the patch
        is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
    '''

    patches = []
    patches_coords = []

    start = time.time()

    '''
        !!! 
        Currently we select only the first 5 regions, because there are too many small areas and 
        too many irrelevant would be selected if we extract patches from all regions.

        And how many regions from which we decide to extract patches is 
        highly related to the SEGMENTATION results.

    '''
    contours_ = sorted(contours, key=cv2.contourArea, reverse=True)
    contours_ = contours_[:5]

    for i, box_ in enumerate(contours_):

        box_ = cv2.boundingRect(np.squeeze(box_))
        # verboseprint('region', i)

        '''

        !!! Take care of difference in shapes:

            Coordinates in bounding boxes: (WIDTH, HEIGHT)
            WSI image: (HEIGHT, WIDTH, channel)
            Mask: (HEIGHT, WIDTH, channel)

        '''

        b_x_start = int(box_[0])
        b_y_start = int(box_[1])
        b_x_end = int(box_[0]) + int(box_[2])
        b_y_end = int(box_[1]) + int(box_[3])

        '''
            !!!
            step size could be tuned for better results.
        '''

        X = np.arange(b_x_start, b_x_end, step=patch_size // 2)
        Y = np.arange(b_y_start, b_y_end, step=patch_size // 2)

        # verboseprint('ROI length:', len(X), len(Y))

        for h_pos, y_height_ in enumerate(Y):

            for w_pos, x_width_ in enumerate(X):

                # Read again from WSI object wastes tooooo much time.
                # patch_img = wsi_.read_region((x_width_, y_height_), level, (patch_size, patch_size))

                '''
                    !!! Take care of difference in shapes
                    Here, the shape of wsi_rgb is (HEIGHT, WIDTH, channel)
                    the shape of mask is (HEIGHT, WIDTH, channel)
                '''
                patch_arr = wsi_rgb[y_height_: y_height_ + patch_size, \
                            x_width_:x_width_ + patch_size, :]
                # verboseprint("read_region (scaled coordinates): ", x_width_, y_height_)

                width_mask = x_width_
                height_mask = y_height_

                patch_mask_arr = mask[height_mask: height_mask + patch_size, \
                                 width_mask: width_mask + patch_size]

                # verboseprint("Numpy mask shape: ", patch_mask_arr.shape)
                # verboseprint("Numpy patch shape: ", patch_arr.shape)

                try:
                    bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)

                except Exception as err:
                    print('Out of the boundary')
                    pass

                #                     f_ = ((patch_arr > PIXEL_TH) * 1)
                #                     f_ = (f_ * PIXEL_WHITE).astype('uint8')

                #                     if np.mean(f_) <= (PIXEL_TH + 40):
                #                         patches.append(patch_arr)
                #                         patches_coords.append((x_width_, y_height_))
                #                         print(x_width_, y_height_)
                #                         print('Saved\n')

                else:
                    bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
                    white_pixel_cnt = cv2.countNonZero(bitwise_grey)

                    '''
                        Patches whose valid area >= 25% of total area is considered
                        valid and selected.
                    '''

                    if white_pixel_cnt >= ((patch_size ** 2) * 0.25):

                        if patch_arr.shape == (patch_size, patch_size, channel):
                            patches.append(patch_arr)
                            patches_coords.append((x_width_, y_height_))
                            # verboseprint(x_width_, y_height_)
                            # verboseprint('Saved\n')

                    else:
                        # verboseprint('Did not save\n')
                        pass

    end = time.time()
    # verboseprint("Time spent on patch extraction: ",  (end - start))

    # patches_ = [patch_[:,:,:3] for patch_ in patches] 
    print("Total number of patches extracted:", len(patches))

    return patches, patches_coords


'''
    Save annotated patches to disk.
'''


def save_annotation_to_disk(patches, patches_coords, level, polygons, polygon_types, wsi_name, patch_size):
    '''
        The paths should be changed
    '''

    # case_name = slide_.split('/')[-1].split('.')[0]

    # patch_coords_dst = './dataset_patches/' + case_name + '/level' + str(level) + '/'
    # array_file = patch_array_dst + 'patch_'
    # array_file = 'C:/Users/yishi.xing/Desktop/data_patches/' + 'patch_'

    time = datetime.datetime.now()

    # Adaptively change the folder name with the current time.
    dst_dir = './Annotated_patches' + '_' + str(time.month) + '_' + str(time.day) + '_' + str(patch_size) + '/'
    array_file_TP = 'TumorPositive/patch_'
    array_file_IP = 'ImmunePositive/patch_'
    array_file_TN = 'TumorNegative/patch_'
    array_file_Other = 'Other/patch_'
    dst_dir_TP = dst_dir + 'TumorPositive/'
    dst_dir_IP = dst_dir + 'ImmunePositive/'
    dst_dir_TN = dst_dir + 'TumorNegative/'
    dst_dir_Other = dst_dir + 'Other/'

    # coords_file = patch_coords_dst + 'patch_coords.csv'
    # mask_file = patch_coords_dst + 'mask'

    # if not os.path.exists(patch_array_dst):
    #    #os.makedirs(patch_array_dst)
    #    verboseprint('mkdir', patch_array_dst)

    # if not os.path.exists(prefix_dir):
    #    #os.makedirs(prefix_dir)
    #    verboseprint('mkdir', prefix_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(dst_dir_TP):
        os.makedirs(dst_dir_TP)
    if not os.path.exists(dst_dir_IP):
        os.makedirs(dst_dir_IP)
    if not os.path.exists(dst_dir_TN):
        os.makedirs(dst_dir_TN)
    if not os.path.exists(dst_dir_Other):
        os.makedirs(dst_dir_Other)
    # verboseprint('Path: ', array_file)
    # verboseprint('Path: ', coords_file)
    # verboseprint('Path: ', mask_file)
    # verboseprint('Number of patches: ', len(patches_coords))
    # verboseprint(patches_coords[:5])

    '''
        Save coordinates to the disk. Here we use pandas DataFrame to organize 
        and save coordinates.
    '''

    # df1_ = pd.DataFrame([x[0] for x in patches_coords], columns = ["coord_x"])
    # df1_["coord_y"] = [y[1] for y in patches_coords]
    # df1_.to_csv(coords_file, encoding='utf-8', index=False)

    '''
    Save patch arrays to the disk
    '''
    # patch_whole = np.array(patches1).shape

    for i, patch_ in enumerate(patches):
        patch_name = dst_dir
        x_, y_ = patches_coords[i]
        patch_polygon = Polygon(
            [(x_, y_), (x_, y_ + patch_size), (x_ + patch_size, y_ + patch_size), (x_ + patch_size, y_)])
        '''
        Classify the patches into 4 classes by intersection with annotation.
        Store the max intersection area between the current patch and the i-th annotation class.
        '''

        max_intersection = []
        for class_i in np.arange(4):
            anno_indices_class_i = [i for i, x in enumerate(polygon_types) if x == (class_i + 1)]
            if len(anno_indices_class_i) == 0:
                # print('No class ', class_i+1, 'annotated')
                max_intersection.append(0)
                continue
            else:
                intersection_area = []
                for index in anno_indices_class_i:
                    if polygons[index].is_valid:
                        area_i = patch_polygon.intersection(polygons[index]).area
                    else:
                        polygon_anno_fixed = polygons[index].buffer(0)
                        area_i = patch_polygon.intersection(polygon_anno_fixed).area
                    intersection_area.append(area_i)
                max_intersection.append(max(intersection_area))
        # print(max_intersection)
        if max_intersection[0] >= (patch_size ** 2 * 0.1):
            patch_name = patch_name + array_file_TP + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
            patch_name = patch_name + '_TumorPositive'
            im = Image.fromarray(patch_)
            im.save(patch_name + '.jpeg')
        elif max_intersection[1] >= (patch_size ** 2 * 0.1):
            patch_name = patch_name + array_file_IP + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
            patch_name = patch_name + '_ImmunePositive'
            im = Image.fromarray(patch_)
            im.save(patch_name + '.jpeg')
        elif max_intersection[2] >= (patch_size ** 2 * 0.2):
            patch_name = patch_name + array_file_TN + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
            patch_name = patch_name + '_TumorNegative'
            im = Image.fromarray(patch_)
            im.save(patch_name + '.jpeg')
            print(max_intersection)
            print(patch_name)
        elif max_intersection[3] >= (patch_size ** 2 * 0.95):
            patch_name = patch_name + array_file_Other + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
            patch_name = patch_name + '_Other'
            im = Image.fromarray(patch_)
            im.save(patch_name + '.jpeg')
        else:
            patch_name = patch_name + 'patch_' + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
        # np.save(patch_name, np.array(patch_))


'''
    Save patches to disk.
'''


def save_to_disk(patches, patches_coords, level, wsi_name, patch_size):
    '''
        The paths should be changed
    '''

    # case_name = slide_.split('/')[-1].split('.')[0]
    # prefix_dir = './dataset_patches/' + case_name + '/level' + str(level) + '/'
    # patch_array_dst = './dataset_patches/' + case_name + '/level' + str(level) + '/patches/'

    # patch_coords_dst = './dataset_patches/' + case_name + '/level' + str(level) + '/'
    # array_file = patch_array_dst + 'patch_'
    # array_file = 'C:/Users/yishi.xing/Desktop/data_patches/' + 'patch_'

    # Adaptively change the folder name with the current time.
    dst_dir = result_root_path + 'patchwise_' + str(TIME_CURRENT.month) + '_' + str(TIME_CURRENT.day) + '_' + str(
        patch_size) + '/'

    # coords_file = patch_coords_dst + 'patch_coords.csv'
    # mask_file = patch_coords_dst + 'mask'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        # verboseprint('mkdir', dst_dir)
    # verboseprint('Path: ', array_file)
    # verboseprint('Path: ', coords_file)
    # verboseprint('Path: ', mask_file)
    # verboseprint('Number of patches: ', len(patches_coords))
    # verboseprint(patches_coords[:5])

    '''
        Save coordinates to the disk. Here we use pandas DataFrame to organize 
        and save coordinates.
    '''

    # df1_ = pd.DataFrame([x[0] for x in patches_coords], columns = ["coord_x"])
    # df1_["coord_y"] = [y[1] for y in patches_coords]
    # df1_.to_csv(coords_file, encoding='utf-8', index=False)

    '''
    Save patch arrays to the disk
    '''
    # patch_whole = np.array(patches1).shape

    for i, patch_ in enumerate(patches):
        patch_name = dst_dir
        patch_name = dst_dir + wsi_name + '/data/'
        if not os.path.exists(patch_name):
            os.makedirs(patch_name)
        x_, y_ = patches_coords[i]
        patch_polygon = Polygon(
            [(x_, y_), (x_, y_ + patch_size), (x_ + patch_size, y_ + patch_size), (x_ + patch_size, y_)])

        patch_name = patch_name + 'patch_' + wsi_name + '_' + str(i) + '_' + str(x_) + '_' + str(y_)
        im = Image.fromarray(patch_)
        im.save(patch_name + '.jpeg')

        # np.save(patch_name, np.array(patch_))

    # Save whole patches: convert list of patches to array.
    # shape: (NUMBER_OF_PATCHES, PATCH_WIDTH, PATCH_HEIGHT, channel)

    # patch_whole = prefix_dir + 'patch_whole'
    # np.save(patch_whole, np.array(patches))

    '''
    Save mask file to the disk
    '''
    # mask_img = Image.fromarray(mask)
    # mask_img.save(mask_file + '.jpeg')


'''
    The whole pipeline of extracting patches.
'''


def extract_(slide_, level, mag_factor):
    '''
    Args:
        slide_: path to target slide.
        level: magnification level. 
        mag_factor: pow(2, level).

    Returns: 
        To-do.
    '''

    start = time.time()

    wsi_, rgba_, shape_ = read_wsi(slide_, level)

    wsi_name = os.path.basename(slide_)
    wsi_name = wsi_name.split('.')[0]
    wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_)

    print('Transformed shape: (height, width, channel)')
    # verboseprint("WSI HSV shape: ", wsi_hsv_.shape)
    print("WSI RGB shape: ", wsi_rgb_.shape)
    # verboseprint("WSI GRAY shape: ", wsi_gray_.shape)
    # verboseprint('\n')

    del rgba_
    gc.collect()

    # bounding_boxes, contour_coords, contours, mask \
    # = segmentation_hsv(wsi_hsv_, wsi_rgb_)

    del wsi_hsv_
    gc.collect()

    # patches, patches_coords = construct_bags(wsi_, wsi_rgb_, contours, mask, \
    #                                        level, mag_factor, patch_size)

    annotation_dir_prefix = annotation_folder + '/output'
    annotation_file = annotation_dir_prefix + wsi_name + '.csv'
    if os.path.exists(annotation_file):
        # verboseprint("annotated!")
        polygons, polygon_types = get_annotations(annotation_file)
        patches, patches_coords = get_patches(wsi_, wsi_rgb_, level, mag_factor, patch_size)
        save_annotation_to_disk(patches, patches_coords, level, polygons, polygon_types, wsi_name, patch_size)
    else:
        # verboseprint("unannotated, extract all!")
        patches, patches_coords = get_patches(wsi_, wsi_rgb_, level, mag_factor, patch_size)
        end_mem = time.time()
        # verboseprint("Time spent before saving to disk: ", (end_mem - start))
        save_to_disk(patches, patches_coords, level, wsi_name, patch_size)

        end = time.time()
        print("Time spent on patch extraction: ", (end - start))
        print(slide_)

        return patches, patches_coords


'''
    The single input version of extract_, to facilitate multiprocessing.
'''


def extract_simpleinput_(filename):
    # return extract_('./WSIs/' + filename[6:14] + '.tiff', 0, 1, os.path.dirname(annotation_file) + '/output')
    patches, patches_coords = extract_(rawdata_root_dir + filename + '.tiff', 0, 1)


'''
    The simpler :version of extract_, to enable real-time patch extraction, without writing to disk.
    Input:
        filename: The name of the WSI to be extracted.
    Output:
        patches: The list of patches(Numpy array).
'''


def extract_patches_(filename, offset_x=0, offset_y=0):
    level = 0
    mag_factor = pow(2, level)
    start = time.time()

    wsi_, rgba_, shape_, DAB_angle, H_angle = read_wsi(filename, level)

    wsi_name = os.path.basename(filename)
    wsi_name = wsi_name.split('.')[0]
    wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_)

    del rgba_
    gc.collect()

    # bounding_boxes, contour_coords, contours, mask \
    # = segmentation_hsv(wsi_hsv_, wsi_rgb_)

    del wsi_hsv_
    gc.collect()

    patches, patches_coords = get_patches(wsi_, wsi_rgb_, level, mag_factor, patch_size, offset_x=offset_x,
                                          offset_y=offset_y)
    if save_patches:
        save_to_disk(patches, patches_coords, level, wsi_name, patch_size)

    end = time.time()

    print("Time spent on patch extraction: ", (end - start))
    return patches, patches_coords, shape_[0], shape_[1], DAB_angle, H_angle, wsi_rgb_


'''
'''


def extract_patches_tf_(filename):
    start = time.time()

    level = 0
    mag_factor = pow(2, level)
    start = time.time()

    wsi_, rgba_, shape_ = read_wsi(filename, level)

    wsi_name = os.path.basename(filename)
    wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_)

    del rgba_
    gc.collect()

    bounding_boxes, contour_coords, contours, mask \
        = segmentation_hsv(wsi_hsv_, wsi_rgb_)

    del wsi_hsv_
    gc.collect()

    wsi_image = np.asarray(wsi_rgb_)

    tfImage = tf.convert_to_tensor(wsi_image)

    ksize_rows = patch_size
    ksize_cols = patch_size

    strides_rows = patch_size
    strides_cols = patch_size

    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image.
    strides = [1, strides_rows, strides_cols, 1]

    rates = [1, 1, 1, 1]

    # padding algorithm to use
    padding = 'VALID'  # or 'SAME'

    tfImage = tf.expand_dims(tfImage, 0)

    image_patches = tf.extract_image_patches(tfImage, ksizes, strides, rates, padding)

    end = time.time()
    print("Time spent on patch extraction: ", (end - start))
    return image_patches


def main_list(file_list):
    pool = Pool(os.cpu_count())
    print(file_list)
    pool.map(extract_simpleinput_, file_list)


def main(wsi_csv):
    pool = Pool(os.cpu_count())
    wsi_root_dir = '/WSIs'
    filedf = pd.read_csv(wsi_csv)
    file_list = list(filedf.iloc[:, 0])
    print(file_list)
    pool.map(extract_simpleinput_, file_list)


# for filename in os.listdir(annotation_dir):
# patches, patches_coords, masks = extract_('./WSIs/' + filename[6:14] + '.tiff', 0, 1, annotation_dir + '/output')
# print(filename[6:14] + '.tiff')

# gc.collect()
'''
if __name__=='__main__':
	sys.exit(main(sys.argv[1]))
'''

