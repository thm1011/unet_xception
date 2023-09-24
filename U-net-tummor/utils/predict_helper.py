# from unet import Unet
from PIL import Image, ImageFile
import numpy as np
import os
import csv, glob
import datetime
import re
import time
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re
import math
import csv
import numpy as np
import math
import cv2
from typing import List
from openslide import OpenSlide
import skimage.morphology as morph
import multiprocessing
from multiprocessing.pool import ThreadPool


def get_best_model(model_dir):
    models = os.listdir(model_dir)
    best_file = ''
    best_loss = float('inf')
    for m in models:
        match = re.search(r'val_loss(.*).h5', m, re.M | re.I)
        if match is None:
            continue
        val_loss = float(match.group(1))
        if val_loss < best_loss:
            best_file = m
    return best_file


def find_overlap_patches(patch_list, i, patch_name, xMin, yMin, xMax, yMax):
    def is_overlap(element):
        _, p, x1, y1, x2, y2 = element
        left_col_max = max(x1, xMin)
        right_col_min = min(x2, xMax)
        up_row_max = max(y1, yMin)
        down_row_min = min(y2, yMax)
        return not (left_col_max >= right_col_min or down_row_min <= up_row_max or patch_name == p)

    return pd.DataFrame(list(filter(is_overlap, patch_list)))


def FillHole(im_in):
    """
    要填充白色目标物中的黑色空洞
    Args:
        im_in: 图像为二值化图像，255白色为目标物，0黑色为背景
    Returns: 填洞后的image

    """
    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    return im_out


# find contours
# im_in is a gray scale image
def find_contours_and_filter(im_in, canvas, area=None, return_mask=True, get_count=False):
    """
    find contours and filter contours with area
    Args:
        im_in: mask图
        canvas: 最终画布
        area: 面积，小于该面积的轮廓将被去除
        return_mask: 是否返回过滤之后的mask图
    Returns:
        最终画布, mask
    """

    contours, hierarchy = cv2.findContours(im_in, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered_idx = []
    tls_counts = 0
    for h, cnt in enumerate(contours):
        if area is not None:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > area:
                tls_counts += 1
                canvas = cv2.drawContours(canvas, contours, h, (255, 255, 0), 3)
                filtered_idx.append(h)
        else:
            canvas = cv2.drawContours(canvas, contours, h, (255, 255, 0), 3)

    if return_mask:
        height, width = im_in.shape
        img_new = np.zeros((height, width, 3))
        img_new[..., 0] = im_in
        img_new[..., 1] = im_in
        img_new[..., 2] = im_in
        img_new = img_new.astype('uint8')
        for idx in filtered_idx:
            img_new = cv2.drawContours(img_new, contours, idx, (0, 0, 255), -1)
        if get_count:
            return img_new, canvas, tls_counts
        else:
            return img_new, canvas
    else:
        if get_count:
            return canvas, tls_counts
        else:
            return canvas


def get_minimunm_area(area_file, downsample):
    """
    Get the minimum area in area_file
    Args:
        area_file:
        downsample:

    Returns:

    """
    areas = []
    with open(area_file, newline='', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            for i in range(len(row)):
                if i == 0:
                    continue
                areas.append(float(row[i]))
    areas = [x / (downsample ** 2) for x in areas]
    return min(areas)


def cut_wsi(img_file: str, downsample_factor, patch_size: int, stride=0):
    slide = OpenSlide(img_file)
    downsamples = list(map(lambda x: round(x, 2), slide.level_downsamples))
    downsample_idx = -1
    for d, downsample in enumerate(downsamples):
        if abs(downsample - downsample_factor) < 0.1:
            downsample_idx = d
            break

    if downsample_idx == -1:
        return None, None, None

    img_width, img_height = slide.level_dimensions[downsample_idx]
    slide_5x = slide.read_region((0, 0), downsample_idx, (img_width, img_height))
    img = np.array(slide_5x)[:, :, :3]  # rgb array

    patch_list, pos_list = [], []  # list storing patch img(numpy array) and position 'xMin, yMin, xMax, yMax'
    if stride == 0:
        nrows = math.ceil(img_height / patch_size)
        ncols = math.ceil(img_width / patch_size)
        # print(nrows, ncols)
    else:
        nrows = math.ceil((img_height - patch_size) / stride + 1)
        ncols = math.ceil((img_width - patch_size) / stride + 1)
    # print(nrows*ncols)
    for row in range(nrows):
        for col in range(ncols):
            if stride == 0:
                y, x = row * patch_size, col * patch_size
            else:
                y, x = row * stride, col * stride
            patch = img[y:y + patch_size, x:x + patch_size]
            patch_list.append(patch)
            key = '{},{},{},{}'.format(x, y, x + patch.shape[1], y + patch.shape[0])
            pos_list.append(key)
    return img, patch_list, pos_list


def concat_patches(imgs: List, patch_size, pos_list: List):
    """
    imgs和pos_list的顺序是一致的
    """
    positions = []
    for i in range(len(pos_list)):
        pos = pos_list[i].split(',')
        # tmp is a list with xMin, yMin, xMax, yMax
        tmp = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        positions.append(tmp)
    ori_width, ori_height, = positions[-1][2:]
    if len(imgs[0].shape) == 3:
        new_img = np.zeros((ori_height, ori_width, 3), dtype='uint8')
    else:
        new_img = np.zeros((ori_height, ori_width), dtype='uint8')

    for i, img in enumerate(imgs):
        new_img[positions[i][1]:positions[i][3], positions[i][0]:positions[i][2]] = img
    return new_img


def find_overlap_patches_wsi(pos_list, i, xMin, yMin, xMax, yMax):
    def is_overlap(element):
        ind, x1, y1, x2, y2 = element
        left_col_max = max(x1, xMin)
        right_col_min = min(x2, xMax)
        up_row_max = max(y1, yMin)
        down_row_min = min(y2, yMax)
        return not (left_col_max >= right_col_min or down_row_min <= up_row_max or ind == i)

    return pd.DataFrame(list(filter(is_overlap, pos_list)))


def is_background_patch(patch_img):
    return patch_img.max() < 1


def predict(model, slide_path: str, patch_size: int, stride: int, downsample: float, min_area, method='union', batch=4):
    """
    TLS segmentation on WSI (5x or 10x). Given a 5x or 10x WSI grayscale (original pixel) images, perform TLS segmentation.
    Args:
        model: a keras model
        slide_path: str
        patch_size: size of patch in 5x or 10x
        stride: stride in 5x or 10x
        downsample: downsample factor
        min_area: minimum area in 5x or 10x to filter small objects
        method: how to merge patch result

    Returns:

    """
    wsi, patch_list, positions = cut_wsi(slide_path, downsample, patch_size, stride=stride)

    if wsi is None:
        pass

    slide_name = os.path.basename(slide_path)
    patches_nonoverlap_idx = []
    pos_list = []
    for i in range(len(positions)):
        pos = positions[i].split(',')
        # tmp is a list with index, xMin, yMin, xMax, yMax
        tmp = [i, int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        if tmp[1] % patch_size == 0 and tmp[2] % patch_size == 0:
            # start_patch = i
            patches_nonoverlap_idx.append(i)
        pos_list.append(tmp)
    print(slide_name, len(pos_list), end='\t')

    mask_img = np.zeros((wsi.shape[:2]))
    # print(mask_img.shape)
    ####################################################################
    # 获得patch_list中每个patch的mask
    ####################################################################
    masks_list = []
    # batch = 8
    start = time.time()
    fg_patches = 0
    for i in range(0, len(patch_list), batch):
        imgs = []
        foreground_idx = []
        mask = []
        for j in range(i, min(i + batch, len(patch_list))):
            img = patch_list[j]
            img_pad = img.copy()
            if img.shape[0] != patch_size or img.shape[1] != patch_size:
                img_pad = np.zeros((patch_size, patch_size, 3), dtype='uint8')
                img_pad[0:img.shape[0], 0:img.shape[1], :] = img
            imgs.append(img_pad / 255.0)

        imgs = np.asarray(imgs)
        mask = model.predict(imgs)
        if isinstance(mask, np.ndarray):
            mask = mask.argmax(axis=-1)
            mask = (mask == 1)
        # filter by area
        for j in range(i, min(i + batch, len(patch_list))):
            h, w = pos_list[j][4] - pos_list[j][2], pos_list[j][3] - pos_list[j][1]
            m = morph.remove_small_objects(mask[j-i], min_area)  # remove small object
            m = m.astype('int')
            masks_list.append(m[:h, :w])


    print('unet time: ', time.time() - start, end='\t')
    # print('patches: ', fg_patches)
    # return masks_list, patches_nonoverlap_idx, patch_list

    ####################################################################
    # 1. 对non-overlap patch的mask进行union等计算
    ####################################################################
    if method == 'union':
        for i, mask in enumerate(masks_list):
            _, x1_p, y1_p, x2_p, y2_p = pos_list[i]
            mask_img[y1_p:(y1_p + mask.shape[0]), x1_p: (x1_p + mask.shape[1])] = \
                mask_img[y1_p:(y1_p + mask.shape[0]), x1_p: (x1_p + mask.shape[1])] + mask
        mask_img = np.clip(mask_img, 0, 1)
        mask_img = mask_img.astype('uint8')

    elif method == 'max':
        pass

    return wsi, mask_img


def filter_5x_with_10x(mask_5x, mask_10x, min_area_5x):
    """
    Determine if the object in mask 5x is true positive
    Args:
        mask_5x:
        mask_10x:
        min_area_5x:

    Returns:

    """
    mask_10x_to5 = cv2.resize(mask_10x, (mask_5x.shape[1], mask_5x.shape[0]))
    intersections = (mask_5x * mask_10x_to5).sum()
    print(intersections)
    return intersections >= min_area_5x






