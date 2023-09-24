from PIL import Image, ImageDraw
import sys
import numpy as np
import os
import cv2


def image_padding(input_image, item_width, return_padded_image=True):
    """
    Pad an image with white background。
    Args:
        input_image: can be an PIL.Image.Image or numpy array
        item_width: int, patch width to be cut
        return_padded_image: boolean. If true, return padded image.
    Returns:
        image: padded image (only return when return_padded_image is true)
        col: how many columns after padding
        rows: how many rows after padding
    """
    if isinstance(input_image, Image.Image):
        #iw, ih = input_image.size
        iw = np.size(input_image,1)
        ih = np.size(input_image,0)
        col = int(np.ceil(iw/item_width))
        row = int(np.ceil(ih/item_width))  #网络的输入为正方形，所以item_width=item_height
        target_size = (int(col*item_width), int(row*item_width))
        if return_padded_image:
            image = Image.new('RGB', target_size, (255, 255, 255))
            image.paste(input_image, (0, 0))
            return image, col, row
        else:
            return col, row
    elif isinstance(input_image, np.ndarray):
        iw = input_image.shape[1]
        ih = input_image.shape[0]
        col = int(np.ceil(iw / item_width))
        row = int(np.ceil(ih / item_width))  # 网络的输入为正方形，所以item_width=item_height
        target_size = (int(row * item_width), int(col * item_width))
        if return_padded_image:
            image = np.ones(target_size + (3,), dtype='uint8') * 255
            image[:ih, :iw, :] = input_image
            return image, col, row
        else:
            return col, row
    else:
        raise ValueError("Incorrect input image type")


# def image_padding_array(input_image, item_width, return_padded_image=True):
#     #iw, ih = input_image.size
#     iw = input_image.shape[1]
#     ih = input_image.shape[0]
#     col = int(np.ceil(iw/item_width))
#     row = int(np.ceil(ih/item_width))  #网络的输入为正方形，所以item_width=item_height
#     target_size = (int(row*item_width), int(col*item_width))
#     if return_padded_image:
#         image = np.zeros(target_size + (3,), dtype='uint8')
#         image[:ih, :iw, :] = input_image
#         return image, col, row
#     else:
#         return col, row


def cut_image(image, col, row, item_width, return_images=True):
    if isinstance(image, Image.Image):
        width, height = image.size
    elif isinstance(image, np.ndarray):
        width, height = image.shape[1], image.shape[0]
        image = image.copy()
    else:
        raise ValueError("Incorrect input image type")
    box_list = []
    count = 0
    for j in range(0, row):
        for i in range(0, col):
            count += 1
            xMin, yMin, xMax, yMax = i * item_width, j * item_width, (i + 1) * item_width, (j+1) * item_width
            if j == row-1:
                yMax = height
            if i == col-1:
                xMax = width
            box = (xMin, yMin, xMax, yMax)
            box_list.append(box)
#    print(count)
    if return_images:
        if isinstance(image, Image.Image):
            image_list = [image.crop(box) for box in box_list]
        elif isinstance(image, np.ndarray):
            image_list = [image[box[1]:box[3], box[0]:box[2]] for box in box_list]
        return image_list
    else:
        return box


# def cut_image_array(image_arr, col, row, item_width, return_images=True):
#     width, height = image_arr.shape[1], image_arr.shape[0]
#     box_list = []
#     image_arr = image_arr.copy()
#     count = 0
#     for j in range(0, row):
#         for i in range(0, col):
#             count += 1
#             xMin, yMin, xMax, yMax = i * item_width, j * item_width, (i + 1) * item_width, (j+1) * item_width
#             if j == row-1:
#                 yMax = height
#             if i == col-1:
#                 xMax = width
#             box = (xMin, yMin, xMax, yMax)
#             box_list.append(box)
# #    print(count)
#     if return_images:
#         image_list = [image_arr[box[1]:box[3], box[0]:box[2]] for box in box_list]
#         return image_list
#     else:
#         return box


def save_images(image_list, img_name, col):
    #index = 1
    x = 0
    y = 0
    for idx, image in enumerate(image_list):
        y = int(idx/col);
        x = idx%col;
        image.save('./crop_test/' + img_name+ '_'+ str(x)+ '_'+ str(y) + '.jpg')


def detect_image(p_image):
    draw = ImageDraw.Draw(p_image)
    draw.ellipse([20,20,40,40],fill = (0, 0, 0))
    return p_image

def joint_image(patch_image, item_width, col, row):
    target_size = (int(col * item_width), int(row * item_width))
    result_image = Image.new('RGB', target_size, (255, 255, 255))
    for index, s_image in enumerate(patch_image):
        h = int(index/col)
        w = index%col
        result_image.paste(s_image,(w*item_width, h*item_width))
    return result_image




if __name__ == '__main__':
    item_width = 512
    file_path = './crop_test/original_image/'
    img_file = os.listdir(file_path)
    for file in img_file:
        input_image = Image.open(file_path+file)
        image, col, row = image_padding(input_image, item_width)
        image_list = cut_image(image, col, row, item_width)
        result_image_list = []
        for img in image_list:
            p_image = detect_image(img)
            result_image_list.append(p_image)
        result_image = joint_image(result_image_list, item_width, col, row)
        result_image.save('./crop_test/'+file)
    # 打开图像
    #input_image = Image.open('./crop_test/original_image/1.jpg')
    # padding
    #image,col,row=image_padding(input_image, item_width)
    # 分为图像
    #image_list = cut_image(image,col,row,item_width)
    # 保存图像
    #save_images(image_list)
