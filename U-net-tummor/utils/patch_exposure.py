import numpy as np
import cv2


def mean_top_pixel(image, top_percent=0.1):
    """
    Calculate mean pixel value of top n pixels in image
    Args:
        image: an grayscale image
        top_percent: 
    Returns: mean 

    """
    assert len(image.shape) < 3
    array = image.reshape(-1).copy()
    array.sort()
    img_shape = image.shape
    b = array[int(img_shape[0]*img_shape[1]*(1-top_percent)):]
    return b.mean()


# image adjust efficient
def imadjust(img:np.ndarray, coefficient):
    """
    Linear brightness adjustment for an image 
    Args:
        img: image
        coefficient: brightness coefficient

    Returns: adjusted image

    """
    trans = img.astype('float')* coefficient
    trans = np.clip(trans, 0, 255)
    trans = trans.astype('uint8')
    return trans


# 3.determine adjust coefficient
def judgment(img, mean=None):
    """
    Determin coefficient for an image and return transformed image
    Args:
        img: 

    Returns:

    """
    if mean is None:
        mean = mean_top_pixel(img)
    # 0-15
    if mean < 5:
        img = imadjust(img, 4)
    # 15-25
    elif mean < 15:
        img = imadjust(img, 6)
    elif mean < 25:
        img = imadjust(img, 4)
    elif mean < 45:
        img = imadjust(img, 3)
    elif mean < 60:
        img = imadjust(img, 1.2)
    # 1 OR great than 60
    else:
        img = img

    return img


def check_grayscale(img):
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img
    
    
def mergeGray2RGB(img_arr_690:np.ndarray, img_arr_620:np.ndarray):
    if len(img_arr_690.shape) == 3:
        img_arr_690 = cv2.cvtColor(img_arr_690, cv2.COLOR_BGR2GRAY)
    if len(img_arr_620.shape) == 3:
        img_arr_620 = cv2.cvtColor(img_arr_620, cv2.COLOR_BGR2GRAY)    
    merged = np.zeros(img_arr_690.shape + (3,), dtype = 'float')
    merged[..., 0] = img_arr_690
    merged[..., 2] = img_arr_690
    merged[..., 1] = img_arr_620
    return merged.astype('uint8')    



def auto_min_max(img:np.ndarray, saturation = None):
    if not saturation:
        saturation = 0.1/100
        
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(-1)
    
    max_val = img.max()
    min_val = img.min()
    hist_count, hist_edge = np.histogram(img, bins = np.arange(min_val, max_val+2))
    nbins = len(hist_count)
    assert nbins == max_val - min_val + 1
    count_sum = len(img)
    ind = 0
    if nbins > 2:
        if hist_count[0] > hist_count[1]:
            count_sum -= hist_count[0]
            ind = 1
        if hist_count[-1] > hist_count[-2]:
            count_sum -= hist_count[-1]
            nbins -= 1
            
    count_max = count_sum * saturation
    count = count_max
    min_display = min_val
    while ind < nbins:
        next_count = hist_count[ind]
        if count < next_count:
            min_display = hist_edge[ind] + (count/next_count) * 1
            break
        count -= next_count
        ind += 1
    
    count = count_max
    max_display = max_val
    ind = len(hist_count) - 1
    # ind = nbins - 1
    while ind >=0:
        next_count = hist_count[ind]
        if count < next_count:
            max_display = hist_edge[ind+1] - (count/next_count) * 1
            break
        count -= next_count
        ind -= 1
    return min_display, max_display


def change_brightness(img, min_display, max_display):
    new_img = img.copy()
    new_img = new_img.astype('float')
    new_img = (new_img - min_display) * (255.0/(max_display - min_display))
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype('uint8')


def auto_exposure(img):
    min_display, max_display = auto_min_max(img)
    return change_brightness(img, min_display, max_display)
    
    
# adjust image
def imgadjust_non_linear(image, tol=[0.01, 0.99]):
    # img : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 1.

    def adjust_single_channel(img, tol):
        if img.dtype == 'uint8':
            nbins = 256
        elif img.dtype == 'uint16':
            nbins = 65535

        N = np.histogram(img, bins=nbins, range=[0, nbins])  # get histogram of image
        cdf = np.cumsum(N[0]) / np.sum(N[0])  # calculate cdf of image
        ilow = np.argmax(cdf > tol[0]) / nbins  # get lowest value of cdf (normalized)
        ihigh = np.argmax(cdf >= tol[1]) / nbins  # get heights value of cdf (normalized)

        lut = np.linspace(0, 1, num=nbins)  # create convert map of values
        lut[lut <= ilow] = ilow  # make sure they are larger than lowest value
        lut[lut >= ihigh] = ihigh  # make sure they are smaller than largest value
        lut = (lut - ilow) / (ihigh - ilow)  # normalize between 0 and 1
        lut = np.round(lut * nbins).astype(img.dtype)  # convert to the original image's type
        # print(lut)
        
        img_out = np.array(
            [[lut[i] for i in row] for row in img])  # convert input image values based on conversion list
        return img_out
    if len(image.shape) == 2:
        return adjust_single_channel(image, tol)
    elif len(image.shape) == 3:
        for i in range(3):
            image[:, :, i] = adjust_single_channel(image[:, :, i], tol)
        return image    