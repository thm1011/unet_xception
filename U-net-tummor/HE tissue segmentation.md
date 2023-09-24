# HE tissue segmentation

### **VOCdevkit/prepare_dataset.ipynb**:

 The dataset is pre-prepared, slicing operations are performed on large slices, as well as dividing the dataset into a training set and a test set, as detailed in the folder train_20230601

JPEGImags_ori:Original histopathological blockbuster

JPEGImages： Large histopathological sections

SegmentationClass_ori： Histopathological blockbuster mask

SegmentationClass：  Histopathological large mask sections

ImageSets： Image naming for training set, test set, validation set

### **train-xception1.ipynb**：

 The overall process of segmenting an organisation:

train_file = 'VOCdevkit/train_20230601/ImageSets/Segmentation/train.txt'
val_file = 'VOCdevkit/train_20230601/ImageSets/Segmentation/val.txt'
img_dir = 'VOCdevkit/train_20230601/JPEGImages'
mask_dir = 'VOCdevkit/train_20230601/SegmentationClass/'

 Just change the path of the corresponding folder.

###  When training the segmentation model, it is necessary to check that the mask is both 0 and 1 pixel values before training on the mask

import matplotlib.pyplot as plt

mask_tmp = Image.open('./train_20230816/SegmentationClass/patch_11807-he_17_4096_5120_1.png')

mask_tmp_arr = np.array(mask_tmp)

plt.imshow(mask_tmp_arr)

plt.imshow(mask_tmp_arr==1)

 mask map (it is possible that the mask map is all black, if the mask map is all black, you need to convert it accordingly)  

**fine-tune**: Fine-tuning is done during training, or if the model breaks during training, continue to restart the model for training.

**loss visualization**： Plot the model's loss curve as well as the F-Score curve.

**logs_512**: The trained model is performed to save under this file.

### h5_to_pb.ipynb：

 The trained .h5 model is converted into a .pb model for deployment.

### **predict -tummor.ipynb：**

file_path = 'test_IHC/'： Image of the file to be tested

result_path = os.path.join('result/', 'test_{:02d}{:02d}'.format(today.month, today.day))： Test file result image

model_path = 'logs_512/20230605/'： Model paths for training

tumor_areas： Predicts the histopathological slice master function and calculates the area of the tissue Area_HE.csv

###  Environment:

tensorflow-GPU   and   keras