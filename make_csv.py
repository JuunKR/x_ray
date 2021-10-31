import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

resolution = 256

# x_data
path = '../_data/'

img_datagen = ImageDataGenerator(
    rescale=1./255,
)

img = img_datagen.flow_from_directory(
    path + 'image',
    target_size = (resolution, resolution),
    batch_size = 566,
    shuffle=False
)
np.save('../_data/csv/img.npy', arr=img[0][0])

# y_data
mask_datagen = ImageDataGenerator(
    rescale=1./255,
)

mask = mask_datagen.flow_from_directory(
    path + 'new_label',
    target_size = (resolution, resolution),
    batch_size = 566,
    shuffle=False
)
np.save('../_data/csv/mask.npy', arr=mask[0][0])




