import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob


def erosion(img):
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(dilate, kernel, iterations=1)

    return erosion


path = '../_data/'
masks_path = path + 'label/'

imgs = glob(masks_path + '1/*.png')

for img in imgs:
    fName = img.split('\\')[-1]
    original_img = cv2.imread(img)

    img90 = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
    new_img = np.zeros_like(img90)

    gray = cv2.cvtColor(img90, cv2.COLOR_BGR2GRAY)
    erosion_img = erosion(gray)
    
    _, dst = cv2.threshold(erosion_img, 127, 255, cv2.THRESH_BINARY) # dst = destination? _ -> retval
        
    retval, labels = cv2.connectedComponents(dst, connectivity=8)
    for i in range(retval):
        # print(i)
        if i == 0:
            new_img[labels==i] = [0,0,0]
        elif i == 1:
            new_img[labels==i] = [255,0,0]
        elif i == 2: 
            new_img[labels==i] = [0,0,255]

    new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('_data/new_label/1/'+ fName, new_img)

    # cv2.imshow('mask', new_img)
    # cv2.waitKey(0)






