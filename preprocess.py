import cv2
import math
from glob import glob
import os
import PIL
from keras.utils.image_utils import load_img,img_to_array
import numpy as np

def preprocess_vdo(file_path,video_upload_path):
    
    cap = cv2.VideoCapture(file_path)
    frameRate = cap.get(5) #frame rate
    x=1
    count=0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames of this particular video in temp folder
            fn = "_frame%d.jpg" % count;count+=1
            filename =os.path.join(video_upload_path, fn)
            cv2.imwrite(filename, frame)
    cap.release()
    
    # reading all the frames from temp folder
    images = glob(video_upload_path+"/*.jpg")

    prediction_images = []
    for i in range(len(images)):
        img = load_img(images[i], target_size=(224,224,3))
        img = img_to_array(img)
        img = img/255
        prediction_images.append(img)
    prediction_images = np.array(prediction_images)
    return prediction_images