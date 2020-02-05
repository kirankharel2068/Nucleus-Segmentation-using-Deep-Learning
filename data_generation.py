# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:52:14 2020

@author: KiranKharel
"""
#importing libraries
import os
import sys
import numpy as np
import warnings
import cv2
from tqdm import tqdm
warnings.filterwarnings('ignore')
#seeding
seed = 42

class DataGen:
    def __init__(self, ids, datasettype, path, image_size = 128, img_channels = 3):
        self.ids = ids
        self.path = path
        self.image_size = image_size
        self.datasettype = datasettype
        self.img_channels = img_channels
        
    def __load__(self, id_name):
        ##path 
        image_path = os.path.join(self.path, id_name, "images",id_name)+".png"
        mask_path = os.path.join(self.path, id_name, "masks/")

        #reading image
        image = cv2.imread(image_path,1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        #reading mask
        mask = np.zeros((self.image_size, self.image_size, 1), dtype = np.bool)
        all_mask = os.listdir(mask_path)
        
        for mask_file in all_mask:
            _mask_path = mask_path+mask_file
            mask_image = cv2.imread(_mask_path, -1)
            mask_image = cv2.resize(mask_image, (self.image_size, self.image_size))
            mask_image = np.expand_dims(mask_image, axis=-1)

            mask = np.maximum(mask, mask_image)
                
        return image, mask
            

    def __getitems__(self):
        X_train = np.zeros((len(self.ids), self.image_size, self.image_size, self.img_channels), dtype = np.uint8)
        Y_train = np.zeros((len(self.ids), self.image_size, self.image_size,1),dtype = np.bool)
        
        print('Getting and Resizing train images and masks.....')
        sys.stdout.flush()

        for n, id_ in tqdm(enumerate(self.ids), total = len(self.ids)):
            img, mask = self.__load__(id_)
    
            X_train[n] = img
            Y_train[n] = mask
        
        return X_train, Y_train
