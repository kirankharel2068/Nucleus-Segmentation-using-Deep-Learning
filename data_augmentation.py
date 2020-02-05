# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:10:48 2020

@author: KiranKharel
"""
from keras.preprocessing.image import ImageDataGenerator
seed = 1

def perform_augmentation(x_check, y_check, split_size, BATCH_SIZE):
    
    #creating the training image and its respective mask
    data_gen_args = dict(
            shear_range=0.5,
            rotation_range=40,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect'#nearest
            )
    
    #creating training image and mask generator
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    #creating the validation Image and Mask generator
    image_datagen_val = ImageDataGenerator()
    mask_datagen_val = ImageDataGenerator()
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    image_datagen.fit(
            x_check[:int(x_check.shape[0]*split_size)], 
            augment=True,
            seed=seed)

    mask_datagen.fit(
            y_check[:int(x_check.shape[0]*split_size)],
            augment=True,
            seed=seed)
    
    image_datagen_val.fit(
            x_check[int(x_check.shape[0]*0.9):],
            augment=True,
            seed=seed)

    mask_datagen_val.fit(
            y_check[int(y_check.shape[0]*split_size):],
            augment=True,
            seed=seed)
    
    x = image_datagen.flow(
            x_check[:int(x_check.shape[0]*split_size)],
            batch_size = BATCH_SIZE,
            shuffle = True,
            seed = seed
            )

    y = mask_datagen.flow(
            y_check[:int(y_check.shape[0]*split_size)],
            batch_size = BATCH_SIZE,
            shuffle = True,
            seed = seed
            )

    x_val = image_datagen_val.flow(
            x_check[int(x_check.shape[0]*split_size):],
            batch_size = BATCH_SIZE,
            shuffle = True,
            seed = seed
            )
    
    y_val = image_datagen_val.flow(
            y_check[int(y_check.shape[0]*split_size):],
            batch_size = BATCH_SIZE,
            shuffle = True, 
            seed = seed
            )
    
    return x, y, x_val, y_val
