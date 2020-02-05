# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:27:15 2020

@author: KiranKharel
"""
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate
from utils import mean_iou

class UNet:
    def __init__(self, image_size = 128, img_channels = 3, kernel_size = (3,3), padding = 'same', strides = 1, kernel_initializer = 'he_normal', activation = 'relu'):
        self.image_size = image_size
        self.img_channels = img_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def down_block(self, x, filters):
        c = Conv2D(filters,
                self.kernel_size, 
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                strides=self.strides, 
                activation=self.activation)(x)
    
        c = Dropout(0.1)(c)
        
        c = Conv2D(filters, 
                self.kernel_size,
                padding=self.padding, 
                kernel_initializer=self.kernel_initializer,
                strides=self.strides,
                activation=self.activation)(c)
    
        p = MaxPooling2D((2,2))(c)
        return c, p

    def bottleneck(self, x, filters):
        c = Conv2D(filters,
                self.kernel_size, 
                padding=self.padding, 
                kernel_initializer= self.kernel_initializer,
                strides=self.strides, 
                activation=self.activation)(x)
    
        c = Dropout(0.1)(c)
    
        c = Conv2D(filters,
               self.kernel_size,
               padding=self.padding, 
               strides=self.strides, 
               kernel_initializer = self.kernel_initializer,
               activation=self.activation)(c)
    
        return c

    def up_block(self, x, skip, filters):
        u = UpSampling2D((2,2))(x)
    
        concat = Concatenate()([u, skip])

        c = Conv2D(filters,
               self.kernel_size, 
               padding=self.padding, 
               strides=self.strides,
               kernel_initializer=self.kernel_initializer,
               activation=self.activation)(concat)
    
        c = Dropout(0.2)(c)
    
        c = Conv2D(filters,
               self.kernel_size,
               padding=self.padding,
               strides=self.strides, 
               kernel_initializer=self.kernel_initializer,
               activation = self.activation)(c)
        return c

    #UNet Model
    def generate_unet(self):
        f = [16, 32, 64, 128, 256]
        inputs = Input((self.image_size, self.image_size, self.img_channels))
        s = Lambda(lambda x: x/255)(inputs)
    
        p0 = s
        #128 -> 64
        c1, p1 = self.down_block(p0, f[0])
        #64--> 32
        c2, p2 = self.down_block(p1, f[1])
        #32 --> 16
        c3, p3 = self.down_block(p2, f[2])
        #16 --> 8
        c4, p4 = self.down_block(p3, f[3])
        
        bn = self.bottleneck(p4, f[4])
    
        #8 --> 16
        u1 = self.up_block(bn, c4, f[3])
        #16 --> 32
        u2 = self.up_block(u1, c3, f[2])
        # 32 --> 64
        u3 = self.up_block(u2, c2, f[1])
        # 64 --> 128
        u4 = self.up_block(u3, c1, f[0])
    
        outputs = Conv2D(1, (1,1),activation='sigmoid')(u4)  
        model = Model(inputs= [inputs], outputs = [outputs])
        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = [mean_iou])
        model.summary()
        return model
