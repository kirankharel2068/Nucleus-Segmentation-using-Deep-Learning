# Nucleus-Segmentation-using-Deep-Learning

This project consist implementation of CNN for automatic segmentation of Nucleus. The Fully convolutional network (U-Net Arcitecture) is used to perform semantic segmentation of the Nucleus.

List of Files
--------------
* Main.ipynb: Here methods from all the modules are called to perform the training and testing of the image datasets.
* UNet_Model.py: Class which implements UNet Architecture to perform the segentation task.
* data_augmentation.py: Performs data augmentaion to create more samples of datasets through various transformation to train the model better.
* data_generation.py: perform data ingestion and makes the image datasets and their respective masks ready to be passed to the Deep learning Model.
* Utils: Consitst of Intersection over Union (IoU) metric also called Jaccard Index for evaluation of model.


