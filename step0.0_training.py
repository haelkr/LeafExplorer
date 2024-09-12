# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:44:12 2023

@author: SAQIBQ and HANNAH

Checks a random image from the training dataset and stores its loaded ground truth mask into results
as loaded_mask_of_train_[number] as well as the ground truth mask in color overlayed with the RGB image
as gt_of_train_[number].

Trains the model.

User Guide:
    put the data into the dataset folder seperated into test, train and val folders
    
    close the pop up windows
    define steps per epoch (at least 10, up to 1000) and number of epochs (at least 20, up to 400)
    you can change the number of images per GPU for me worked only one so far
"""
print("loading packages...")
import cv2, random
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import mrcnn
from mrcnn.visualize import display_instances, display_top_masks, save_masks #apply_mask, apply_jmask, 
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from matplotlib import pyplot as plt
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw
import mrcnn.visualize as visualize
import tensorflow as tf
import functions as fc

print("done loading packages")
#########################
#"F:\\Hannah\\logs"
ROOT_DIR = "C:\\Users\\Exjobb\\Downloads\\Mask_RCNN"
sys.path.append(ROOT_DIR)  # To find local version of the library
DIR = "F:\\Hannah"
DEFAULT_LOGS_DIR = os.path.join(DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
CUSTOM_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR , "object20230518T1820", "mask_rcnn_object_0020.h5")
DATA_PATH = os.path.join(DIR, "dataset")
RESULT_DIR = os.path.join(DIR, "train_results")
fc.delete_and_create_directory(RESULT_DIR)
IMAGE_DIR = os.path.join(DIR, "to_predict", "images")



from numba import jit, cuda

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
assert is_gpu, "No GPU available, training takes too much time without"

#########################

class CocoLikeDataset(utils.Dataset):
    def load_data(self, annotation_json, images_dir):
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        print("-", len(coco_json['annotations']),"leaves in this dataset")
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        print("-", image_id+1, "images added")
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids



##############################
def objects_per_class(dataset):
    """Compute number of objects per class
    Input: dataset
    Output: image ids and class counts
    """
    
    # Get list of all class IDs
    class_ids = dataset.class_ids
    print("- Class ids:", class_ids)
    
    # Initialize dictionary to count objects in each class
    class_counts = {}
    for class_id in class_ids:
        class_name = dataset.class_names[class_id]
        class_counts[class_name] = 0
    
    # Iterate over all images and count objects in each class
    image_ids = dataset.image_ids
    for image_id in image_ids:
        _, class_ids = dataset.load_mask(image_id)
        for class_id in class_ids:
            class_name = dataset.class_names[class_id]
            class_counts[class_name] += 1

    return image_ids, class_counts
##############################

print("loading training dataset:")
dataset_train = CocoLikeDataset()
dataset_train.load_data(os.path.join(DATA_PATH, "train","_annotations.coco.json"), os.path.join(DATA_PATH, "train"))
dataset_train.prepare()
train_ids, train_classes = objects_per_class(dataset_train)

print("loading validation dataset:")
dataset_val = CocoLikeDataset()
dataset_val.load_data(os.path.join(DATA_PATH, "val","_annotations.coco.json"), os.path.join(DATA_PATH, "val"))
dataset_val.prepare()
val_ids, val_classes = objects_per_class(dataset_val)

print("loading test dataset:")
dataset_test = CocoLikeDataset()
dataset_test.load_data(os.path.join(DATA_PATH, "test","_annotations.coco.json"), os.path.join(DATA_PATH, "test"))
dataset_test.prepare()
test_ids, test_classes = objects_per_class(dataset_test)


###############################################################################
####################### Test a random image and its annotation
print("""check, if the annotations are loaded correctly. If you want to get rid of the pop up windows comment
      line 196 visualize.display_top_masks(image, mask,... 
      and line 205 visualize.display_instances(image, bbox,... """)
dataset = dataset_train
image_ids = train_ids
class_counts = train_classes
image_id = random.choice(image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
visualize.save_top_masks(image, mask, class_ids, dataset.class_names, limit=2, path = os.path.join(RESULT_DIR, "loaded_mask_of_train{}.jpg".format(image_id)))  #limit to total number of classes
visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=2)  #limit to total number of classes


# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
visualize.save_display_instances(image, bbox, mask, class_ids, dataset_train.class_names,
                                 saving = True, path_to_saved_image = os.path.join(RESULT_DIR, "gt_of_train{}.jpg".format(image_id)),
                                  title="Predictions")
visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names,
                                  title="Predictions")


###############################################################################

steps = int(input("How many steps per epoch?"))
epoch_num = int(input("How many epochs?"))

############ Define a configuration for the model
#change IMAGES_PER_GPU = 1#3
class MarbleConfig(Config):
    # define the name of the configuration
    NAME = "object"
    # number of classes (background + blue marble + non-Blue marble)
    NUM_CLASSES = 1 + 1#changed from 2
    # number of training steps per epoch
    STEPS_PER_EPOCH = steps #10#00
 #   IMAGE_RESIZE_MODE = 'none'
  #  IMAGE_MIN_DIM = 768
   # IMAGE_MAX_DIM = 1024
  #  DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_MIN_CONFIDENCE = 0.8 # Skip detections with < 90% confidence

    IMAGES_PER_GPU = 1 #1 worked   
    IMAGE_MIN_DIM = IMAGE_MAX_DIM = 1024
    #TF_GPU_ALLOCATOR=cuda_malloc_async#added here to overcome oom
# prepare config
config = MarbleConfig()
config.display() 




#changed here have the directories been


########################
#Weights are saved to root D: directory. need to investigate how they can be
#saved to the directory defined... "logs_models"

###############

################# Calculate Class Weight according to Objects presents in the image

total_samples = sum(class_counts.values())
class_weights = {}
for class_id, count in class_counts.items():
    if count == 0:
        class_weights[class_id] = 0
    else:
        class_weights[class_id] = total_samples / (len(class_counts) * count)
# Compute class weights as the inverse of the frequency of samples in each class

#########################################################3


####### Define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epoch_num, layers='heads')
#changed epochs from 400 to 20

#changed: here was the prediction until end part                      
