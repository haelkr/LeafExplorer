# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:44:12 2023

@author: SAQIBQ and HANNAH



The model is loaded in inference mode and all three datsets training, validation and testing evaluated.
This includes 
    saving the detected masks colored in the results/display_instances folder,
    calculating the AP, precision, recall and overlap of every image for IoU50 and 75,
        this is averaged over all images
    plotting precision recall curve for whole dataset, stored in the results folder,
    calculating (total) ground truth and prediction vectors,
    saving confusion matrix for every dataset into results folder,
    calculating and saving mAP, mAP75, mAR and F1 score, F1 score 75 as well as tp,fp and fn in results.csv.
    
    
Guide: set the ROOT_DIR (necessary?)
    set DIR 
    set WEIGHTS PATH -> here I can make things automatically
    set DATA_PATH
    set RESULTS_DIR
    
change between detection confidence 0.5 and 0.75 in confidence_value and take the corresponding prec rec curves and ap values
224: DETECTION_MIN_CONFIDENCE = 0.75 or 0.5
    
    
For me: change log, change dataset, make sure it does not overwrite results

How to proceed: all leaves are fp + fn, make it percentagewise as well


 
"""

confidence_value = 0.75
print("loading packages...")
import cv2, csv
from tqdm import tqdm
import os#, random
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
import pandas as pd
import functions as fc
import utils_for_confusion as ufc
import skimage
#########################


ROOT_DIR = "C:\\Users\\Exjobb\\Downloads\\Mask_RCNN"
sys.path.append(ROOT_DIR)  # To find local version of the library
DIR = "F:\\Hannah"
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DATA_PATH = os.path.join(DIR, "dataset")
DEFAULT_LOGS_DIR = os.path.join(DIR, "logs")
CUSTOM_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR , "shear", "mask_rcnn_object_0100.h5")
#COCO_WEIGHTS_PATH = "mask_rcnn_object_0129_april.h5"
RESULT_DIR = os.path.join(DIR, "eval_results")


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

####################   COMPUTE Number of Class Object
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





################  PREDICTION  ####################
###################################################
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap, compute_recall
from numpy import mean
#from matplotlib.patches import Rectangle

def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax



# define the prediction configuration
class PredictionConfig(Config):
    NAME = "object"
    # number of classes (background + Blue Marbles + Non Blue marbles)
    NUM_CLASSES = 1 + 1#changed 2
    # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    DETECTION_MIN_CONFIDENCE = confidence_value
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
if not os.path.exists(RESULT_DIR): 
    os.makedirs(RESULT_DIR)
fc.delete_and_create_directory(os.path.join(RESULT_DIR, "display_instances"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR, "color_splash"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR, "display_instances","training" ))
fc.delete_and_create_directory(os.path.join(RESULT_DIR, "display_instances","validation"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR, "display_instances", "test"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR,"color_splash", "test"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR,"color_splash", "validation"))
fc.delete_and_create_directory(os.path.join(RESULT_DIR,"color_splash", "training"))


class_names = ['BG', 'leaves']
path = os.path.join(RESULT_DIR, "results.csv")
#delete file if it exists and create new one
f1 = open( path, 'w', newline='')
with open(path, 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["dataset", "mAP", "mAR", "F1 score", "mAP75", "F1 score 75", "tp", "fp", "fn"])
 
    # calculate the mAP for a model on a given dataset
    def evaluate_model(dataset, model, cfg, dset):
        APs = list()
        ARs = list()
        Precison = list()
        Recall = list()
        
        AP75s = list()
        F1_scores75 = list()
        Precison75 = list()
        Recall75 = list()
        
# =============================================================================
#         Overlaps = list()
#         Class_id = list()
#         Scores = list()
# =============================================================================
        F1_scores = list()
        plt.close('all')
        
        
        #-------------------------FROM FILE evaluation_I_did_earlier:-----------------------
        #from https://github.com/Altimis/Confusion-matrix-for-Mask-R-CNN/blob/master/Confusion%20matrix%20for%20Mask%20R-CNN.ipynb
        #compute Ap, precision, recall for whole dataset     
        total_gt = np.array([]) 
        total_pred = np.array([]) 
    
        #RUN DETECTION
        #compute total_gt, total_pred and mAP for each image in the test dataset
        # Compute total ground truth boxes(total_gt) and total predicted boxes(total_pred) and mean average precision for each Image 
        #in the test dataset
    
        for image_id in tqdm(dataset.image_ids):
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)#, use_mini_mask=False) #evtl use_mini_mask kommentieren
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            #ax = plt.gca()
            visualize.save_display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, saving = True,
                                             path_to_saved_image = os.path.join(RESULT_DIR, "display_instances",dset,"{}.jpg".format(image_id)),
                                             scores = r['scores'], title="Predictions")
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,scores = r['scores'], ax=ax, title="Predictions")
            splash = visualize.color_splash(image, r['masks'])
            # Save output
            file_name = os.path.join(RESULT_DIR,"color_splash",dset, "{}.jpg".format(image_id))#"\\splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            skimage.io.imsave(file_name, splash)

            
            #compute gt_tot and pred_tot
            gt, pred = ufc.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
            total_gt = np.append(total_gt, gt)
            total_pred = np.append(total_pred, pred)
            #check if the vectors len are equal
            assert len(total_gt) == len(total_pred), "length of ground truth vec and predicted vec have to be equal!"
            
            
            # calculate statistics, including AP
            AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            AP75, precisions75, recalls75, overlaps75 = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.75)
            # store
            APs.append(AP)
            AP75s.append(AP75)
            AR, positive_ids = compute_recall(r["rois"], gt_bbox, iou=0.2)
            ARs.append(AR)
            F1_scores.append((2* (mean(precisions) * mean(recalls)))/(mean(precisions) + mean(recalls)))
            F1_scores75.append((2* (mean(precisions75) * mean(recalls75)))/(mean(precisions75) + mean(recalls75)))
            #visualize.plot_precision_recall(AP, precisions, recalls)
            
            for precision in precisions:
                Precison.append(precision)
            for recall in recalls:
                Recall.append(recall)
            for precision in precisions75:
                Precison75.append(precision)
            for recall in recalls75:
                Recall75.append(recall)
    #     for overlap in overlaps:
    #         Overlaps.append(overlap)
    #     for class_id in r['class_ids']:
    #         Class_id.append(class_id)
    #     for score in r['scores']:
    #         Scores.append(score)
              
        # calculate the mean AP across all images
        mAP = mean(APs)
        mAP75 = mean(AP75s)
        mAR = mean(ARs)
        F1_score_2 = (2 * mAP * mAR)/(mAP + mAR)
        F1_score75_2 = (2 * mAP75 * mAR)/(mAP75 + mAR)
        
        #sort Recall ascending, keep indices and sort Precision in the same way
        np_Recall = np.array(Recall)
        np_Precison = np.array(Precison)
        
        sorted_indices = np_Recall.argsort() #array([2, 1, 4, 3, 0, 7, 5, 6])
        
        Recall = np_Recall[sorted_indices] #array([-1. ,  0. ,  0. ,  0.1,  1. ,  4. ,  5. , 10. ])
        Precison = np_Precison[sorted_indices]
        
        #calculate the average of all y values connected to the same x value
        df = pd.DataFrame({'x': Recall, 'y': Precison})
        df_mean = df.groupby('x').mean().reset_index()
        
        # Precision Recall curve
        visualize.plot_precision_recall(mAP, df_mean['y'], df_mean['x'], dataset = dset, 
                                        path_to_saved_image=os.path.join(RESULT_DIR, dset+ "prec_rec.png"))
    
        np_Recall75 = np.array(Recall75)
        np_Precison75 = np.array(Precison75)
        
        sorted_indices75 = np_Recall75.argsort() #array([2, 1, 4, 3, 0, 7, 5, 6])
        
        Recall75 = np_Recall75[sorted_indices75] #array([-1. ,  0. ,  0. ,  0.1,  1. ,  4. ,  5. , 10. ])
        Precison75 = np_Precison75[sorted_indices75]
        
        #calculate the average of all y values connected to the same x value
        df = pd.DataFrame({'x': Recall75, 'y': Precison75})
        df_mean = df.groupby('x').mean().reset_index()
        
        # Precision Recall curve
        visualize.plot_precision_recall(mAP75, df_mean['y'], df_mean['x'], dataset = dset, 
                                        path_to_saved_image=os.path.join(RESULT_DIR, dset+ "prec_rec75.png"),iou=75)
    
    
    
        total_gt=total_gt.astype(int)
        total_pred=total_pred.astype(int)
        
     
        # # Confusion Matrix
        tp,fp,fn=ufc.plot_confusion_matrix_from_data(total_gt,total_pred,fz=18, figsize=(20,20), lw=0.5, 
                                                     path_to_saved_image = os.path.join(RESULT_DIR,"confusion_matrix_{}.jpg".format(dset)))#columns= ["BG", "leaf", "sum"],
     
        #The vertical axis represent the ground-truth classes and the horizontal axis represent the predicted classes.
        #BG class is the background class.
        # It is not taken into account in the calculation of the mAP -> use [1].     
        filewriter.writerow([dset, mAP, mAR, F1_score_2,mAP75, F1_score75_2, tp[1], fp[1], fn[1]])
        
        
        
        return mAP, mAR, F1_score_2, tp[1], fp[1], fn[1],mAP75, F1_score75_2
    
        
    
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir=DEFAULT_LOGS_DIR, config=cfg)
    # load model weights
    model.load_weights(CUSTOM_WEIGHTS_PATH, by_name=True)
    #model.load_weights('mask_rcnn_object_0516_best.h5', by_name=True)
    
    
    ####### evaluate model on training dataset
    for dataset in [[dataset_train, "training"],[dataset_val, "validation"], [dataset_test, "test"]]:
        mAP, mAR, F1_score, tp, fp, fn, mAP75, F1_score75 = evaluate_model(dataset[0], model, cfg, dset=dataset[1])
        print(dataset[1], " mAP50: %.3f" % mAP)
        print(dataset[1], " mAP75: %.3f" % mAP75)
        print(dataset[1], " mAR: %.3f" % mAR)
        print("tp for class leaf:",tp)
        print("fp for class leaf:",fp)
        print("fn for class leaf:",fn)
        print(dataset[1], " F1-score: %.3f" % F1_score)
        print(dataset[1], " F1-score75: %.3f" % F1_score75)
    
    

                    
