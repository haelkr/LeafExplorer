#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#conda activate analysis

"""
Guide: set root dir, put the data into the three folders RGB, thermal and csv
"""

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import random, cv2, math, glob
import os
from tqdm import tqdm
from skimage.feature import graycomatrix as greycomatrix
from plantcv import plantcv as pcv

from skimage import img_as_ubyte
from plantcv.plantcv import fatal_error
from plantcv.plantcv._debug import _debug
from plantcv.plantcv import params
from plantcv.plantcv.transform import rescale

import functions_imagewise as fc


#
ROOT_DIR ="F:\\from_Disk"# "C:\\Users\\Exjobb\\Documents\\Preprocessing"

#Data path
RGB_DIR = os.path.join(ROOT_DIR,"original_data", "RGB")
THERMAL_DIR = os.path.join(ROOT_DIR,"original_data", "thermal")
CSV_DIR = os.path.join(ROOT_DIR, "original_data","csv")

MASK_DIR = os.path.join(ROOT_DIR, "...")    #!!!!!!!!!!!!!!!!!!!

#final paths
fc.delete_and_create_directory(os.path.join(ROOT_DIR,"preprocessed_data"))
FINAL_RGB_DIR = os.path.join(ROOT_DIR,"preprocessed_data","RGB")
FINAL_THERMAL_DIR = os.path.join(ROOT_DIR,"preprocessed_data","thermal")
FINAL_CSV_DIR = os.path.join(ROOT_DIR,"preprocessed_data","csv")

fc.delete_and_create_directory(FINAL_RGB_DIR)
fc.delete_and_create_directory(FINAL_THERMAL_DIR)
fc.delete_and_create_directory(FINAL_CSV_DIR)





# get them in to the right order to not mix up labeled and real pics
rgb_names = sorted(os.listdir(RGB_DIR), key=lambda x: int("".join([i for i in x if i.isdigit()]))) 
print(rgb_names)
IR_names = sorted(os.listdir(THERMAL_DIR), key=lambda x: int("".join([i for i in x if i.isdigit()]))) 
print(IR_names)
csv_names = sorted([v for v in os.listdir(CSV_DIR) if "csv" in v], key=lambda x: int("".join([i for i in x if i.isdigit()]))) 
print(csv_names)

assert len(rgb_names) == len(IR_names) == len(csv_names), "Not the same number of RGB, thermal and csv files!"
#converting the images
for rgb_name, Ir_name, csv_name, j in tqdm(zip(rgb_names,IR_names, csv_names, range(len(rgb_names)) ), desc='preprocessing images',total =len(rgb_names)):
#for images in nameList:


    #------------------------------Data Loading-----------------------------------------------
    rgb = fc.data_loader(os.path.join(RGB_DIR, rgb_name))
    ir = fc.data_loader(os.path.join(THERMAL_DIR, Ir_name))
    ir = cv2.rotate(ir, cv2.ROTATE_90_COUNTERCLOCKWISE)
    csv = fc.csv_loader(os.path.join(CSV_DIR, csv_name))
    csv = cv2.rotate(csv, cv2.ROTATE_90_COUNTERCLOCKWISE)


    
    #--------------------------Cropping rgb---------------------------------------
    rgb_new, data = fc.cropping(rgb)
    rgb = rgb_new



    #-----------------------------Circle Drawing---------------------------------------------
     
    #rgbs
    R_rgb = 569 
    X_rgb,Y_rgb = fc.circlecropper(data)
    
    #thermal images
    X_ir,Y_ir,R_ir = 228,311,121
 
  
  
    #--------------------------------Thermal Cutting----------------------------------------
    
    ir_c = [X_ir,Y_ir,R_ir]
    rgb_c = [X_rgb,Y_rgb,R_rgb]
    cutting = data

    bottom_edge = fc.scaling(ir_c[1], ir_c[2]/rgb_c[2], cutting[1]-cutting[0]-rgb_c[1])
    top_edge = fc.scaling(ir_c[1], ir_c[2]/rgb_c[2], (-1)*rgb_c[1])
    left_edge = fc.scaling(ir_c[0], ir_c[2]/rgb_c[2], (-1)*rgb_c[0])
    right_edge = fc.scaling(ir_c[0], ir_c[2]/rgb_c[2], cutting[3]-cutting[2]-rgb_c[0])
    
    if top_edge <0:
      rgb_top = math.floor(rgb_c[2]/ir_c[2]*abs(top_edge))
      rgb = rgb[rgb_top:,:,:]
      top_edge = 0
    if left_edge <0:
      rgb_left = math.floor(rgb_c[2]/ir_c[2]*abs(left_edge))
      rgb = rgb[:,rgb_left:,:]
      left_edge = 0

      
    cropped_ir = ir[top_edge: bottom_edge , left_edge : right_edge,:] 
    cropped_csv = csv[top_edge: bottom_edge , left_edge : right_edge] 
  
  
    #----------------------------------Saving-------------------------------------------------------
  
    #ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0])) 


    Image.fromarray(rgb).save(os.path.join(FINAL_RGB_DIR,rgb_name))
    Image.fromarray(cropped_ir).save(os.path.join(FINAL_THERMAL_DIR,Ir_name))
    np.savetxt(os.path.join(FINAL_CSV_DIR,csv_name),cropped_csv, delimiter=',', fmt='%s',encoding='utf-8')
  

#-------------------------------------Test-----------------------------------------------
csv = np.genfromtxt (os.path.join(FINAL_CSV_DIR,csv_names[0]), delimiter=",")
fc.one_plot(csv)
#how to extract thermal information later. 
  