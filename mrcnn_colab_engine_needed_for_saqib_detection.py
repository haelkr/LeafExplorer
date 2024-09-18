import os
import sys
# Import Mask RCNN
import mrcnn
import numpy as np
from skimage.measure import find_contours


# Root directory of the project
ROOT_DIR = os.path.abspath("./") #/content/calculate_object_area_exercise
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.model as modellib
import colorsys
import random
import cv2


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+1
    NAME = 'coco'

def get_mask_contours(mask):
    #mask = masks[:, :, i]
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    contours_mask = []
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        contours_mask.append(np.array(verts, np.int32))
    return contours_mask

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 255 if bright else 180
    hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def draw_mask(img, pts, color, alpha=0.5):
    h, w, _ = img.shape

    overlay = img.copy()
    output = img.copy()

    cv2.fillPoly(overlay, pts, color)
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    return output
 
def draw_mask1(img, pts, color, alpha=0.5):
    h, w, _ = img.shape

    overlay = img.copy()
    output = img.copy()

    cv2.fillPoly(overlay, pts, color)
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    return output

def detect_contours_maskrcnn(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.detect([img_rgb])
    r = results[0]
    object_count = len(r["class_ids"])
    
    objects_ids = []
    objects_contours = []
    bboxes = []
    for i in range(object_count):
        # 1. Class ID
        class_id = r["class_ids"][i]
        # 2. Boxes
        box = r["rois"][i]
        
        # 3. Mask
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)
        bboxes.append(box)
        objects_contours.append(contours[0])
        objects_ids.append(class_id)
    return objects_ids, bboxes, objects_contours
    
def detect_contours_maskrcnn_with_original(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.detect([img_rgb])
    r = results[0]
    object_count = len(r["class_ids"])
    
    objects_ids = []
    objects_contours = []
    bboxes = []
    for i in range(object_count):
        # 1. Class ID
        class_id = r["class_ids"][i]
        # 2. Boxes
        box = r["rois"][i]
        
        # 3. Mask
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)
        bboxes.append(box)
        objects_contours.append(contours[0])
        objects_ids.append(class_id)
    return objects_ids, bboxes, objects_contours,r
