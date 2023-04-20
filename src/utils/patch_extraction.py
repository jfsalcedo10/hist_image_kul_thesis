import sys
#Adding ASAP to the environment
ASAP_PATH = 'D:\\Program Files (x86)\\ASAP 2.1\\bin'
if ASAP_PATH not in sys.path:
    sys.path.append(ASAP_PATH)
print(sys.path)

from PIL import Image
from PIL import TiffImagePlugin
import numpy as np
import cv2
import openslide
import multiresolutionimageinterface as mir
import os
import random

NORMAL_PATH = '/content/drive/MyDrive/Thesis/Normal/'
TUMOR_PATH = '/content/drive/MyDrive/Thesis/Tumor/'
TUMOR_ANN_PATH = '/content/drive/MyDrive/Thesis/Tumor_Annotations/'
TUMOR_MASK_PATH = '/content/drive/MyDrive/Thesis/Tumor_Masks/'
PATCH_POS_PATH = '/content/drive/MyDrive/Thesis/Patches_Positive/'
PATCH_NEG_PATH = '/content/drive/MyDrive/Thesis/Patches_Negative/'

MAG_FACTOR = 256          # magnification factor 
PATCHES_PER_BBOX = 20     # number of samples per bounding box
ADAPTIVE_QUANT = 0        # adapt number of samples based on box size
PATCH_SIZE = 256          # size of sampled patches
THRESH = 0.2              # % of patch that should be tumor (for tumor patches only)

tumor_wsis = os.listdir(TUMOR_PATH)
normal_wsis = os.listdir(NORMAL_PATH)

# Mask annotations
def ann_to_mask(wsi):
  mr_image = reader.open(TUMOR_PATH + wsi)
  ann_path = wsi.split('.', 1)[0] + '.xml'

  # load annotation
  xml_repository.setSource(TUMOR_ANN_PATH + ann_path)
  xml_repository.load()

  # save mask file (warning: takes long)
  mask_path = wsi.replace('tumor', 'mask')
  annotation_mask.convert(annotation_list, TUMOR_MASK_PATH + mask_path, 
                          mr_image.getDimensions(), mr_image.getSpacing(), 
                          label_map, conversion_order)
  
if __name__ == '__main__':
  reader = mir.MultiResolutionImageReader()

  annotation_list = mir.AnnotationList()
  xml_repository = mir.XmlRepository(annotation_list)
  annotation_mask = mir.AnnotationToMask()

  # adjust labels based on dataset (camelyon16 vs 17)
  camelyon17_type_mask = False
  label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
  conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']

  # get list of all tumor WSIs and convert to masks
  tumor_wsis = os.listdir(TUMOR_PATH)
  for wsi in tumor_wsis:
    ann_to_mask(wsi)

# Patch extraction
def read_wsi(wsi_path, mag_factor):
  wsi_full_size = openslide.OpenSlide(wsi_path)
  mag_options = wsi_full_size.level_downsamples
  mag_level = mag_options.index(mag_factor)
  wsi_scaled = np.array(wsi_full_size.read_region((0, 0), mag_level,
                                        wsi_full_size.level_dimensions[mag_level]))
  return wsi_full_size, wsi_scaled

def get_tissue_contours(wsi_scaled):
  hsv_img = cv2.cvtColor(wsi_scaled, cv2.COLOR_BGR2HSV)
  lower_red = np.array([20, 20, 20])
  upper_red = np.array([200, 200, 200])
  tissue_mask = cv2.inRange(hsv_img, lower_red, upper_red)

  close_kernel = np.ones((20, 20), dtype=np.uint8)
  image_close = cv2.morphologyEx(np.array(tissue_mask), cv2.MORPH_CLOSE, 
                                 close_kernel)
  open_kernel = np.ones((5, 5), dtype=np.uint8)
  image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, 
                                open_kernel)
  
  tissue_contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)

  return tissue_contours

def get_tumor_contours(mask_img):
  bw_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
  tumor_contours, _ = cv2.findContours(np.array(bw_mask), cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
  
  return tumor_contours

def get_bbox(contours):
  bounding_boxes = [cv2.boundingRect(c) for c in contours]
  bounding_boxes_big = [i for i in bounding_boxes if i[2] > 10 or i[3] > 10]
  return bounding_boxes_big
