import os
import sys
import json


# Getting the openslide tools for windows
# OPENSLIDE_PATH = r'D:\kuleuven\thesis\openslide-win64\bin'
OPENSLIDE_PATH = os.path.join(
    'D:', os.sep, 'kuleuven', 'thesis', 'openslide-win64', 'bin')
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide  # type: ignore
        print(f'OpenSlide version: {openslide.__version__}')
        print(f'OpenSlide library version: {openslide.__library_version__}')
else:
    import openslide  # type: ignore

from pprint import pprint

IMAGE_PATH = os.path.join('..', 'data', 'training', 'images')
ANNOTATION_PATH = os.path.join('..', 'data', 'training', 'annotations')
MASK_PATH = os.path.join('..', 'data', 'training', 'masks')
LABELS_PATH = os.path.join('..', 'data', 'tumor_images.json') # path with JSON telling if image is tumor or not

# Patch attributes
MAG_FACTOR = 256          # magnification factor
PATCHES_PER_BBOX = 20     # number of samples per bounding box
ADAPTIVE_QUANT = 0        # adapt number of samples based on box size
PATCH_SIZE = 256          # size of sampled patches
# % of patch that should be tumor (for tumor patches only)
TUMOR_THRESHOLD = 0.2

images_to_patch = [image for image in os.listdir(IMAGE_PATH) if os.path.isfile(os.path.join(IMAGE_PATH,image))]
with open(LABELS_PATH,'r') as f:
    images_labels = json.load(f)


mask_path_test = os.path.join(MASK_PATH, images_to_patch[1].replace('.tif','_mask.tif'))
slide = openslide.OpenSlide(mask_path_test)
print(openslide.OpenSlide.detect_format(mask_path_test))
region = (0, 0)
level = slide.get_best_level_for_downsample(MAG_FACTOR)
size = slide.level_dimensions[level]
print(size)


## region = slide.read_region(region, level, size).convert('L')
region = slide.get_thumbnail(size)