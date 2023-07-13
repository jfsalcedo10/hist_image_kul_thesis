from typing import Optional, Union
from PIL import Image
from PIL import TiffImagePlugin
import numpy as np
import cv2
# Comes from ASAP library (PATH HAS TO BE DEFINED in the script the class is defined.)
import multiresolutionimageinterface as mir  # type: ignore
import os
import random

# Getting the openslide tools for windows
# OPENSLIDE_PATH = r'D:\kuleuven\thesis\openslide-win64\bin'
OPENSLIDE_PATH = os.path.join(
    'D:', os.sep, 'kuleuven', 'thesis', 'openslide-win64-20171122', 'bin')
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide  # type: ignore
        # print(f'OpenSlide version: {openslide.__version__}')
        print(f'OpenSlide library version: {openslide.__library_version__}')
else:
    import openslide  # type: ignore


class PatchGenerator:

    def __init__(self,
                 image_path: str,
                 annotation_path: str,
                 mask_path: str,
                 mag_factor: int,
                 patches_per_bbox: int,
                 patch_size: int,
                 tumor_threshold: float,
                 adaptive_quant: int,
                 lower_bound,
                 upper_bound):

        self.image_path = image_path
        self.annotation_path = annotation_path
        self.mask_path = mask_path
        self.patch_pos_path = os.path.join(image_path, 'patches', 'tumor')
        self.patch_neg_path = os.path.join(image_path, 'patches', 'normal')

        # Making patching folders if they don't exist
        try:
            os.makedirs(self.patch_pos_path)
            os.makedirs(self.patch_neg_path)
        except OSError as e:
            if not os.path.isdir(self.patch_pos_path) or not os.path.isdir(self.patch_neg_path):
                raise e

        # Patch atrributes
        self.mag_factor = mag_factor                # magnification factor
        self.patches_per_bbox = patches_per_bbox    # number of samples per bounding box
        # adapt number of samples based on box size
        self.adaptive_quant = adaptive_quant
        self.patch_size = patch_size                # size of sampled patches
        # % of patch that should be tumor (for tumor patches only)
        self.tumor_threshold = tumor_threshold
        self.lower_bound = lower_bound              # RGB min threshold for colors
        self.upper_bound = upper_bound              # RGB max threshold for colors

    def read_wsi(self, wsi_path):
        wsi_full_size = openslide.OpenSlide(wsi_path)

        mag_level = wsi_full_size.get_best_level_for_downsample(
            self.mag_factor)
        print(mag_level)
        # Rescaling image
        region = (0, 0)
        size = wsi_full_size.level_dimensions[mag_level]
        try:
            wsi_scaled = np.array(
                wsi_full_size.read_region(region, mag_level, size))
        except openslide.OpenSlideError as e:
            print(f'Openslide extraction failed. Trying with ASAP...')
            reader = mir.MultiResolutionImageReader()
            slide = reader.open(wsi_path)
            wsi_scaled = np.array(slide.getUCharPatch(
                startX=region[0], startY=region[1], width=size[0], height=size[1], level=mag_level))
            wsi_full_size = slide
        print(wsi_scaled.shape)

        return wsi_full_size, wsi_scaled

    def get_tissue_contours(self, wsi_scaled):
        hsv_img = cv2.cvtColor(wsi_scaled, cv2.COLOR_BGR2HSV)
        tissue_img = self.extract_tissue(hsv_img)
        tissue_contours, _ = cv2.findContours(tissue_img, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        return tissue_contours

    def extract_tissue(self, hsv_img):
        tissue_mask = cv2.inRange(hsv_img, self.lower_bound, self.upper_bound)

        kernel_close = np.ones((20, 20), dtype=np.uint8)
        kernel_open = np.ones((5, 5), dtype=np.uint8)

        image_closed = cv2.morphologyEx(np.array(tissue_mask), cv2.MORPH_CLOSE,
                                        kernel_close)
        image_open = cv2.morphologyEx(np.array(image_closed), cv2.MORPH_OPEN,
                                      kernel_open)

        return image_open

    def get_tumor_contours(self, mask_img):
        bw_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        tumor_contours, _ = cv2.findContours(np.array(bw_mask), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

        return tumor_contours

    def get_bbox(self, contours):
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes_big = [
            i for i in bounding_boxes if i[2] > 10 and i[3] > 10]
        return bounding_boxes_big

    def get_tumor_patches_const(self, wsi_full_size, mask_full_size, img_idx,
                                tumor_contours):

        patch_idx = 0

        for bbox in self.get_bbox(tumor_contours):
            # tumor bounding box
            x, y, w, h = bbox
            patch_in_box = 0

            # sample until desired number of patches is reached (constant)
            while (patch_in_box < self.patches_per_bbox):
                # start from a random point within downsampled bounding box
                rand_x = random.randint(x, x+w)
                rand_y = random.randint(y, y+h)

                # calculate point same point for full size WSI
                real_x = rand_x * self.mag_factor
                real_y = rand_y * self.mag_factor

                # extract mask for selected patch
                mask_patch = mask_full_size.read_region((real_x, real_y), 0,
                                                        (self.patch_size, self.patch_size))
                mask_patch_np = np.array(mask_patch)[:, :, 0]

                # check what proportion of patch is tumor
                tumor_percent = (mask_patch_np.sum(axis=0).sum(
                    axis=0)/255)/(self.patch_size*self.patch_size)
                mask_patch.close()

                # save patch only if threshold for tumor proportion is met
                if (tumor_percent > self.tumor_threshold):
                    wsi_patch = wsi_full_size.read_region((real_x, real_y), 0,
                                                          (self.patch_size, self.patch_size))
                    wsi_patch.save(self.patch_pos_path +
                                   img_idx + '_' + str(patch_idx), 'PNG')
                    patch_idx += 1
                    patch_in_box += 1
                    wsi_patch.close()

        return 0

    def get_normalT_patches_const(self, wsi_full_size, mask_full_size, img_idx,
                                  tissue_contours):

        patch_idx = 0

        for bbox in self.get_bbox(tissue_contours):
            # tissue bounidng box
            x, y, w, h = bbox
            patch_in_box = 0

            # sample until desired number of patches is reached (constant)
            while (patch_in_box < self.patches_per_bbox):
                rand_x = random.randint(x, x+w)
                rand_y = random.randint(y, y+h)

                real_x = rand_x * self.mag_factor
                real_y = rand_y * self.mag_factor

                # extract mask for selected patch
                mask_patch = mask_full_size.read_region((real_x, real_y), 0,
                                                        (self.patch_size, self.patch_size))
                mask_patch_np = np.array(mask_patch)[:, :, 0]
                mask_patch.close()

                # only continue if patch has no cancerous tissue
                if (mask_patch_np.sum(axis=0).sum(axis=0) == 0):
                    # extract selected patch
                    wsi_patch = wsi_full_size.read_region((real_x, real_y), 0,
                                                          (self.patch_size, self.patch_size))
                    hsv_patch = cv2.cvtColor(
                        np.array(wsi_patch), cv2.COLOR_BGR2HSV)

                    tissue_patch = self.extract_tissue(hsv_patch)

                    # check what proportion of patch is tissue
                    tissue_percent = (tissue_patch.sum(axis=0).sum(
                        axis=0)/255)/(self.patch_size*self.patch_size)

                    # save patch only if threshold for tissue proportion is met
                    if (tissue_percent > self.tumor_threshold):
                        wsi_patch.save(self.patch_neg_path +
                                       img_idx + '_T' + str(patch_idx), 'PNG')
                        patch_idx += 1
                        patch_in_box += 1
                        wsi_patch.close()
                        print('Saving')

        return 0

    def get_patches(self, wsi_full_size: Union[openslide.OpenSlide, mir.MultiResolutionImage], is_tumor, img_idx, tissue_contours, mask_full_size: Optional[Union[openslide.OpenSlide, mir.MultiResolutionImage]] = None):
        patch_idx = 0

        for bbox in self.get_bbox(tissue_contours):
            x, y, w, h = bbox
            patch_in_box = 0

            while (patch_in_box < self.patches_per_bbox):
                rand_x = random.randint(x, x+w)
                rand_y = random.randint(y, y+h)

                real_x = rand_x * self.mag_factor
                real_y = rand_y * self.mag_factor

                image_proportion_percent = 0

                wsi_patch = wsi_full_size.read_region(
                    (real_x, real_y), 0, (self.patch_size, self.patch_size))
                patch_path = ''

                if bool(is_tumor):
                    assert mask_full_size is not None
                    mask_patch = mask_full_size.read_region(
                        (real_x, real_y), 0, (self.patch_size, self.patch_size))
                    mask_patch_np = np.array(mask_patch)[:, :, 0]

                    patch_path = os.path.join(
                        self.patch_pos_path, f'{img_idx}_T_{patch_idx}.PNG')
                    # check what proportion of patch is tumor
                    image_proportion_percent = (mask_patch_np.sum(axis=0).sum(
                        axis=0)/255)/(self.patch_size*self.patch_size)
                    mask_patch.close()

                else:
                    cvt_image = self.extract_tissue(cv2.cvtColor(
                        np.array(wsi_patch), cv2.COLOR_BGR2HSV))
                    patch_path = os.path.join(
                        self.patch_neg_path, f'{img_idx}_N_{patch_idx}.PNG')
                    image_proportion_percent = (cvt_image.sum(axis=0).sum(
                        axis=0)/255)/(self.patch_size*self.patch_size)

                # extract selected patch if meets criteria
                if (image_proportion_percent > self.tumor_threshold):
                    wsi_patch.save(patch_path, 'PNG')
                    patch_idx += 1
                    patch_in_box += 1
                    wsi_patch.close()
                    print(
                        f'Patch {patch_idx} from image {img_idx} was saved in {patch_path} succesfully.')
            return 0

    def get_contours(self, image, is_tumor):
        contours = None

        if bool(is_tumor):
            cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(
                np.array(cvt_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            cvt_image = self.extract_tissue(
                cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            contours, _ = cv2.findContours(
                cvt_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def get_image_region(self, slide: Union[openslide.OpenSlide, mir.MultiResolutionImage], location, level, size) :

        if type(slide) == openslide.OpenSlide:
            assert type(slide) is openslide.OpenSlide
            slide_patch = np.array(slide.read_region(location, level, size))
            slide.close()
        else:
            assert type(slide) is mir.MultiResolutionImage
            print(f'Openslide extraction failed. Trying with ASAP...')
            slide_patch = slide.getUCharPatch(  # type: ignore
                startX=location[0], startY=location[1], width=size[0], height=size[1], level=level)

        return slide_patch

    def get_mask_from_annotation(self, wsi_image, is_camelyon17: bool = False):
        # Check if there is an annotation file
        print(f'Is Camelyon17? {is_camelyon17}')
        annotation_file = wsi_image.split('.', 1)[0] + '.xml'
        annotation_path = os.path.join(self.annotation_path, annotation_file)
        if not (os.path.exists(annotation_path)):
            print(f'No annotation file found for {wsi_image}.')
            return

        reader = mir.MultiResolutionImageReader()
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)

        # adjust labels based on dataset (CAMELYON 16 or 17)
        annotation_mask = mir.AnnotationToMask()
        label_map = {'metastases': 1, 'normal': 2} if is_camelyon17 else {
            '_0': 1, '_1': 1, '_2': 0}
        conversion_order = ['metastases',
                            'normal'] if is_camelyon17 else ['_0', '_1', '_2']

        # Loading WSI associated to annotation
        slide = reader.open(os.path.join(self.image_path, wsi_image))
        xml_repository.setSource(annotation_path)
        xml_repository.load()

        # Save mask file generated from annotations
        mask_path = os.path.join(self.mask_path, wsi_image)
        annotation_mask.convert(annotation_list, mask_path, slide.getDimensions(
        ), slide.getSpacing(), label_map, conversion_order)

        print(f'Mask for {annotation_file} created successfully.')
