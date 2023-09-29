# This script lists the functions for image and annotations analytics.
# Nadya D. 2023


import math
import cv2
import numpy as np
import logging


def get_objects_stats(coco_json_data):
    '''
    This function generates basic COCO annotations statistics:
    (1) number of objects per image and 
    (2) number of objects per class.
    Takes COCO JSON data as input and 
    returns (1), (2), number of images in the JSON data, 
    and number of annotated objects.
    '''

    objects_count = {}
    num_objects_per_img = {}
    num_objects_per_class = {}
    num_of_objects = 0

    for i in coco_json_data['images']:
        annotations = list(filter(lambda a: a['image_id'] == i['id'], coco_json_data['annotations']))
        objects_count[i['file_name']] = len(annotations)
        num_of_objects += len(annotations)
        for a in annotations:
            c = list(filter(lambda cat: cat['id'] == a['category_id'], coco_json_data['categories']))[0]['name']
            if c not in num_objects_per_class:
                num_objects_per_class[c] = 0
            num_objects_per_class[c] += 1
    for f in objects_count: 
        if objects_count[f]>=10:
            n = '>=10'
        else:
            n = str(objects_count[f])
        if n not in num_objects_per_img:
            num_objects_per_img[n] = 0
        num_objects_per_img[n] += 1

    return {'num_per_img': num_objects_per_img,
            'num_per_class': num_objects_per_class,
            'num_of_imgs': len(objects_count),
            'num_of_objects': num_of_objects}


class ImageMetricsGenerator():
    '''
    This class is used to generate basic image quality metrics.
    
    Attributes
    ----------
    image_path: str
        the filename of the image.
    bgr_arr: NumPy array
        the image file opened as an array in BGR colour format.
    grey_arr: NumPy array
        the image file opened as an array in greyscale format.
    yuv_arr: NumPy array
        the image file opened as an array in YUV colour format.

    Methods
    -------
    get_sharpness()
        Returns tne sharpness of the image.
    get_contrast()
        Returns the contrast of the image.
    get_luminance():
        Returns the luminance of the image.
    get_img_metrics():
        Returns a dictionary of the three image quality metrics.
    '''

    @classmethod 
    def __init__(self, image_path):
        self.logger = logging.getLogger()
        self.img_path = image_path

        try:
            image_array = cv2.imread(image_path)
        except Exception:
            self.logger.exception(f'Failed to open file: {self.img_path}')

        try: 
            image_shape = image_array.shape
        except AttributeError:
            self.logger.exception(f'Failed to read {self.img_path} as an image.')
            return

        if len(image_shape) == 3:
            grey_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            bgr_img = image_array
        else:
            grey_img = image_array
            bgr_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        self.bgr_arr = bgr_img
        self.grey_arr = grey_img
        self.yuv_arr = yuv_img


    def get_sharpness(self):
        # Returns tne sharpness of the image.
        sharpness = math.nan
        try:
            sharpness = round(float(cv2.Laplacian(self.grey_arr, cv2.CV_64F).var()), 2)
        except Exception:
            self.logger.exception(f'Failed to compute blurriness value for {self.img_path}')
        return sharpness


    def get_contrast(self):
        # Returns the contrast of the image.
        contrast = math.nan
        try:
            contrast = round(float(self.grey_arr.std()), 4)
        except Exception:
            self.logger.exception(f'Failed to compute contrast value for {self.img_path}')
        return contrast


    def get_luminance(self):
        # Returns the luminance of the image.
        img_luminance = math.nan
        try:
            y, u, v = cv2.split(self.yuv_arr)
            img_luminance = np.average(y)
        except Exception:
            self.logger.exception(f'Failed to compute luminance value for {self.img_path}')
        return img_luminance


    def get_img_metrics(self):
        # Returns a dictionary of the three image quality metrics.
        sharpness = self.get_sharpness()
        contrast = self.get_contrast()
        img_luminance = self.get_luminance()
        return {'luminance': img_luminance, 'contrast': contrast, 'sharpness': sharpness}
