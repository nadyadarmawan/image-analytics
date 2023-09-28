import math
import cv2
import numpy as np
import logging


def get_objects_stats(coco_json_data):
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
    @classmethod 
    def __init__(self, image_name, image_array):
        self.logger = logging.getLogger()
        self.img_name = image_name
        
        if len(image_array.shape) == 3:
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
        sharpness = math.nan
        try:
            sharpness = round(float(cv2.Laplacian(self.grey_arr, cv2.CV_64F).var()), 2)
        except Exception:
            self.logger.exception(f'Failed to compute blurriness value for {self.img_name}')
        return sharpness


    def get_contrast(self):
        contrast = math.nan
        try:
            contrast = round(float(self.grey_arr.std()), 4)
        except Exception:
            self.logger.exception(f'Failed to compute contrast value for {self.img_name}')
        return contrast


    def get_img_luminance(self):
        img_luminance = math.nan
        try:
            y, u, v = cv2.split(self.yuv_arr)
            img_luminance = np.average(y)
        except Exception:
            self.logger.exception(f'Failed to compute luminance value for {self.img_name}')
        return img_luminance


    def get_img_metrics(self):
        sharpness = self.get_sharpness()
        contrast = self.get_contrast()
        img_luminance = self.get_img_luminance()        
        return {'luminance': img_luminance, 'contrast': contrast, 'sharpness': sharpness}
