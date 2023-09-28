from matplotlib import pyplot as plt
import argparse
import json
import os
import math
import statistics
import numpy as np
import cv2

from analytics_generator import ImageMetricsGenerator, get_objects_stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', required=True,
                    help='Path to the parent directory of images')
    ap.add_argument('-j', '--labels', required=False,
                    help='Path to coco_labels.json')

    args = vars(ap.parse_args())
    return args


def add_bar_labels(x, y, total_num):
    for i in range(len(x)):
        pctg = str(round(100*int(y[i])/total_num, 1))+'%'
        plt.text(i, y[i], pctg, ha = 'center')


def compare_min_max(min_value, max_value, value):
    update_min_img = False
    update_max_img = False
    if value > max_value:
        max_value = value
        update_max_img = True
    if value < min_value:
        min_value = value
        update_min_img = True
    
    return min_value, max_value, update_min_img, update_max_img


def main():
    args = parse_args()
    img_dir = args['images']
    
    sharpness_values = []
    contrast_values = []
    luminance_values = []
    min_sharpness = min_contrast = min_lum = math.inf
    max_sharpness = max_contrast = max_lum = 0
    min_sharpness_img = max_sharpness_img = min_contrast_img = max_contrast_img = min_lum_img = max_lum_img = ''

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            metrics_generator = ImageMetricsGenerator(file, image)
            img_metrics_dict = metrics_generator.get_img_metrics()
            sharpness_values.append(img_metrics_dict['sharpness'])
            contrast_values.append(img_metrics_dict['contrast'])
            luminance_values.append(img_metrics_dict['luminance'])
            print(file, img_metrics_dict['luminance'])
            min_sharpness, max_sharpness, update_min, update_max = compare_min_max(min_sharpness, max_sharpness, img_metrics_dict['sharpness'])
            if update_min: 
                min_sharpness_img = img_path
            if update_max:
                max_sharpness_img = img_path
            min_contrast, max_contrast, update_min, update_max = compare_min_max(min_contrast, max_contrast, img_metrics_dict['contrast'])
            if update_min: 
                min_contrast_img = img_path
            if update_max:
                max_contrast_img = img_path
            min_lum, max_lum, update_min, update_max = compare_min_max(min_lum, max_lum, img_metrics_dict['luminance'])
            if update_min: 
                min_lum_img = img_path
            if update_max:
                max_lum_img = img_path

    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.title('Luminance Distribution')
    plt.boxplot(luminance_values, vert=False)
    
    plt.subplot(2,2,2)
    plt.title('Contrast Distribution')
    plt.boxplot(contrast_values, vert=False)
    
    plt.subplot(2,2,3)
    plt.title('Sharpness Distribution')
    plt.boxplot(sharpness_values, vert=False)

    plt.subplot(2,2,4)
    plt.text(-0.1, 0.9, 'Images with:')
    plt.text(-0.1, 0.8, '- Minimum Luminance: ' + min_lum_img)
    plt.text(-0.1, 0.7, '- Maximum Luminance: ' + max_lum_img)
    plt.text(-0.1, 0.6, '- Minimum Contrast: ' + min_contrast_img)
    plt.text(-0.1, 0.5, '- Maximum Contrast: ' + max_contrast_img)
    plt.text(-0.1, 0.4, '- Minimum Sharpness: ' + min_sharpness_img)
    plt.text(-0.1, 0.3, '- Maximum Sharpness: ' + max_sharpness_img)
    plt.grid(False)
    plt.axis('off')

    plt.savefig('images_analysis.png')

    if args['labels']:
        labels = args['labels']
        f = open(labels)
        labels_data = json.load(f)
        object_stats_dict = get_objects_stats(labels_data)
        ordered_num_per_img = {key: object_stats_dict['num_per_img'][key] for key in sorted(object_stats_dict['num_per_img'].keys())}

        plt.figure(figsize=(16,9))
        plt.subplot(1,2,1)
        plt.title('Number of Objects per Image')        
        plt.bar(list(ordered_num_per_img.keys()), list(ordered_num_per_img.values()), color ='tab:orange', width =0.4)
        plt.xticks(np.arange(len(list(object_stats_dict['num_per_img'].keys()))))
        add_bar_labels(list(ordered_num_per_img.keys()), list(ordered_num_per_img.values()), object_stats_dict['num_of_imgs'])
        plt.xlabel('Number of objects per image')
        plt.ylabel('Number of images')

        plt.subplot(1,2,2)
        plt.title('Number of Objects per Class')
        plt.bar(list(object_stats_dict['num_per_class'].keys()), list(object_stats_dict['num_per_class'].values()), color ='tab:purple', width =0.4)
        add_bar_labels(list(object_stats_dict['num_per_class'].keys()), list(object_stats_dict['num_per_class'].values()), object_stats_dict['num_of_objects'])
        plt.xlabel('Class')
        plt.ylabel('Number of objects')

        plt.savefig('annotations_analysis.png')


if __name__ == '__main__':
    main()