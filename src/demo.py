# This script presents the statistics generated using analytics_generator.py
# as barcharts and boxplots.
# Takes path to the directory of images and path to COCO JSON file (optional) as inputs.

# Nadya D. 2023

from matplotlib import pyplot as plt
import argparse
import json
import os
import math
import numpy as np

from analytics_generator import ImageMetricsGenerator, get_objects_stats


def parse_args():
    # This function parses command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', required=True,
                    help='Path to the parent directory of images')
    ap.add_argument('-j', '--labels', required=False,
                    help='Path to coco_labels.json')

    args = vars(ap.parse_args())
    return args


def add_bar_labels(x, y, total_num):
    '''
    This function adds data labels to a barchart.
    Takes arrays of x and y values, and
    the total number of observations as inputs.
    '''
    for i in range(len(x)):
        pctg = str(round(100*int(y[i])/total_num, 1))+'%'
        plt.text(i, y[i], pctg, ha = 'center')


def compare_min_max(min_info, max_info, value, image_name):
    '''
    This function compares the current (stored) minimum and maximum values
    with the 'new' value of interest.
    Returns updated minimum and maximum info
    of values and the associated image path.
    '''
    if value > max_info[0]:
        max_info[0] = value
        max_info[1] = image_name
    if value < min_info[0]:
        min_info[0] = value
        min_info[1] = image_name

    return min_info, max_info


def main():
    args = parse_args()
    img_dir = args['images']
    
    sharpness_values = []
    contrast_values = []
    luminance_values = []
    min_sharpness = [math.inf, '']
    min_contrast = [math.inf, '']
    min_lum = [math.inf, '']
    max_sharpness = [-math.inf, '']
    max_contrast = [-math.inf, '']
    max_lum = [-math.inf, '']

    # Loop through the directory of images
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            img_path = os.path.join(root, file)
            # Get image quality metrics
            metrics_generator = ImageMetricsGenerator(img_path)
            img_metrics_dict = metrics_generator.get_img_metrics()
            if not img_metrics_dict:
                continue
            sharpness_values.append(img_metrics_dict['sharpness'])
            contrast_values.append(img_metrics_dict['contrast'])
            luminance_values.append(img_metrics_dict['luminance'])

            min_sharpness, max_sharpness = compare_min_max(min_sharpness, max_sharpness, img_metrics_dict['sharpness'], img_path)
            min_contrast, max_contrast = compare_min_max(min_contrast, max_contrast, img_metrics_dict['contrast'], img_path)
            min_lum, max_lum = compare_min_max(min_lum, max_lum, img_metrics_dict['luminance'], img_path)

    # Plot image quality metrics distributions
    plt.figure(figsize=(12,12))
    plt.suptitle('Image Quality Distributions', size='x-large')
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
    plt.text(-0.1, 0.8, '- Minimum Luminance: ' + min_lum[1])
    plt.text(-0.1, 0.7, '- Maximum Luminance: ' + max_lum[1])
    plt.text(-0.1, 0.6, '- Minimum Contrast: ' + min_contrast[1])
    plt.text(-0.1, 0.5, '- Maximum Contrast: ' + max_contrast[1])
    plt.text(-0.1, 0.4, '- Minimum Sharpness: ' + min_sharpness[1])
    plt.text(-0.1, 0.3, '- Maximum Sharpness: ' + max_sharpness[1])
    plt.grid(False)
    plt.axis('off')

    plt.savefig('images_analysis.png')
    print("Charts for image quality metrics saved as 'images_analysis.png'")

    # Analyse annotations if given as an input
    if args['labels']:
        labels = args['labels']
        # Read JSON file
        f = open(labels)
        labels_data = json.load(f)
        # Get objects statistics
        object_stats_dict = get_objects_stats(labels_data)
        ordered_num_per_img = {key: object_stats_dict['num_per_img'][key] for key in sorted(object_stats_dict['num_per_img'].keys())}
        n_imgs = object_stats_dict['num_of_imgs']
        n_objects =  object_stats_dict['num_of_objects']

        # Plot annotations statistics
        plt.figure(figsize=(16,9))
        plt.suptitle('Annotations Distributions', size='x-large')
        plt.subplot(1,2,1)
        plt.title('Number of Objects per Image')
        plt.bar(list(ordered_num_per_img.keys()), list(ordered_num_per_img.values()), color ='tab:orange', width =0.4)
        plt.xticks(np.arange(len(list(object_stats_dict['num_per_img'].keys()))))
        add_bar_labels(list(ordered_num_per_img.keys()), list(ordered_num_per_img.values()), n_imgs)
        plt.xlabel('Number of objects per image')
        plt.ylabel('Number of images')

        plt.subplot(1,2,2)
        plt.title('Number of Objects per Class')
        plt.bar(list(object_stats_dict['num_per_class'].keys()), list(object_stats_dict['num_per_class'].values()), color ='tab:purple', width =0.4)
        add_bar_labels(list(object_stats_dict['num_per_class'].keys()), list(object_stats_dict['num_per_class'].values()), n_objects)
        plt.xlabel('Class')
        plt.ylabel('Number of objects')

        plt.savefig('annotations_analysis.png')
        print("Charts for annotation distributions saved as 'annotations_analysis.png'")


if __name__ == '__main__':
    main()
