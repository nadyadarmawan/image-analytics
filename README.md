# image-analytics

# Purpose
This repository contains scripts for object detection dataset analysis.
  
# Getting Started
Setup by doing:
```
pip install -r requirements.txt
```

# Using this Repo
Input: 
- A directory of images,
- (Optional) Object detection annotations in COCO JSON format (refer to `demo/annotation.json`)

Output:
- image sharpness,
- image luminance,
- image contrast,
- (if annotations are given) number of objects per image,
- (if annotations are given) number of object per class.

### 1. Image Sharpness
The sharpness is measured using the variance of the Laplacian method where a lower variance is associated with a blurrier image. The image is converted into a grayscale array before this value is calculated. In many cases, an image is classified as 'blurry' if the variance of the Laplacian is below 120.

### 2. Image Luminance
The luminance, or lightness, is calculated by taking the average of luma channel of the image in YUV format. 

### 3. Image Contrast
The contrast of the image here is defined as a function of variance of all pixel brightness in the entire image array. Firstly the image array is turned into a grayscale array before being normalised using sklearn MinMaxScaler() function. Then the resulting array is scaled and the variance of the features present is presented as the contrast of the image. The possible output is range [0,1].

# Usage
## Testing
Simple test for image quality metrics:
```
python src/demo.py --images ./black_and_white
```
Running the above code will output contrast and sharpness values of 0 for both black and white images, which is expected. 
The generated luminance values are 0 for black and 255 for white, which is also expected. 

## Demo
```
python src/demo.py --images ./demo/images --labels ./demo/annotations.json
```
Running the above code will generate these charts:

Furthermore, for documentation purposes, below is the summary table that presents the images with minimum and maximum values for each measurement.
