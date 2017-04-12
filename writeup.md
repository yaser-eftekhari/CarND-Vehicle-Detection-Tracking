# Vehicle Detection Project

## Overview
The goal of this project is to create a pipeline that can detect and ideally track vehicles in a video stream using Linear SVM as the machine learning classifier.

To achieve this goal, the classifier was created and trained offline and the result saved in a pickle file for later use. The pipeline then loads the classifier and utilizes that in almost real-time.

Labeled data used for training the classifier are a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).

The pipeline works as follows:

* Find the region of interest in the image (lower half of the image)
* Transform the image to YUV color space
* Find the Histogram of Oriented Gradients (HOG) features for the image
* Using a sliding window over the image use the following features for the classifier:
  * Sample Histogram of Oriented Gradients (HOG) features pertaining to the window of interest
  * Binned color features
  * Histograms of color
* Normalize the three accumulated features
* Use the trained classifier to search for vehicles in the window of interest
* For windows containing a vehicle, create a heat map
* Average heatmaps created over frames, threshold the averaged heatmap to reject ourliers
* Using lable technique, group detected cars and find the bounding boxes for display purposes

## Tour of files included
To make finding codes easier in this project, I decided to include this section and briefly describe what each python file does:

- `proj5_util.py`: This file includes all the necessary utility functions required for this project. It includes:
  - `History` class is used to keep track of detection history over frames of the video
  - `find_cars` function operated on an input image, converts it to YUV color space, finds HOG over the entire image, performs a sliding window search over the image, finds all features of interest, performs classification and returns all windows containing cars
  - `get_hog_features` function finds the HOG features of the input image
  - `bin_spatial` function returns the spatial features of the image
  - `color_hist` function returns the histogram of color features of the image
  - `single_img_features` function returns all desired features of the input image. This is useful when using `search_windows` function.
  - `search_windows` function receives windows of interest and the image and returns all windows containing a vehicle. This function should be used as a replacement for `find_cars` function
  - `extract_features` function is used when training the classifier. It is very much the same as `single_img_features` function with the difference that this function operates on a list of filenames. So it reads all files and creates a list of features for each file/image.
  - `draw_boxes` function is used to draw boundary boxes on images
- `make_model.py`: This file is used to create a Linear SVM and train it. The trained classifier is saved as a pickle file for later use.
- `run.py`: This file is the main entry point to the project.

[//]: # (Image References)
[car_not_car]: ./output_images/car_not_car.png
[HOG_example]: ./output_images/HOG_example.png
[sliding_windows]: ./output_images/sliding_windows.png
[pipeline_output]: ./output_images/pipeline_output.png
[bboxes_and_heat]: ./output_images/bboxes_and_heat.png
[labels_boxes]: ./output_images/labels_boxes.png
[video1]: ./project_video.mp4

[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---

### Writeup / README

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Training the classifier is done in file `make_model.py`. I started by reading in all the `vehicle` and `non-vehicle` images from the dataset. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car and Not-Car Samples][car_not_car]

The code for this step is contained in function `get_hog_features` in lines 156 through 167 of the file called `proj5_util.py`. This function is used in function `extract_features` in the same file to extract all features from all car or non-car images.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG example for car and non-car images][HOG_example]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I used the reduced set of vehicle and non-vehicle classes provided in the lessons to see what parameters perform well while having a reasonable performance.

I also consulted the forum to see suggested color spaces and parameters. Finally, I settled with the following ones:
```
color_space = 'YUV'
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Perform HOG on all channels
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, spatial image features and histogram of color features. As discussed before, `extract_features` function was used with parameters:

```
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 128    # Number of histogram bins
spatial_feat = True # Spatial features on
hist_feat = True # Histogram features on
hog_feat = True # HOG features on
```

Lines 35 through 47 `make_model.py` shows how to extract features from car and non-car data. Lines 48-56 shows how to combine extracted features and the normalization. Lines 59-70 show how to prepare the training and test data (20% split for test data) as well as creating and training the classifier.

---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented as part of `find_cars` function in `proj5_util.py` file. The main loop is in lines 113-151. This is basically using the HOG sub-sampling approach to save on calculating HOG features for overlapping windows.

Two search scales were used: 1 and 1.5. They translate to window sizes of 64x64 and 96x96, respectively. Figure below shows the complete sliding window for these scales. For better display of the windows, the lower images show 4 overlapping windows for each scale. These two scales led to better tradeoff between speed and accuracy of detection.

![Sliding window search][sliding_windows]

As can be seen only the lower right quadrant of the image is searched for cars. This was done to improve the speed of the pipeline. Realistically, the lower half has to be search with left and right side of the car.

The overlap of 1 cell (about 70%) was also decided based on trial and error. Again, this was a good compromise between speed and accuracy.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (1 and 1.5) using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![Pipeline output examples][pipeline_output]

As discussed before, the optimization happened heuristically by changing the parameters for the features (histogram bin sizes, image shape, cell sizes, etc.). However, to improve the speed of the pipeline, the complete sweep of the frame is done only 1 in every 5 frames. For the other 4 frames, the detected boundaries are only checked to see if a car is present or not. In case a car was detected in the same spot, the boundaries were drawn as normal.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (lines 147-151 in `find_cars` function). From the positive detections I created a heatmap and added the heatmap for the current frame to a list of heatmaps (lines 27-52 `History` class). The list keeps only the last `n` heatmaps. The average and weighted average of all heatmaps was calculated and thresholded to identify vehicle positions (lines 36-38 `History` class). I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (lines 61 same class). I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap and the bounding boxes for a series of 5 frames of the video. Note that the heatmap has an inherent averaging from all previous frames built into it.

![Boundaries and heatmaps][bboxes_and_heat]

Here is the output of `scipy.ndimage.measurements.label()` and the resulting resulting bounding boxes:

![Lable and boundaries][labels_boxes]

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- First issue is the speed of the pipeline. It is currently at about 7 frames/second, which is not bad but far from ideal. After doing some timing analysis, the main loop for searching car in the frame found to be the bottleneck. In order to solve the issue, an adaptive search should be implemented where the scale is low for parts of the frame closer to the horizon and increase as getting to the bottom of the frame.

- The data set had many duplicates for cars. Since all similar pictures are in a series, ideally one would sample the dataset every 5 or 6 pictures to get just one or two of each image.

- There is no vehicle tracking specifically. It would be nice to implement an algorithm that can track objects through frames. This can clearly be seen when the two cars are close to each other; the pipeline detects two cars as one. Although there are limitations to what a camera can see (and possibly one of the cars changing direction) but the algorithm should track both cars to some extent.

- Another method to detect cars from heatmaps is to use `scipy.ndimage.filters.maximum_filter` filter. The local maximum peaks can also be used to create an adaptive thresholding to reduce the false positives.

- Utilizing the general SVM with linear kernel will also enable us to use detection probability along the actual classification label. That would also give us a great threshold for removing false positives from detections.

- Using deep neural networks instead of a machine learning classifier should also result in a more robust detection.
