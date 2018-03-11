# Vehicle Detection


In this project, the goal is to write a software pipeline to detect vehicles in a video.

# The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
 
* Prior to classification, Normalize  features and randomize a selection for training and testing.
* Implement a sliding-window technique and use  trained classifier to search for vehicles in images.
* Run  pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


# Files & Code Quality

These are key files: 
* [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb) (script used to setup & execute the pipeline)
* [project_video_out2.mp4](./project_video_out2.mp4) (a video recording of the vehicle detection pipeline in action)
* [README](./README.md) (this readme file has the write up for the project!)
* Training data was downloaded from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.

### 1. Read in images, experimented with Color Spaces & settled on YCbrCb space

I ran trial and error with the color spaces, tried out RGB space with HOG, HSV space with HOG and YCrCb with HOG. I found YCrCb gave +3-5% more accuracy with the SVC (Support Vector Classification) algorithm.

```
car_image = mpimg.imread(cars[car_ind])
ycrcb= cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

```

![YCrCb](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/YCrCb.png)

### 2. # Calculate HOG features
I calculated HOG features for the images, as HOG is known to provide robust image feature outlines by detecting gradient changes across the image. One key point is to make sure to grayscale the image before calculating HOG.
```
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

```
![HOG](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/hog.png)

### 3. Feature extraction after normalization
Next, I applied feature extraction on all the images, converted images into rows of image feature vectors.
Each feature vector consists of HOG features calculated on all 3 channels, Y, Cr, Cb, and then appended together to make 1 long feature vector.

```
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel='ALL',
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
```

After applying HOG, I normalized all the feature vector values by removing the mean and the deviation. This is to ensure the Machine Learning algorithm doesn't get biased towards one particular feature due to higher mean values/ higher relative standard deviation within that particular sub feature.

```
# Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
```
![Normalization](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/normalization.png)

### 3. Apply Support Vector Classification
Next, I classified the normalized data by applying SVC. I thought about applying Naive Bayes, but stuck with SVC because even though it took longer to train, the predictions seem to be much faster with SVC. For vehicle detection, seems it would be more relevant to have faster predictions.

Prior to applying classification, I split up the data set into training and validation sets, so I could measure performance on unseen data. I also randomized data selection to avoid any time series bias if training on video streams in the future.

```
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
    
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel='ALL',
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

Total samples: 17760 feature vector length (864,)
y labels length 17760
2.28 Seconds to train SVC...
Test Accuracy of SVC =  0.9786
My SVC predicts:  [ 1.  1.  0.  1.  0.  0.  1.  0.  1.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  0.  0.  1.  0.  1.  0.]
0.00159 Seconds to predict 10 labels with SVC

```
I ended up with an accuracy of 97%, which seemed reasonably good without overfitting.

### 3. Find car within image

After preparing the prediction model, i wrote a function to find cars within an image. This function combined code from 3 sources provided by Udacity: 

#1. Break up the image into 64x64 windows (size of the trained images) with the ability to scale to sub multiples of this base window size to simulate cars moving away from the vehicle. I computed HOG features for the entire image. Then, i took the window segment size , took HOG features within that window and ran a prediction using the SVC model previously trained. I then slid this fixed window around the entire image. Since most cars of interest are within the lower half of the image, i generally cropped the image to analyze to the bottom half only. On receiving a positive match for a car, I drew a rectangle at the edges of the current window and overlayed the rectangle on the original image

```
if scale != 1:
    #Resize the image according to selected scale
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
# Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

# Extract the image patch
    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

# Run prediction with SVC
test_prediction = svc.predict(test_features)
            
    if test_prediction == 1 :
# Draw a rectangle around the current window
cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
# Add rectangle to list of rectangles
bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

```
![Normalization](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/sliding_window.png)

### 4. base pipeline

I created a pipeline to find prediction rectangles across the image at various scales, this also simulated somewhat overlapping windows.

```
# Build a pipeline to test basic rectangle detection
def pipeline(test_img):

	ystart = 400
    ystop = 500
    scale = 4.0
    img,rectangles = find_cars(test_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)        
    all_rectangles.append(rectangles)

    ystart = 400
    ystop = 500
    scale = 3.0
    img,rectangles = find_cars(test_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)        
    all_rectangles.append(rectangles)

    ...

```
I tested the pipeline on various test images
![Normalization](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/pipeline_test_basic.png)

### 4. Filter pipeline with heatmap

To get rid of False positives (rectangles in the middle of the road), I added in a Heatmap scheme to keep rectangles only where multiple rectangles were detected at multiple scales. Basic code was provided by Udacity, I selected a filter value of 4 [at least 4 detections/confirmations in a rectangle required else that rectangle was discarded]. I aso used the scipy label function to find group overlapping rectangles into 1 large rectangle.

```
for rect in rects:
    heatmap_img = add_heat(heatmap_img, rect)
heatmap_img = apply_threshold(heatmap_img, 4)

```

Heatmap

![Heatmap](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/heatmap1.png)

Threshold

![Threshold](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/heatmap_threshold.png)


Label

![label](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/label.png)


Bounding Box

![label](https://raw.githubusercontent.com/eshnil2000/CarND-Vehicle-Detection/master/result_images/bounding_box.png)






### 5. Final pipeline

I assembled all the steps above in a single pipeline, to feed the video into.

```
# Build the final pipeline which combines the basic rectangle detection pipeline with the heatmap functionality
def pipeline_final(test_img):
    out,rects=pipeline(test_img)
    
    heatmap_img = np.zeros_like(test_img[:,:,0])
    for rect in rects:
        heatmap_img = add_heat(heatmap_img, rect)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    draw_img, rect = draw_labeled_bboxes(np.copy(test_img), labels)

    return draw_img
```

```
#Run the video through the final pipeline
test_out_file = 'project_video_out2.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(pipeline_final)
%time clip_test_out.write_videofile(test_out_file, audio=False)

```

### 5. Reflections

The final pipeline performs reasonably well, but there is shakiness, false positives and rectangle drops in some frames (false negatives). As a future improvement, I could try several approaches:
1. A more sophisticated classification algorithm (using more features), and certainly playing with some of the SVC parameters like Alpha etc which I did not tune in this exercise.
2. More training data. Given the performance of the pipeline, it would certainly be beneficial to train on more images with empty roads to get rid of some of the noisy outliers. Similarly, there are patches of dark road where the white car was not detected, training the model under varying conditions would also help, where the HOG gradients may be made more sensitive.
3. A more comprehensive sliding window algorthm may help to smooth out some of the abrupt fluctuations
4. Maintaining history between frames may provide for a more stable rectangle detection by filtering out rectangles that were not detected in previous 10-15 frames or so.
 
