# Face segmentation with CNN and CRF

We try different methods to complete face segmentation:
- [A CNN Cascade for Landmark Guided Semantic Part Segmentation](https://arxiv.org/pdf/1609.09642.pdf). Models and more details please refer to Aaron Jackson's [website](https://aaronsplace.co.uk/papers/jackson2016guided/index.html). We add CRF as postprocessing. CRF is implemented by [pydensecrf](https://github.com/lucasb-eyer/pydensecrf). 
- [On Face Segmentation, Face Swapping, and Face Perception](https://arxiv.org/abs/1704.06729). Original [codes and models](https://github.com/YuvalNirkin/face_segmentation).
- Generate convex hull according to landmarks.

Before using all three methods we detect landmarks and crop the image. Instead of using landmarks detection network in `A CNN Cascade for Landmark Guided Semantic Part Segmentation` we use [2D-FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) to detect landmarks which works very well on large pose images. We also try different methods to crop the image.


## Codes
- face_segment_part.py: A CNN Cascade for Landmark Guided Semantic Part Segmentation.
- face_segment_yuval.py: On Face Segmentation, Face Swapping, and Face Perception.
- face_segment_contour.py: Detected landmarks and get convex hull.


## Dependencies
Please download [caffe](http://caffe.berkeleyvision.org/)(minimum version: 1.0) for face_segment_yuval.py, download [caffe-future](http://aaronsplace.co.uk/papers/jackson2016guided/index.html) for face_segment_part.py

Then run the following command in a terminal:
```
pip install -r requirements.txt
```


## Results
- face_segment_part:

No crf:  
<img src="images/image04250_part_nocrf_min.png" width="500">  
Add crf:  
<img src="images/image04250_part_crf_min.png" width="500">  
No crf:  
<img src="images/image00091_part_nocrf_min.png" width="500">  
Add crf:  
<img src="images/image00091_part_crf_min.png" width="500">

- face_segment_yuval:

Model face_seg_fcn8s(size:500x500):  
<img src="images/image04250_yuval_min.png" width="500">  
<img src="images/image00091_yuval_min.png" width="500">  
Model face_seg_fcn8s_300_no_aug(size:300x300):  
<img src="images/image04250_yuval_300_nocrf_min.png" width="500">  
<img src="images/image00091_yuval_300_nocrf_min.png" width="500">  
Add crf to second model:  
<img src="images/image04250_yuval_300_crf_min.png" width="500">  
<img src="images/image00091_yuval_300_crf_min.png" width="500">  

- face_segment_contour:
<img src="images/image04250_contour_nocrf.png" width="500">  
<img src="images/image00091_contour_nocrf.png" width="500">  
Add crf:  
<img src="images/image04250_contour_crf.png" width="500">  
<img src="images/image00091_contour_crf.png" width="500">


## TODO
- [ ] Modify probability computation in face_segment_yuval.py
- [ ] Add speed test
- [x] Add landmark detection
- [x] Add image cropping
- [x] Contour -> segment
- [x] Unify code style
- [x] Improve CRF results
- [x] Add example images to README
- [x] Compare results of all methods
