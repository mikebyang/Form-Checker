# Mask R-CNN for Checking Form
## Introduction

[Mask-RCNN](https://github.com/matterport/Mask_RCNN) provides a method of object instance segmentation. The approach detects an object and provides a segmentation mask which covers the object.

## Description

This project is aimed at producing some application which is able to use artificial intelligence to monitor the form of a subject performing an exercise. Application is supposed to be able to indicate the moments where the subject is showing improper form if it should occur.

Program reduces the subject in a video given to a frame made up of points and line segments. Points will be place at locations designated as points of interest (POI). The head, neck, shoulders, chest, elbow, wrist, hands, core, pelvis, knee, sole of feet, and front of foot were used for this project for a total of 12 POIs. Later versions may use other POIs or add more POIs to the list of parts to monitor while performing certain exercises.

Subject's frame is compared to a trained model trained using the frame of athletes who are able to perform these exercises. The program is supposed to alert the user if the subject's frame is differing from the athletes frame by some hard-coded tolerance for each exercise.

## Implementation

This project is build upon the code used in the guide "Splash of Color" listed in resources below. The original code has been rearranged slightly in a manner based on personal preference.

## Resources

These are readings and sources that were used to develop this project.

### Readings

Mask R-CNN:
* [Splash of Color](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) was the step-by-step guide used to figure out how to train a custom model using Mask R-CNN. This guide also helped to provide some of the resources listed here.
* [R-CNN (pdf)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=AF8817DD0F70B32AA08B2ECBBA8099FA?doi=10.1.1.715.2453&rep=rep1&type=pdf)
* [Fast R-CNN](https://arxiv.org/abs/1504.08083)
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
* [Mask R-CNN](https://arxiv.org/abs/1703.06870)
* [Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation](https://arxiv.org/abs/1502.02734)
* [From Image-level to Pixel-level Labeling with Convolutional Networks](https://arxiv.org/abs/1411.6228)

Motion Tracking:
* [Content Based Video Retrieval Systems](https://arxiv.org/ftp/arxiv/papers/1205/1205.1641.pdf)

### Tools and Sources

* [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
* [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) was used to create annotations for each image.
* [flickr](https://www.flickr.com/) was used to get pictures for training and validation.
* [Reddit](https://www.reddit.com/)