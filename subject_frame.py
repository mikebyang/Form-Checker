'''
Identify subject in image or photo
Reduce subject into a frame made up of points and line segments
Points are located at areas designated as points of interest (POI)
Line segments are used to connect POIs
'''

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os

#project root directory
ROOT_DIR = os.path.abspath('./')

import warnings
warnings.filterwarnings('ignore')

from mrcnn import utils, model as modellib
from mrcnn import visualize

#path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,'mask_rcnn_coco.h5')

#logs and trained models directory
LOG_DIR = os.path.join(ROOT_DIR,'logs')

#directory containing images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, 'poi_test_images')

import image_configs
from train_datasets import People_Dataset
from train_datasets import POI_Dataset

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = People_Dataset()
    dataset_train.load_People(args.dataset, 'train')
    dataset_train.prepare()
    # Validation dataset
    dataset_val = People_Dataset()
    dataset_val.load_People(args.dataset, "val")
    dataset_val.prepare()
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')



    # dataset_train = POI_Dataset()
    # dataset_train.load_POIs(args.dataset, 'train')
    # dataset_train.prepare()

import datetime
import numpy as np
import skimage

def detectPerson(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def conv_to_frame(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = detectPerson(image, r['masks'])
        # Save output
        file_name = os.path.join(ROOT_DIR,"result","splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = os.path.join(ROOT_DIR,"result","splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = detectPerson(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    # image = skimage.io.imread("./training_data_people/train/4965080266_a1fb538be2_o.jpg")
    # skimage.io.imshow(image)
    # skimage.io.show()

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse as ap
    '''
    TODO:
    add command for path to logs and dataset
    use assert in if conditions to make sure necessary fields are provided based on command
    '''
    #parse command line arguments
    parser = ap.ArgumentParser(description='Simplifies subject into a simple frame consisting of points and line segments')

    parser.add_argument('command', metavar='<command>', help="'train' or 'frame'")
    parser.add_argument('--dataset', required=False,
                        action='store', nargs='?',
                        default=IMAGE_DIR,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=False,
                        action='store', nargs='?',
                        default=COCO_WEIGHTS_PATH,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'",)
    parser.add_argument('--logs', required=False,
                        action='store', nargs='?',
                        default=LOG_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')

    args = parser.parse_args()
    
    # Validate arguments
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == 'frame':
        assert args.image or args.video,\
            "Image or video is needed to for frame"

    #check if logs directory exists
    if not os.path.exists(args.logs):
        # no logs might mean no trained model to be used -> so maybe throw error?
        os.mkdir(args.logs)

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = image_configs.SubjectConfig() if args.command == 'train' else image_configs.InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs) if args.command == 'train' else \
        modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    
    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        conv_to_frame(model, image_path=args.image, video_path=args.video)