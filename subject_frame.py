'''
Identify subject in image or photo
Reduce subject into a frame made up of points and line segments
Points are located at areas designated as points of interest (POI)
Line segments are used to connect POIs
'''
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
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

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
    dataset_val.load_balloon(args.dataset, "val")
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

def conv_to_frame(model):
    pass

if __name__ == '__main__':
    import argparse as ap
    '''
    TODO:
    add command for path to logs and dataset
    use assert in if conditions to make sure necessary fields are provided based on command
    '''
    #parse command line arguments
    parser = ap.ArgumentParser(description='Simplifies subject into a simple frame consisting of points and line segments')

    parser.add_argument('command', required=True, metavar='<command>', help="'train' or 'frame'")
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

    args = parser.parse_args()
    
    #Check command line arguments
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == 'frame':
        pass

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    #configurations
    config = image_configs.SubjectConfig() if args.command == 'train' else image_configs.InferenceConfig()
    config.display()

    #create model
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
        conv_to_frame(model)