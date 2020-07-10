from mrcnn import config

############################################################
#  Configurations
############################################################

class SubjectConfig(config.Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'subject'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + person(s)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
class InferenceConfig(SubjectConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class POI_Config(config.Config):
    NAME = 'Points of Interest'
    '''
    Points of interest: 12
    Head, neck, shoulders, chest, elbow, wrist, hands, core, pelvis, knee, sole of feet, front of foot
    '''
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12 #background and POIs