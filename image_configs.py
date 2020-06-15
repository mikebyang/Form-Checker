from mrcnn import config

class SubjectConfig(config.Config):
    NAME = 'coco'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 #background and person(s)
    
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