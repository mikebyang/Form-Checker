from mrcnn import utils

class People_Dataset(utils.Dataset):
    def load_People(self, dataset_dir, subset):
        assert subset in ["train", "val"]

    def load_mask(self, image_id):
        pass
    def image_reference(self, image_id):
        pass

class POI_Dataset(utils.Dataset):
    def load_POIs(self, dataset_dir, subset):
        pass
    def load_mask(self, image_id):
        pass
    def image_reference(self, image_id):
        pass