from .dataset import DetDataset

NAMES = ['car', 'airplane']

class UCAS_AOD(DetDataset):
    def __init__(self, root, image_sets, aug=None):
        super(UCAS_AOD, self).__init__(root, image_sets, NAMES, aug)
