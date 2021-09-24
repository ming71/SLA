from .dataset import DetDataset

NAMES = ['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                     'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court']


class DOTA(DetDataset):
    def __init__(self, root, image_sets, aug=None):
        super(DOTA, self).__init__(root, image_sets, NAMES, aug)
